"""Batch/sync adjudication for layer0 review queue."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_QUEUE = _ROOT / "data" / "To_run" / "00_filtering" / "layer0_human_review_queue.csv"
_DEFAULT_DB = _ROOT / "data" / "raw" / "ukrpoetry_database.csv"
_DEFAULT_WORKDIR = _ROOT / "data" / "To_run" / "00_filtering" / "batch_adjudication"

_TEXT_COL = "Poem full text (copy and paste)"

_DEFAULT_MAX_POEM_CHARS = 12000

_SYSTEM = (
    "You adjudicate ONE Facebook poetry post. The database claims `expected` distinct poems; "
    "a rule-based splitter produced `got` segments.\n"
    "Return exactly ONE JSON object (no markdown, no prose):\n"
    '{"parent_id":"","verdict":"annotation_correct|annotation_wrong|ambiguous",'
    '"true_poem_count":0,"annotation_action":"keep|correct_to_1|cannot_decide",'
    '"confidence":"high|medium|low","use_manual_override":false,'
    '"line_starts_zero_based":null,"rationale_short":"","notes":null}\n'
    "Definitions: a poem = a distinct verse composition. Repeated refrain / closing echo = usually ONE poem. "
    "Editorial intro without a second full poem = metadata often wrong.\n"
    "true_poem_count: your best count of DISTINCT poems. It MAY differ from DB `expected` when metadata is wrong.\n"
    "line_starts_zero_based: ONLY when verdict is annotation_correct AND true_poem_count>=2: "
    "pipe-separated 0-based line indices of the FIRST line of EACH poem, MUST start with 0, "
    "exactly true_poem_count values. If you cannot list valid cuts, use verdict ambiguous or lower true_poem_count; "
    "do not leave line_starts null while claiming annotation_correct with true_poem_count>=2.\n"
    "If the user message says TRUNCATED, be conservative about mid-poem cuts."
)

_SYSTEM_RETRY_SUFFIX = (
    "\nROUND2: A prior adjudication+rules pass STILL failed QC alignment. "
    "If the DB over-counts, set annotation_wrong with a lower true_poem_count. "
    "If the count is right, you must either output valid line_starts_zero_based matching true_poem_count "
    "(for >=2 poems) or use ambiguous — avoid annotation_correct with >=2 poems and null line_starts."
)

_SYSTEM_RUN_SYNC_EXTRA = (
    "\nLINE_INDEX_QC (mandatory): Indices in line_starts_zero_based refer to FULL_TEXT_LINES_BELOW "
    "split on \\n only, 0-based, counting from 0 up to lines_total-1 (use the same lines_total as in "
    "the JSON metadata block). You MUST output exactly true_poem_count indices, the first MUST be 0, "
    "each index MUST satisfy 0 <= index < lines_total (never use lines_total itself as a start). "
    "Each poem segment (from one start to the next, or to EOF) MUST contain at least one non-space "
    "character—do not start a poem on a blank-only line. "
    "Whenever true_poem_count >= 2, line_starts_zero_based MUST be a non-null pipe-separated list "
    "that satisfies the above, including for verdict annotation_wrong if you still claim multiple poems. "
    "If you cannot, lower true_poem_count or use verdict ambiguous."
)


def _sanitize_custom_id(parent_id: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(parent_id).strip())
    return f"adj_{s}"[:128]


def _truncate_poem(text: str, max_chars: int) -> tuple[str, bool]:
    t = text if isinstance(text, str) else str(text)
    if len(t) <= max_chars:
        return t, False
    half = max_chars // 2 - 50
    return (
        t[:half]
        + "\n\n[...TRUNCATED_MIDDLE_FOR_TOKEN_LIMIT...]\n\n"
        + t[-half:],
        True,
    )


def _build_user_block(
    parent_id: str,
    expected: int,
    got: int,
    raw_ann: str,
    author: str,
    poem: str,
    truncated: bool,
) -> str:
    nlines = poem.count("\n") + (1 if poem else 0)
    meta = {
        "parent_id": parent_id,
        "expected": expected,
        "got": got,
        "multiple_poem_info": raw_ann,
        "author": author,
        "lines_total": nlines,
        "truncated": truncated,
    }
    return json.dumps(meta, ensure_ascii=False) + "\n---\nFULL_TEXT_LINES_BELOW\n---\n" + poem


def cmd_prepare(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = workdir / "batch_input.jsonl"
    manifest_path = workdir / "input_manifest.json"

    queue_path = Path(args.queue).resolve()
    db_path = Path(args.database).resolve()
    if not db_path.is_file():
        print(f"missing database: {db_path}", file=sys.stderr)
        return 1

    if queue_path.suffix.lower() == ".json":
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
    else:
        qdf = pd.read_csv(queue_path, low_memory=False)
        items = qdf.to_dict(orient="records")

    db = pd.read_csv(db_path, low_memory=False)
    db.columns = db.columns.str.strip()
    if "ID" not in db.columns or _TEXT_COL not in db.columns:
        print("database missing ID or poem text column", file=sys.stderr)
        return 1
    db["_pid"] = db["ID"].astype(str).str.strip()
    text_by_id = db.set_index("_pid")[_TEXT_COL].to_dict()

    manifest: list[dict] = []
    n_trunc = 0
    n_missing = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for it in items:
            pid = str(it.get("parent_id", "")).strip()
            if not pid:
                continue
            exp = int(it.get("expected_sub_poems", 1) or 1)
            got = int(it.get("obtained_sub_poems", 1) or 1)
            raw_ann = str(it.get("multiple_poem_info_raw", "") or "")
            author = str(it.get("author", "") or "")
            raw_poem = text_by_id.get(pid)
            if raw_poem is None or (isinstance(raw_poem, float) and pd.isna(raw_poem)):
                poem = ""
                n_missing += 1
                truncated = False
            else:
                poem, truncated = _truncate_poem(str(raw_poem), args.max_poem_chars)
                if truncated:
                    n_trunc += 1

            cid = _sanitize_custom_id(pid)
            user = _build_user_block(pid, exp, got, raw_ann, author, poem, truncated)
            sys_content = _SYSTEM + (_SYSTEM_RETRY_SUFFIX if getattr(args, "retry", False) else "")
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": args.max_completion_tokens,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            line = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            manifest.append(
                {
                    "custom_id": cid,
                    "parent_id": pid,
                    "expected_sub_poems": exp,
                    "obtained_sub_poems": got,
                    "truncated": truncated,
                }
            )

    manifest_path.write_text(
        json.dumps(
            {
                "queue_source": str(queue_path),
                "database": str(db_path),
                "model": args.model,
                "n_requests": len(manifest),
                "n_missing_text": n_missing,
                "n_truncated": n_trunc,
                "max_poem_chars": args.max_poem_chars,
                "retry_round2": bool(getattr(args, "retry", False)),
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {jsonl_path} ({len(manifest)} lines)")
    print(f"wrote {manifest_path}")
    if n_missing:
        print(f"warning: {n_missing} parent_id(s) not found in database (empty poem in batch)")
    if n_trunc:
        print(f"note: {n_trunc} poem(s) middle-truncated at {args.max_poem_chars} chars")
    if getattr(args, "retry", False):
        wd = str(workdir)
        print(
            "After finalize in this --workdir, merge on top of the canonical adjudication, then filter:\n"
            "  python src/00_gpt_human_review_batch.py merge-adjudications \\\n"
            "    --base data/To_run/00_filtering/batch_adjudication/layer0_gpt_adjudication.json \\\n"
            f"    --override {wd}/layer0_gpt_adjudication.json \\\n"
            "    --out data/To_run/00_filtering/batch_adjudication/layer0_gpt_adjudication_merged.json\n"
            "  PYTHONPATH=src python src/00_filtering.py \\\n"
            "    --gpt-adjudication data/To_run/00_filtering/batch_adjudication/layer0_gpt_adjudication_merged.json"
        )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    load_dotenv()
    workdir = Path(args.workdir).resolve()
    jsonl_path = workdir / "batch_input.jsonl"
    if not jsonl_path.is_file():
        print(f"missing {jsonl_path}; run prepare first", file=sys.stderr)
        return 1

    client = OpenAI()
    with open(jsonl_path, "rb") as f:
        up = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"description": "layer0 human review adjudication"},
    )
    meta = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "status": batch.status,
        "created_at": batch.created_at,
    }
    (workdir / "last_batch.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("\nNext: poll until status is completed, then finalize:")
    print(f"  python src/00_gpt_human_review_batch.py poll --batch-id {batch.id}")
    print(f"  python src/00_gpt_human_review_batch.py finalize --batch-id {batch.id}")
    return 0


def _normalize_adjudication_line_starts(obj: dict) -> None:
    ls = obj.get("line_starts_zero_based")
    if isinstance(ls, list):
        obj["line_starts_zero_based"] = "|".join(str(int(x)) for x in ls)


def _merge_adjudication_write(
    workdir: Path,
    success_objs: list[dict],
    *,
    replace_adjudication: bool,
) -> tuple[int, int]:
    """Merge rows into ``layer0_gpt_adjudication.json``."""
    batch_by: dict[str, dict] = {}
    for obj in success_objs:
        pid = str(obj.get("parent_id", "")).strip()
        if pid:
            batch_by[pid] = obj

    out_json = workdir / "layer0_gpt_adjudication.json"
    if replace_adjudication:
        merged_list = list(batch_by.values())
        merged_list.sort(key=lambda o: str(o.get("parent_id", "")))
        n_prior = 0
    else:
        prior_by: dict[str, dict] = {}
        if out_json.is_file():
            try:
                prior_raw = json.loads(out_json.read_text(encoding="utf-8"))
                if isinstance(prior_raw, list):
                    for obj in prior_raw:
                        if isinstance(obj, dict) and str(obj.get("parent_id", "")).strip():
                            prior_by[str(obj["parent_id"]).strip()] = obj
            except json.JSONDecodeError:
                pass
        n_prior = len(prior_by)
        merged_by = {**prior_by, **batch_by}
        merged_list = [merged_by[k] for k in sorted(merged_by)]

    out_json.write_text(json.dumps(merged_list, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(merged_list), n_prior


def cmd_run_sync(args: argparse.Namespace) -> int:
    """Run synchronous chat completions and merge outputs."""
    load_dotenv()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    queue_path = Path(args.queue).resolve()
    db_path = Path(args.database).resolve()
    if not db_path.is_file():
        print(f"missing database: {db_path}", file=sys.stderr)
        return 1
    if not queue_path.is_file():
        print(f"missing queue: {queue_path}", file=sys.stderr)
        return 1

    if queue_path.suffix.lower() == ".json":
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
    else:
        qdf = pd.read_csv(queue_path, low_memory=False)
        items = qdf.to_dict(orient="records")

    db = pd.read_csv(db_path, low_memory=False)
    db.columns = db.columns.str.strip()
    if "ID" not in db.columns or _TEXT_COL not in db.columns:
        print("database missing ID or poem text column", file=sys.stderr)
        return 1
    db["_pid"] = db["ID"].astype(str).str.strip()
    text_by_id = db.set_index("_pid")[_TEXT_COL].to_dict()

    client = OpenAI()
    sys_content = (
        _SYSTEM
        + _SYSTEM_RUN_SYNC_EXTRA
        + (_SYSTEM_RETRY_SUFFIX if getattr(args, "retry", False) else "")
    )
    success_objs: list[dict] = []
    exp_for_each: list[int] = []
    max_retries = max(1, int(args.max_retries))

    for it in items:
        pid = str(it.get("parent_id", "")).strip()
        if not pid:
            continue
        exp = int(it.get("expected_sub_poems", 1) or 1)
        got = int(it.get("obtained_sub_poems", 1) or 1)
        raw_ann = str(it.get("multiple_poem_info_raw", "") or "")
        author = str(it.get("author", "") or "")
        raw_poem = text_by_id.get(pid)
        if raw_poem is None or (isinstance(raw_poem, float) and pd.isna(raw_poem)):
            print(f"error: parent_id {pid!r} not found in database", file=sys.stderr)
            return 1
        poem, truncated = _truncate_poem(str(raw_poem), args.max_poem_chars)
        if truncated:
            print(f"note: {pid} poem middle-truncated at {args.max_poem_chars} chars")
        user = _build_user_block(pid, exp, got, raw_ann, author, poem, truncated)
        cid = _sanitize_custom_id(pid)

        last_err: Exception | None = None
        obj: dict | None = None
        for attempt in range(max_retries):
            try:
                comp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                )
                content = comp.choices[0].message.content
                if not content or not str(content).strip():
                    raise ValueError("empty model content")
                obj = json.loads(content)
                if not isinstance(obj, dict):
                    raise TypeError("model JSON was not an object")
                break
            except Exception as e:
                last_err = e
                wait = min(60.0, 2.0 ** attempt * 2.0)
                print(
                    f"warn: {pid} attempt {attempt + 1}/{max_retries} failed ({e!r}); "
                    f"sleeping {wait:.0f}s",
                    file=sys.stderr,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        if obj is None:
            print(f"error: giving up on parent_id={pid} after {max_retries} tries: {last_err!r}", file=sys.stderr)
            return 1

        obj.setdefault("parent_id", pid)
        obj["_custom_id"] = cid
        _normalize_adjudication_line_starts(obj)
        success_objs.append(obj)
        exp_for_each.append(exp)
        print(f"ok: {pid} verdict={obj.get('verdict')!r} true_poem_count={obj.get('true_poem_count')!r}")

    n_merged, n_prior = _merge_adjudication_write(
        workdir,
        success_objs,
        replace_adjudication=bool(getattr(args, "replace_adjudication", False)),
    )
    print(
        f"wrote {workdir / 'layer0_gpt_adjudication.json'} "
        f"({n_merged} rows; this run updated {len(success_objs)} parent_id(s))"
    )
    if not getattr(args, "replace_adjudication", False) and n_prior:
        print(f"(merged with prior file that had {n_prior} adjudication(s) with parent_id)")

    overrides: list[tuple[str, str]] = []
    for obj, exp in zip(success_objs, exp_for_each, strict=True):
        pid = str(obj.get("parent_id", "")).strip()
        ls = obj.get("line_starts_zero_based")
        use = obj.get("use_manual_override")
        verdict = obj.get("verdict")
        if (
            use
            and verdict == "annotation_correct"
            and isinstance(ls, str)
        ):
            parts = [p.strip() for p in ls.split("|") if p.strip()]
            if len(parts) == exp and parts[0] == "0" and all(p.isdigit() for p in parts):
                overrides.append((pid, ls))

    if overrides and args.write_overrides:
        ov_path = Path(args.overrides_out).resolve()
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["parent_id,line_starts_zero_based"]
        for p, ls in overrides:
            lines.append(f"{p},{ls}")
        ov_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {ov_path} ({len(overrides)} override rows)")
    elif overrides:
        print(f"info: {len(overrides)} rows qualify for overrides; pass --write-overrides to emit CSV")

    meta = {
        "mode": "run-sync",
        "model": args.model,
        "queue": str(queue_path),
        "database": str(db_path),
        "n_ok": len(success_objs),
        "merged_total": n_merged,
    }
    (workdir / "last_run_sync.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    load_dotenv()
    client = OpenAI()
    bid = args.batch_id
    interval = max(5, args.interval)
    while True:
        b = client.batches.retrieve(bid)
        print(f"status={b.status}")
        if b.status in ("completed", "failed", "cancelled", "expired"):
            if b.status != "completed":
                print(b.model_dump_json(indent=2) if hasattr(b, "model_dump_json") else str(b))
                return 1
            out_path = Path(args.workdir).resolve() / "last_batch.json"
            prev = {}
            if out_path.is_file():
                prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev.update(
                {
                    "batch_id": b.id,
                    "status": b.status,
                    "output_file_id": b.output_file_id,
                    "error_file_id": b.error_file_id,
                }
            )
            out_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
            print("batch completed. output_file_id:", b.output_file_id)
            return 0
        time.sleep(interval)


def cmd_finalize(args: argparse.Namespace) -> int:
    load_dotenv()
    workdir = Path(args.workdir).resolve()
    client = OpenAI()
    b = client.batches.retrieve(args.batch_id)
    if b.status != "completed":
        print(f"batch status is {b.status}, not completed", file=sys.stderr)
        return 1
    if not b.output_file_id:
        print("no output_file_id", file=sys.stderr)
        return 1

    raw = client.files.content(b.output_file_id).read()
    out_jsonl = workdir / "batch_output_raw.jsonl"
    out_jsonl.write_bytes(raw)

    manifest_path = workdir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp_by_cid = {m["custom_id"]: m["expected_sub_poems"] for m in manifest["items"]}
    pid_by_cid = {m["custom_id"]: m["parent_id"] for m in manifest["items"]}

    results: list[dict] = []
    overrides: list[tuple[str, str]] = []

    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id", "")
        resp = row.get("response", {})
        if resp.get("status_code") != 200:
            results.append(
                {
                    "custom_id": cid,
                    "parent_id": pid_by_cid.get(cid),
                    "error": resp.get("body"),
                }
            )
            continue
        body = resp.get("body", {})
        try:
            content = body["choices"][0]["message"]["content"]
            obj = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            results.append({"custom_id": cid, "parent_id": pid_by_cid.get(cid), "parse_error": str(e)})
            continue

        pid = obj.get("parent_id") or pid_by_cid.get(cid, "")
        obj["_custom_id"] = cid
        _normalize_adjudication_line_starts(obj)
        results.append(obj)

        exp = exp_by_cid.get(cid)
        ls = obj.get("line_starts_zero_based")
        if isinstance(ls, list):
            ls = "|".join(str(int(x)) for x in ls)
        use = obj.get("use_manual_override")
        verdict = obj.get("verdict")
        if (
            use
            and verdict == "annotation_correct"
            and isinstance(ls, str)
            and exp is not None
        ):
            parts = [p.strip() for p in ls.split("|") if p.strip()]
            if len(parts) == exp and parts[0] == "0" and all(p.isdigit() for p in parts):
                overrides.append((pid, ls))

    out_json = workdir / "layer0_gpt_adjudication.json"
    batch_by: dict[str, dict] = {}
    for obj in results:
        if not isinstance(obj, dict):
            continue
        pid = str(obj.get("parent_id", "")).strip()
        if not pid:
            cid = obj.get("custom_id") or obj.get("_custom_id")
            if cid:
                pid = str(pid_by_cid.get(str(cid), "")).strip()
        if pid:
            batch_by[pid] = obj

    if getattr(args, "replace_adjudication", False):
        merged_list = results
        n_prior = 0
    else:
        prior_by: dict[str, dict] = {}
        if out_json.is_file():
            try:
                prior_raw = json.loads(out_json.read_text(encoding="utf-8"))
                if isinstance(prior_raw, list):
                    for obj in prior_raw:
                        if isinstance(obj, dict) and str(obj.get("parent_id", "")).strip():
                            prior_by[str(obj["parent_id"]).strip()] = obj
            except json.JSONDecodeError:
                pass
        n_prior = len(prior_by)
        merged_by = {**prior_by, **batch_by}
        merged_list = [merged_by[k] for k in sorted(merged_by)]

    out_json.write_text(json.dumps(merged_list, ensure_ascii=False, indent=2), encoding="utf-8")
    if getattr(args, "replace_adjudication", False):
        print(f"wrote {out_json} ({len(merged_list)} results, replace mode)")
    else:
        carried = len(merged_list) - len(batch_by)
        print(
            f"wrote {out_json} ({len(merged_list)} results; "
            f"this batch {len(batch_by)} parent_id(s), carried over {carried} from prior file)"
        )
        if n_prior and len(batch_by) < n_prior:
            print(
                f"note: prior file had {n_prior} adjudication(s); merged so this batch only replaced "
                f"{len(batch_by)} id(s) (rest kept).",
                file=sys.stderr,
            )

    if overrides and args.write_overrides:
        ov_path = Path(args.overrides_out).resolve()
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["parent_id,line_starts_zero_based"]
        for pid, ls in overrides:
            lines.append(f"{pid},{ls}")
        ov_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {ov_path} ({len(overrides)} override rows) — review before running 00_filtering.py")
    elif overrides:
        print(f"info: {len(overrides)} rows qualify for overrides; pass --write-overrides to emit CSV")

    if b.error_file_id:
        err_raw = client.files.content(b.error_file_id).read()
        err_path = workdir / "batch_errors_raw.jsonl"
        err_path.write_bytes(err_raw)
        print(f"warning: batch reported errors; see {err_path}")

    return 0


def cmd_merge_adjudications(args: argparse.Namespace) -> int:
    """Merge two adjudication arrays; override wins by parent_id."""
    base_path = Path(args.base).resolve()
    ovr_path = Path(args.override).resolve()
    out_path = Path(args.out).resolve()
    if not base_path.is_file():
        print(f"missing --base file: {base_path}", file=sys.stderr)
        return 1
    if not ovr_path.is_file():
        print(
            f"missing --override file: {ovr_path}\n"
            "If you ran finalize without --workdir, the batch output is under the default "
            "batch_adjudication/ folder (same path as --base). In that case you do not need "
            "merge — pass that layer0_gpt_adjudication.json directly to 00_filtering.py.",
            file=sys.stderr,
        )
        return 1
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ovr = json.loads(ovr_path.read_text(encoding="utf-8"))
    if not isinstance(base, list) or not isinstance(ovr, list):
        print("both --base and --override must be JSON arrays", file=sys.stderr)
        return 1
    by_pid: dict[str, dict] = {}
    n_base_pid = 0
    for obj in base:
        if isinstance(obj, dict) and str(obj.get("parent_id", "")).strip():
            by_pid[str(obj["parent_id"]).strip()] = obj
            n_base_pid += 1
    for obj in ovr:
        if isinstance(obj, dict) and str(obj.get("parent_id", "")).strip():
            by_pid[str(obj["parent_id"]).strip()] = obj
    merged = [by_pid[k] for k in sorted(by_pid)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ovr = sum(1 for o in ovr if isinstance(o, dict) and str(o.get("parent_id", "")).strip())
    print(f"wrote {out_path} ({len(merged)} objects; base had {n_base_pid} with parent_id; {n_ovr} from override)")
    if n_base_pid == 0 and len(base) > 0:
        print(
            "warning: --base list had no usable parent_id rows; merged output is override-only. "
            "Restore a full layer0_gpt_adjudication.json (e.g. re-run finalize with the large batch_id "
            "from last_batch.json in the same workdir), then merge again.",
            file=sys.stderr,
        )
    elif len(merged) == n_ovr and n_ovr < n_base_pid:
        print(
            "warning: merged length equals override only while base had more rows — "
            "check that base objects include non-empty parent_id.",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI Batch adjudication for layer0 review queue.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="Build batch_input.jsonl + input_manifest.json")
    pr.add_argument(
        "--queue",
        type=Path,
        default=_DEFAULT_QUEUE,
        help="Human-review queue: default layer0_human_review_queue.csv (.json with an 'items' array also supported).",
    )
    pr.add_argument("--database", type=Path, default=_DEFAULT_DB)
    pr.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    pr.add_argument("--model", default="gpt-4o-mini")
    pr.add_argument("--max-poem-chars", type=int, default=_DEFAULT_MAX_POEM_CHARS)
    pr.add_argument("--max-completion-tokens", type=int, default=450, dest="max_completion_tokens")
    pr.add_argument(
        "--retry",
        action="store_true",
        help="Round-2 stricter system prompt (use with the shrunk human_review_queue CSV).",
    )
    pr.set_defaults(func=cmd_prepare)

    su = sub.add_parser("submit", help="Upload JSONL and create batch job")
    su.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    su.add_argument("--completion-window", default="24h", dest="completion_window")
    su.set_defaults(func=cmd_submit)

    po = sub.add_parser("poll", help="Poll batch until completed")
    po.add_argument("--batch-id", required=True, dest="batch_id")
    po.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    po.add_argument("--interval", type=int, default=30)
    po.set_defaults(func=cmd_poll)

    fi = sub.add_parser("finalize", help="Download batch output and parse to JSON / optional overrides CSV")
    fi.add_argument("--batch-id", required=True, dest="batch_id")
    fi.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    fi.add_argument(
        "--replace-adjudication",
        action="store_true",
        help="Write only this batch (do not merge with existing layer0_gpt_adjudication.json).",
    )
    fi.add_argument("--write-overrides", action="store_true")
    fi.add_argument(
        "--overrides-out",
        type=Path,
        default=_ROOT / "data" / "To_run" / "00_filtering" / "layer0_split_overrides.csv",
    )
    fi.set_defaults(func=cmd_finalize)

    rs = sub.add_parser(
        "run-sync",
        help=(
            "Call Chat Completions for each queue row (with retries), then merge into "
            "layer0_gpt_adjudication.json under --workdir (no Batch API)."
        ),
    )
    rs.add_argument(
        "--queue",
        type=Path,
        default=_DEFAULT_QUEUE,
        help="Human-review queue CSV or JSON with an 'items' array.",
    )
    rs.add_argument("--database", type=Path, default=_DEFAULT_DB)
    rs.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    rs.add_argument(
        "--model",
        default="gpt-5.1",
        help="Chat Completions model (default: gpt-5.1).",
    )
    rs.add_argument("--max-poem-chars", type=int, default=_DEFAULT_MAX_POEM_CHARS)
    rs.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        dest="max_completion_tokens",
    )
    rs.add_argument("--temperature", type=float, default=0.0)
    rs.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Attempts per parent_id on empty/parse/API errors (default: 6).",
    )
    rs.add_argument(
        "--retry",
        action="store_true",
        help="Append round-2 stricter instructions to the system prompt.",
    )
    rs.add_argument(
        "--replace-adjudication",
        action="store_true",
        dest="replace_adjudication",
        help="Write only this run's rows (do not merge with existing adjudication file).",
    )
    rs.add_argument("--write-overrides", action="store_true")
    rs.add_argument(
        "--overrides-out",
        type=Path,
        default=_ROOT / "data" / "To_run" / "00_filtering" / "layer0_split_overrides.csv",
    )
    rs.set_defaults(func=cmd_run_sync)

    mg = sub.add_parser(
        "merge-adjudications",
        help="Merge two layer0_gpt_adjudication.json arrays (--override wins on parent_id)",
    )
    mg.add_argument("--base", type=Path, required=True, help="First JSON array (e.g. round 1)")
    mg.add_argument("--override", type=Path, required=True, help="Second JSON array (e.g. round 2)")
    mg.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output merged JSON array path",
    )
    mg.set_defaults(func=cmd_merge_adjudications)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
