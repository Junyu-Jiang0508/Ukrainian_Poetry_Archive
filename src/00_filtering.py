"""Layer 0-1 preprocessing: post -> sub-poems -> stanzas."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from utils.workspace import filtering_processed_dir

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _ROOT / "data" / "raw" / "ukrpoetry_database.csv"
_OUT_DIR = filtering_processed_dir(_ROOT)

_TEXT_COL = "Poem full text (copy and paste)"
_MULTI_COL = "Multiple poem info"

_STRATEGY_ORDER = (
    "asterisk_line",
    "dash_rule_line",
    "number_line_only",
    "number_line_with_text",
    "double_blank",
)

_RE_ASTERISK_LINE = re.compile(r"^\s*\*{2,}\s*$")
_RE_ASTERISK_SPACED_LINE = re.compile(r"^\s*(?:\*\s*){2,}\s*$")
_RE_RULE_SEPARATOR_LINE = re.compile(r"^\s*(?:[-_=]){3,}\s*$")
_RE_NUMBER_ONLY_LINE = re.compile(
    r"^\s*(?:[IVXLCDM]{1,8}|[ivxlcdm]{1,8}|\d{1,2})\.\s*$"
)
_RE_NUMBER_THEN_TEXT = re.compile(
    r"^\s*(?:\d{1,2}|[IVXLCDM]{1,8}|[ivxlcdm]{1,8})\.\s+\S"
)
_RE_DOUBLE_BLANK_SPLIT = re.compile(r"\n(?:[ \t]*\n)+")
_RE_TRAILING_DATE = re.compile(
    r"^\s*\d{1,2}\.\d{1,2}\.(?:\d{2}|\d{4})?\s*$"
)
_RE_STANZA_DECORATION = re.compile(r"^\s*(\*\s*){2,}\s*$")


def _norm_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_record_id(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    if isinstance(raw, float) and float(raw).is_integer():
        return str(int(raw))
    s = str(raw).strip()
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def parse_expected_poem_count(val: object) -> int:
    """Parse expected poem count."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 1
    s = str(val).strip().lower()
    m = re.match(r"(\d+)\s*poems?", s)
    if m:
        return max(1, int(m.group(1)))
    m = re.match(r"(\d+)\s*poem\b", s)
    if m:
        return max(1, int(m.group(1)))
    return 1


def _is_asterisk_only_boundary_line(ln: str) -> bool:
    return bool(_RE_ASTERISK_LINE.match(ln) or _RE_ASTERISK_SPACED_LINE.match(ln))


def _split_by_line_predicate(
    text: str,
    is_boundary,
    *,
    keep_boundary_line: bool = False,
    require_nonzero_index: bool = False,
) -> list[str]:
    lines = _norm_newlines(text).split("\n")
    breaks = [
        i
        for i, ln in enumerate(lines)
        if (not require_nonzero_index or i > 0) and is_boundary(ln)
    ]
    if not breaks:
        return [text]

    segs: list[str] = []
    start = 0
    for b in breaks:
        chunk = "\n".join(lines[start:b]).strip()
        if chunk:
            segs.append(chunk)
        start = b if keep_boundary_line else b + 1
    tail = "\n".join(lines[start:]).strip()
    if tail:
        segs.append(tail)
    return segs if segs else [text.strip()]


def split_by_asterisk_lines(text: str) -> list[str]:
    return _split_by_line_predicate(text, _is_asterisk_only_boundary_line)


def split_by_dash_rule_lines(text: str) -> list[str]:
    return _split_by_line_predicate(text, lambda ln: bool(_RE_RULE_SEPARATOR_LINE.match(ln)))


def _is_strong_separator_line(ln: str) -> bool:
    if not ln.strip():
        return False
    if _is_asterisk_only_boundary_line(ln):
        return True
    if _RE_RULE_SEPARATOR_LINE.match(ln):
        return True
    return False


def split_by_first_separator_boundaries(text: str, expected_n: int) -> list[str] | None:
    """Split by separator-only lines to match ``expected_n``."""
    if expected_n <= 1:
        return None
    lines = _norm_newlines(text).split("\n")
    sep_idx = [i for i, ln in enumerate(lines) if _is_strong_separator_line(ln) and i > 0]
    need = expected_n - 1
    if len(sep_idx) < need:
        return None
    if len(sep_idx) == need:
        chosen = sep_idx[:need]
    else:
        if need == 1:
            chosen = [sep_idx[len(sep_idx) // 2]]
        else:
            chosen = sorted(
                {sep_idx[int(round(i * (len(sep_idx) - 1) / (need - 1)))] for i in range(need)}
            )
        if len(chosen) < need:
            return None
    segs: list[str] = []
    start = 0
    for s in chosen:
        chunk = "\n".join(lines[start:s]).strip()
        if chunk:
            segs.append(chunk)
        start = s + 1
    tail = "\n".join(lines[start:]).strip()
    if tail:
        segs.append(tail)
    if len(segs) != expected_n:
        return None
    return segs


def split_by_number_only_lines(text: str) -> list[str]:
    return _split_by_line_predicate(
        text,
        lambda ln: bool(_RE_NUMBER_ONLY_LINE.match(ln)),
        keep_boundary_line=True,
        require_nonzero_index=True,
    )


def split_by_number_then_text_lines(text: str) -> list[str]:
    return _split_by_line_predicate(
        text,
        lambda ln: bool(_RE_NUMBER_THEN_TEXT.match(ln)),
        keep_boundary_line=True,
        require_nonzero_index=True,
    )


def split_by_double_blank(text: str) -> list[str]:
    t = _norm_newlines(text)
    parts = [p.strip() for p in _RE_DOUBLE_BLANK_SPLIT.split(t) if p.strip()]
    return parts if parts else [t.strip()]


def _strategy_fn(name: str):
    return {
        "asterisk_line": split_by_asterisk_lines,
        "dash_rule_line": split_by_dash_rule_lines,
        "number_line_only": split_by_number_only_lines,
        "number_line_with_text": split_by_number_then_text_lines,
        "double_blank": split_by_double_blank,
    }[name]


def merge_segments_down(segments: list[str], target_n: int) -> list[str]:
    """Greedily merge adjacent shortest pair until len == target_n."""
    segs = [s.strip() for s in segments if s.strip()]
    while len(segs) > target_n:
        best_i = min(
            range(len(segs) - 1),
            key=lambda i: len(segs[i]) + len(segs[i + 1]),
        )
        merged = segs[best_i] + "\n\n" + segs[best_i + 1]
        segs = segs[:best_i] + [merged] + segs[best_i + 2 :]
    return segs


def split_text_at_line_starts(text: str, starts: list[int]) -> list[str]:
    """Cut text by 0-based poem start lines."""
    lines = _norm_newlines(text).split("\n")
    st = sorted({int(x) for x in starts})
    if not st:
        return [_norm_newlines(text).strip()]
    if st[0] != 0:
        st = [0] + st
    st.append(len(lines))
    out: list[str] = []
    for a, b in zip(st[:-1], st[1:]):
        chunk = "\n".join(lines[a:b]).strip()
        if chunk:
            out.append(chunk)
    return out if out else [_norm_newlines(text).strip()]


def load_split_overrides(path: Path) -> dict[str, list[int]]:
    """Load manual split overrides."""
    if not path.is_file():
        return {}
    odf = pd.read_csv(path, dtype=str)
    if "parent_id" not in odf.columns:
        return {}
    col = "line_starts_zero_based"
    if col not in odf.columns:
        return {}
    out: dict[str, list[int]] = {}
    for _, r in odf.iterrows():
        pid = str(r["parent_id"]).strip()
        raw = str(r.get(col, "")).strip()
        parts = [p.strip() for p in raw.replace(",", "|").split("|") if p.strip()]
        starts: list[int] = []
        for p in parts:
            if re.fullmatch(r"-?\d+", p):
                starts.append(int(p))
        if pid and starts:
            out[pid] = starts
    return out


def _line_starts_from_gpt_field(ls: object, expected_n: int) -> list[int] | None:
    """Validate GPT line starts."""
    if ls is None or (isinstance(ls, float) and pd.isna(ls)):
        return None
    if isinstance(ls, list):
        parts = [str(int(x)) for x in ls]
    else:
        s = str(ls).strip()
        if not s or s.lower() == "null":
            return None
        parts = [p.strip() for p in s.replace(",", "|").split("|") if p.strip()]
    if len(parts) != expected_n:
        return None
    if parts[0] != "0":
        return None
    if not all(p.isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def _repair_line_starts_omitted_leading_zero(
    ls: object, declared_n: int
) -> tuple[list[int], int] | None:
    """Repair GPT starts when leading 0 is omitted."""
    if declared_n < 2:
        return None
    if isinstance(ls, list):
        raw_parts = [str(int(x)) for x in ls]
    else:
        s = str(ls).strip()
        if not s or s.lower() == "null":
            return None
        raw_parts = [p.strip() for p in s.replace(",", "|").split("|") if p.strip()]
    if len(raw_parts) != declared_n or not all(p.isdigit() for p in raw_parts):
        return None
    if raw_parts[0] == "0":
        return None
    ints = [int(p) for p in raw_parts]
    if ints != sorted(ints) or len(set(ints)) != len(ints):
        return None
    if ints[0] <= 0:
        return None
    return ([0] + ints, declared_n + 1)


def load_gpt_adjudication(path: Path) -> dict[str, dict]:
    """Load GPT adjudication rows keyed by parent id."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict] = {}
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        if obj.get("parse_error") or obj.get("error"):
            continue
        pid = str(obj.get("parent_id", "")).strip()
        if not pid:
            continue
        verdict = (obj.get("verdict") or "").strip()
        if verdict == "ambiguous":
            continue
        tc = obj.get("true_poem_count")
        if tc is None or (isinstance(tc, float) and pd.isna(tc)):
            continue
        try:
            exp_n = max(1, int(tc))
        except (TypeError, ValueError):
            continue
        line_starts = _line_starts_from_gpt_field(obj.get("line_starts_zero_based"), exp_n)
        if line_starts is None:
            repaired = _repair_line_starts_omitted_leading_zero(
                obj.get("line_starts_zero_based"), exp_n
            )
            if repaired:
                line_starts, exp_n = repaired
        out[pid] = {
            "verdict": verdict,
            "true_poem_count": exp_n,
            "line_starts": line_starts,
        }
    return out


def split_into_subpoems(text: str, expected_n: int) -> tuple[list[str], str, bool, bool]:
    """Return (segments, method, qc_ok, repaired)."""
    if not text or not str(text).strip():
        if expected_n <= 1:
            return ([""], "empty", True, False)
        return ([""], "empty", False, False)
    text = str(text)
    if expected_n <= 1:
        return ([_norm_newlines(text).strip()], "single", True, False)

    best_rank, best_name = 0, _STRATEGY_ORDER[0]
    best_segs: list[str] = []
    best_dist = 10**9
    for rank, name in enumerate(_STRATEGY_ORDER):
        segs = [s for s in _strategy_fn(name)(text) if s.strip()]
        dist = abs(len(segs) - expected_n)
        if dist < best_dist or (dist == best_dist and rank < best_rank):
            best_dist = dist
            best_rank = rank
            best_segs = segs
            best_name = name

    segs = [s.strip() for s in best_segs if s.strip()]
    repaired = False
    if len(segs) > expected_n:
        segs = merge_segments_down(segs, expected_n)
        repaired = True
    qc_ok = len(segs) == expected_n
    if not qc_ok:
        alt = split_by_first_separator_boundaries(text, expected_n)
        if alt is not None:
            return (alt, "separator_boundaries", True, False)
    return (segs, best_name, qc_ok, repaired)


def _norm_author_key(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def peel_trailing_poem_metadata(text: str, author_display: str) -> tuple[str, str, str]:
    """Strip optional trailing date/signature lines."""
    t = _norm_newlines(text).strip()
    if not t:
        return "", "", ""

    trailing_date = ""
    trailing_signature = ""
    lines = t.split("\n")

    while lines and not lines[-1].strip():
        lines.pop()

    author_key = _norm_author_key(author_display) if author_display else ""

    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if _RE_TRAILING_DATE.match(last):
            trailing_date = last.strip()
            lines.pop()
            continue
        if author_key and _norm_author_key(last) == author_key:
            trailing_signature = last.strip()
            lines.pop()
            continue
        break

    body = "\n".join(lines).strip()
    return body, trailing_date, trailing_signature


def _non_empty_line_count(block: str) -> int:
    return sum(1 for ln in block.split("\n") if ln.strip())


def _substantial_stanza(block: str, min_lines: int = 3, min_chars: int = 120) -> bool:
    return _non_empty_line_count(block) >= min_lines or len(block.strip()) >= min_chars


def refine_stanza_decoration_blocks(blocks: list[str]) -> list[str]:
    """Resolve separator-only decoration blocks."""
    if not blocks:
        return blocks
    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if _RE_STANZA_DECORATION.match(_norm_newlines(b).strip()):
            prev = out[-1] if out else ""
            nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
            if (
                prev
                and nxt
                and _substantial_stanza(prev)
                and _substantial_stanza(nxt)
            ):
                i += 1
                continue
            if out:
                out[-1] = (out[-1].rstrip() + "\n" + b.strip()).strip()
            elif nxt:
                blocks[i + 1] = (b.strip() + "\n" + nxt).strip()
            i += 1
            continue
        out.append(b)
        i += 1
    return out


def segment_stanzas(
    poem_text: str,
    author_for_peel: str = "",
) -> tuple[list[str], str, str]:
    """Return stanzas and trailing metadata."""
    peeled, td, ts = peel_trailing_poem_metadata(poem_text, author_for_peel)
    if not peeled:
        return [], td, ts

    raw_blocks = [
        p.strip() for p in _RE_DOUBLE_BLANK_SPLIT.split(_norm_newlines(peeled)) if p.strip()
    ]
    blocks = refine_stanza_decoration_blocks(raw_blocks)
    stanzas: list[str] = []
    for b in blocks:
        inner = b.strip()
        if not inner:
            continue
        inner = _norm_newlines(inner).strip("\n")
        if inner:
            stanzas.append(inner)
    return stanzas, td, ts


def run_layer0(
    df: pd.DataFrame,
    overrides: dict[str, list[int]] | None = None,
    adjudication: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if _TEXT_COL not in df.columns:
        raise KeyError(f"Missing {_TEXT_COL!r}; columns={list(df.columns)[:30]}")

    overrides = overrides or {}
    adjudication = adjudication or {}
    layer0_rows: list[dict] = []
    review_rows: list[dict] = []

    for idx, row in df.iterrows():
        raw_id = normalize_record_id(row.get("ID", ""))
        parent_key = raw_id or f"__row_index_{idx}"

        author = row.get("Author of poem", "")
        if pd.isna(author):
            author = ""
        author = str(author)

        text = row.get(_TEXT_COL, "")
        if pd.isna(text):
            text = ""
        text = str(text)

        expected_n = parse_expected_poem_count(row.get(_MULTI_COL))
        gpt_adj = adjudication.get(parent_key)
        gpt_applied = False
        gpt_verdict_for_row: str | None = None

        if parent_key in overrides:
            osegs = split_text_at_line_starts(text, overrides[parent_key])
            if len(osegs) == expected_n:
                segments, method, qc_ok, repaired = (
                    osegs,
                    "manual_csv",
                    True,
                    False,
                )
            else:
                segments, method, qc_ok, repaired = split_into_subpoems(text, expected_n)
        elif gpt_adj is not None:
            gpt_applied = True
            expected_n = int(gpt_adj["true_poem_count"])
            gpt_verdict_for_row = str(gpt_adj.get("verdict") or "") or None
            starts = gpt_adj.get("line_starts")
            if starts:
                segs_gpt = split_text_at_line_starts(text, starts)
                if len(segs_gpt) == expected_n:
                    segments, method, qc_ok, repaired = (
                        segs_gpt,
                        "gpt_batch_linecuts",
                        True,
                        False,
                    )
                else:
                    segments, method, qc_ok, repaired = split_into_subpoems(text, expected_n)
            else:
                segments, method, qc_ok, repaired = split_into_subpoems(text, expected_n)
        else:
            segments, method, qc_ok, repaired = split_into_subpoems(text, expected_n)

        if not qc_ok:
            review_rows.append(
                {
                    "parent_id": parent_key,
                    "expected_sub_poems": expected_n,
                    "obtained_sub_poems": len(segments),
                    "split_strategy": method,
                    "repaired_by_merge": repaired,
                    "multiple_poem_info_raw": row.get(_MULTI_COL, ""),
                    "author": author,
                }
            )

        base_meta = row.to_dict()
        for k, seg in enumerate(segments, start=1):
            poem_id = f"{parent_key}_{k}"
            layer0_rows.append(
                {
                    **base_meta,
                    "parent_id": parent_key,
                    "sub_poem_index": k,
                    "poem_id": poem_id,
                    "layer0_split_strategy": method,
                    "layer0_qc_count_match": qc_ok,
                    "layer0_merge_repaired": repaired,
                    "layer0_expected_sub_poems": expected_n,
                    "layer0_obtained_sub_poems": len(segments),
                    "layer0_gpt_applied": gpt_applied,
                    "layer0_gpt_verdict": gpt_verdict_for_row,
                    _TEXT_COL: seg,
                }
            )

    layer0_df = pd.DataFrame(layer0_rows)
    review_df = pd.DataFrame(review_rows)
    return layer0_df, review_df


def run_layer1(layer0_df: pd.DataFrame) -> pd.DataFrame:
    """Build stanza table from layer0 output."""
    if layer0_df.empty:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "parent_id",
                "sub_poem_index",
                "stanza_index",
                "stanza_label",
                "stanza_text",
                "total_stanzas_in_poem",
                "full_poem_text",
                "trailing_date_line",
                "trailing_signature_line",
                "author",
            ]
        )

    df = layer0_df.copy()
    if "poem_id" not in df.columns:
        raise ValueError("Layer 0 output must contain poem_id")

    stanza_rows: list[dict] = []
    for _, row in df.iterrows():
        poem_id = row["poem_id"]
        text = row.get(_TEXT_COL, "")
        if pd.isna(text):
            text = ""
        author = row.get("Author of poem", "")
        if pd.isna(author):
            author = ""

        stanzas, td, ts = segment_stanzas(str(text), str(author))
        full_poem = "\n\n".join(stanzas) if stanzas else str(text).strip()
        total = len(stanzas)
        if total == 0:
            stanza_rows.append(
                {
                    "poem_id": poem_id,
                    "parent_id": row.get("parent_id", ""),
                    "sub_poem_index": row.get("sub_poem_index", ""),
                    "stanza_index": 0,
                    "stanza_label": "",
                    "stanza_text": "",
                    "total_stanzas_in_poem": 0,
                    "full_poem_text": full_poem,
                    "trailing_date_line": td,
                    "trailing_signature_line": ts,
                    "author": author,
                }
            )
            continue

        for si, st in enumerate(stanzas, start=1):
            stanza_rows.append(
                {
                    "poem_id": poem_id,
                    "parent_id": row.get("parent_id", ""),
                    "sub_poem_index": row.get("sub_poem_index", ""),
                    "stanza_index": si,
                    "stanza_label": f"s{si}",
                    "stanza_text": st,
                    "total_stanzas_in_poem": total,
                    "full_poem_text": full_poem,
                    "trailing_date_line": td,
                    "trailing_signature_line": ts,
                    "author": author,
                }
            )

    return pd.DataFrame(stanza_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer 0–1: poem split + stanza segmentation.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Raw ukrpoetry_database CSV (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        help=f"Output directory (default: {_OUT_DIR})",
    )
    parser.add_argument(
        "--gpt-adjudication",
        type=Path,
        default=None,
        help=(
            "Optional path to layer0_gpt_adjudication.json (Batch finalize). "
            "Applies true_poem_count / line cuts for adjudicated parent_ids; "
            "manual layer0_split_overrides.csv still wins when both exist."
        ),
    )
    args = parser.parse_args(argv)

    inp = args.input.resolve()
    out_dir = args.out_dir.resolve()
    if not inp.is_file():
        print(f"missing input: {inp}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    layer0_path = out_dir / "layer0_poems_one_per_row.csv"
    review_path = out_dir / "layer0_human_review_queue.csv"
    layer1_path = out_dir / "layer1_stanzas_one_per_row.csv"

    print(f"Reading {inp} …")
    df = pd.read_csv(inp, low_memory=False)

    overrides_path = out_dir / "layer0_split_overrides.csv"
    overrides = load_split_overrides(overrides_path)
    if overrides:
        print(f"Loaded {len(overrides)} manual split override(s) from {overrides_path.name}")

    adjudication: dict[str, dict] = {}
    if args.gpt_adjudication is not None:
        adj_path = args.gpt_adjudication.resolve()
        if not adj_path.is_file():
            print(f"warning: missing --gpt-adjudication file {adj_path}", file=sys.stderr)
        else:
            adjudication = load_gpt_adjudication(adj_path)
            if adjudication:
                print(f"Loaded {len(adjudication)} GPT adjudication row(s) from {adj_path}")
            else:
                print(f"warning: no adjudication entries parsed from {adj_path}", file=sys.stderr)

    print("Layer 0 …")
    layer0_df, review_df = run_layer0(df, overrides, adjudication)
    layer0_df.to_csv(layer0_path, index=False, encoding="utf-8-sig")
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    print("Layer 1 …")
    layer1_df = run_layer1(layer0_df)
    layer1_df.to_csv(layer1_path, index=False, encoding="utf-8-sig")

    n_posts = len(df)
    n_poems = len(layer0_df)
    n_review = len(review_df)
    n_stanza_rows = len(layer1_df)
    n_posts_qc_ok = n_posts - n_review

    summary = (
        f"posts={n_posts}  layer0_poems={n_poems}  posts_qc_exact_count={n_posts_qc_ok}  "
        f"review_queue_posts={n_review}  layer1_rows={n_stanza_rows}\n"
        f"wrote:\n  {layer0_path}\n  {review_path}\n  {layer1_path}\n"
    )
    print(summary)
    (out_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
