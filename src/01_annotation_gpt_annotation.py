"""Batch GPT annotation: pronoun identification per stanza (async, chunked)."""

import argparse
import asyncio
import csv
import json
import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv

from utils.workspace import filtering_processed_dir, repository_root_for_script

_ROOT = repository_root_for_script(__file__)
_FILTERING_DIR = filtering_processed_dir(_ROOT)

                                                                               
LAYER0_CSV = _FILTERING_DIR / "layer0_poems_one_per_row.csv"
LAYER1_CSV = _FILTERING_DIR / "layer1_stanzas_one_per_row.csv"

                                                                               
_DEFAULT_OUTPUT_DIR = _ROOT / "data" / "Annotated_GPT"

OUTPUT_DIR = _DEFAULT_OUTPUT_DIR
OUTPUT_CSV = OUTPUT_DIR / "pronoun_annotation.csv"
OUTPUT_JSONL = OUTPUT_DIR / "pronoun_annotation_raw.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_done_ids.txt"
COST_REPORT_FILE = OUTPUT_DIR / "token_usage_report.txt"


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSONL, CHECKPOINT_FILE, COST_REPORT_FILE
    OUTPUT_DIR = output_dir
    OUTPUT_CSV = OUTPUT_DIR / "pronoun_annotation.csv"
    OUTPUT_JSONL = OUTPUT_DIR / "pronoun_annotation_raw.jsonl"
    CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_done_ids.txt"
    COST_REPORT_FILE = OUTPUT_DIR / "token_usage_report.txt"


load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

PRICE = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
}

                                                                               
CONCURRENCY = 15
MAX_LINES_PER_CHUNK = 30
MAX_COMPLETION_TOKENS = 12000

MAX_RETRIES = 5
BACKOFF_BASE = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


                                                                                 
                   
                                                                                 

class PronounAnalysis(BaseModel):
    english_pronoun: str = Field(
        ...,
        description=(
            "Exact personal pronoun token from your Shakespearean translation. "
            "If pro-dropped in source, render the INFERRED pronoun explicitly — "
            "NEVER write '(pro-drop)'. "
            "ONLY personal pronouns (I/me/my/mine/myself, we/us/our/ours/ourselves, "
            "thou/thee/thy/thine/thyself, ye/you/your/yours/yourselves, "
            "he/him/his/himself, she/her/hers/herself, it/its/itself, "
            "they/them/their/theirs/themselves)."
        ),
    )
    person: Literal["1st", "2nd", "3rd", "Impersonal"] = Field(
        ...,
        description="Impersonal ONLY for Ukrainian impersonal verb constructions with no personal subject.",
    )
    number: Literal["Singular", "Plural", "None"] = Field(
        ...,
        description=(
            "MORPHOLOGICAL number only. ви-forms are ALWAYS 'Plural' even if "
            "polite address to one person. ти-forms are ALWAYS 'Singular'. "
            "'None' only for genuinely impersonal constructions."
        ),
    )
    vy_register: Literal[
        "genuine_plural", "polite_singular", "ambiguous", "not_applicable"
    ] = Field(
        ...,
        description=(
            "ONLY for ви-forms (source is ви/вас/вам/вами/ваш*): "
            "'genuine_plural' = addressing multiple people (nation, army, enemy collective, crowd); "
            "'polite_singular' = formal address to one identified person; "
            "'ambiguous' = cannot determine; "
            "'not_applicable' = all non-ви pronouns (ти-forms, 1st/3rd person, etc.)."
        ),
    )
    is_pro_drop: bool = Field(
        ...,
        description=(
            "True if subject pronoun is absent in source and inferred from verb morphology. "
            "Possessives are NEVER pro-drop."
        ),
    )
    source_mapping: str = Field(
        ...,
        description=(
            "Source pronoun as dictionary/lemma form (Cyrillic preferred). "
            "Append ' (IMPLIED)' if pro-drop. Append ' [RU]' if Russian source."
        ),
    )


class StanzaAnalysis(BaseModel):
    stanza_index: int = Field(
        ...,
        description="1-based index matching the input stanza numbering.",
    )
    shakespearean_segment: str = Field(
        ...,
        description="Shakespearean English translation of this stanza.",
    )
    pronouns_found: List[PronounAnalysis] = Field(
        ...,
        description="All PERSONAL pronouns in this stanza. Empty list if none.",
    )


class PoemStanzaAnnotation(BaseModel):
    stanzas: List[StanzaAnalysis] = Field(
        ...,
        description="One entry per input stanza, in the same order.",
    )


                                                                                 
         
                                                                                 

STANZA_PROMPT = """\
You are a Ukrainian linguistics specialist. You will receive numbered stanzas
from a contemporary Ukrainian, Russian, or Crimean Tatar poem (Facebook, 2014-2025).
Each stanza may contain multiple lines separated by newlines. For EACH stanza:

STEP 1 — Translate the entire stanza to Shakespearean English (dost, hath, art, etc.).
Preserve grammatical person/number. Render pro-drop subjects explicitly.

2nd-person mapping (STRICT — apply before translating):
  ти / тебе / тобі / твій / твоя / твоє / твої  →  thou / thee / thy / thine / thyself  (SINGULAR)
  ви / вас / вам / вами / ваш*                  →  ye / you / your / yours / yourselves  (PLURAL)
  Polite "ви" to one addressee still maps to ye/you/your — NEVER thou/thee/thy.

STEP 2 — Identify ALL personal pronouns in your translation.
For each, provide:
- english_pronoun: exact token (lowercase)
- person: "1st" | "2nd" | "3rd" | "Impersonal"
  "Impersonal" = Ukrainian bezosovovo construction ONLY
- number: MORPHOLOGICAL form only. CRITICAL RULE:
  ви-forms → ALWAYS "Plural" (even polite address to one person).
  ти-forms → ALWAYS "Singular".
  "None" → Impersonal only.
  NEVER set "Singular" for a ви/вас/вам/ваш* source form.
- vy_register: ONLY for ви-forms (source is ви/вас/вам/вами/ваш*):
  "genuine_plural"  — addressing multiple people. Cues:
      plural vocative ("друзі", "брати", "українці", "воїни");
      enemy as collective ("ви прийшли на нашу землю", addressing army/Russia);
      nation/community addressed as group;
      imperative plural forms.
  "polite_singular" — formal address to one identified person. Cues:
      named individual; intimate/romantic singular addressee;
      clearly one interlocutor in dialogue context.
  "ambiguous" — cannot determine from context.
  "not_applicable" — all non-ві pronouns (ти-forms, 1st/3rd person, etc.).
  DEFAULT BIAS: in wartime poetry, ви addressing unnamed collective entities
  (the enemy, the nation, soldiers) is genuine_plural, NOT polite_singular.
- is_pro_drop: true if subject pronoun absent in source, inferred from verb morphology.
  Possessives are NEVER pro-drop.
- source_mapping: dictionary/lemma form (Cyrillic). Append " (IMPLIED)" if pro-drop.
  Append " [RU]" if Russian source.

Return JSON matching PoemStanzaAnnotation. No extra text.\
"""


                                                                                 
                  
                                                                                 

def _build_chunk_mapping(
    stanza_texts: list[str],
    stanza_indices: list[int],
    max_lines: int = MAX_LINES_PER_CHUNK,
) -> list[tuple[int, str]]:
    """Returns list of (original_stanza_index, chunk_text) after splitting."""
    mapping: list[tuple[int, str]] = []
    for text, idx in zip(stanza_texts, stanza_indices):
        lines = text.split("\n")
        if len(lines) <= max_lines:
            mapping.append((idx, text))
        else:
            for start in range(0, len(lines), max_lines):
                chunk = "\n".join(lines[start: start + max_lines])
                if chunk.strip():
                    mapping.append((idx, chunk))
    return mapping


                                                                                 
            
                                                                                 

def _normalize_pronoun(raw: str) -> str:
    if not raw:
        return raw
    s = raw.strip().lower()
    s = re.sub(r"[\[\]()（）]", "", s)
    return s.strip()


def _get_temporal_period(year: Optional[int]) -> str:
    if year is None:
        return "unknown"
    if year < 2014:
        return "pre_2014"
    if year <= 2021:
        return "2014_2021"
    return "post_2022"


def _load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_checkpoint(poem_id: str):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(str(poem_id) + "\n")


def _append_raw_jsonl(record: dict):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_NULL_AS_EMPTY_FOR_CSV = frozenset({
    "pronoun_word", "person", "number", "vy_register",
    "is_pro_drop", "source_mapping", "qa_flag",
})


def _append_rows_to_csv(rows: list, write_header: bool):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_rows = [
        {
            k: ("" if (v is None and k in _NULL_AS_EMPTY_FOR_CSV) else v)
            for k, v in r.items()
        }
        for r in rows
    ]
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(out_rows)


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    p = PRICE.get(model)
    if p is None:
        return None
    return (prompt_tokens / 1000 * p["input"]
            + completion_tokens / 1000 * p["output"])


                                                                                 
                            
                                                                                 

_semaphore: asyncio.Semaphore | None = None


async def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with _semaphore:
                completion = await fn(*args, **kwargs)
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            }
            return completion, usage
        except RateLimitError as e:
            wait = BACKOFF_BASE ** attempt
            log.warning("rate_limit %s/%s %.1fs %s", attempt, MAX_RETRIES, wait, e)
            await asyncio.sleep(wait)
        except APITimeoutError as e:
            wait = BACKOFF_BASE ** attempt
            log.warning("timeout %s/%s %.1fs %s", attempt, MAX_RETRIES, wait, e)
            await asyncio.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                log.warning("server %s %s/%s %.1fs", e.status_code, attempt, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
            else:
                log.error("api %s %s", e.status_code, e)
                return None, None
        except Exception as e:
            log.error("%s", e)
            return None, None
    log.error("retries exhausted")
    return None, None


async def analyze_stanza_batch(
    chunks: list[str],
    poem_id: str,
) -> tuple[Optional[PoemStanzaAnnotation], Optional[dict]]:
    """Send a batch of text chunks (<=MAX_LINES each) to GPT."""
    if not chunks:
        return None, None

    numbered = "\n\n".join(
        f"[{i + 1}]\n{s}" for i, s in enumerate(chunks)
    )

    async def _call():
        return await client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": STANZA_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze ALL stanzas below. "
                        "Return one StanzaAnalysis per stanza, "
                        "preserving the stanza_index numbers.\n\n"
                        + numbered
                    ),
                },
            ],
            response_format=PoemStanzaAnnotation,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.1,
        )

    completion, usage = await _call_with_retry(_call)
    if completion is None:
        return None, None

    parsed = completion.choices[0].message.parsed
    raw = completion.choices[0].message.model_dump(warnings=False)

    _append_raw_jsonl({
        "type": "stanza_batch",
        "poem_id": str(poem_id),
        "chunk_count": len(chunks),
        "raw_response": raw,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if parsed is None:
        log.error("poem_id=%s parse None (truncated?)", poem_id)
        return None, usage

    return parsed, usage


                                                                                 
                
                                                                                 

_UKR_TOKEN = re.compile(r"[а-яіїєґ']+", re.IGNORECASE)
_THOU_FORMS = frozenset({"thou", "thee", "thy", "thine", "thyself"})


def _source_indicates_plural_vy(source: str) -> bool:
    s = source.strip().lower()
    for cut in (" (implied)", " [ru]"):
        if cut in s:
            s = s.split(cut)[0]
    tokens = _UKR_TOKEN.findall(s.replace("\u2019", "'"))
    if {"ви", "вас", "вам", "вами"}.intersection(tokens):
        return True
    return any(t.startswith("ваш") for t in tokens)


_VALID_PERSON_PRONOUN: dict[str, set[str]] = {
    "1st": {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
    },
    "2nd": {
        "thou", "thee", "thy", "thine", "thyself",
        "ye", "you", "your", "yours", "yourself", "yourselves",
    },
    "3rd": {
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    },
    "Impersonal": {"it", "its"},
}


def _validate_pronoun_row(row: dict) -> str:
    raw_pronoun = row.get("pronoun_word")
    if raw_pronoun is None:
        return "OK"

    person = row.get("person", "")
    number = row.get("number", "")
    pronoun = _normalize_pronoun(str(raw_pronoun))

    if "pro" in pronoun and "drop" in pronoun:
        return (
            "ERROR: english_pronoun contains pro-drop placeholder; "
            "model failed to render explicit pronoun"
        )

    source = str(row.get("source_mapping") or "")
    is_vy = _source_indicates_plural_vy(source)

    if is_vy and pronoun in _THOU_FORMS:
        return (
            "INCONSISTENT: source is ви-form (plural) but english_pronoun is "
            "singular thou-form"
        )

    if is_vy and number == "Singular":
        return "INCONSISTENT: source is ви-form but number='Singular' (must be 'Plural')"

    vy_reg = row.get("vy_register", "")
    if is_vy and vy_reg == "not_applicable":
        return "INCONSISTENT: source is ви-form but vy_register='not_applicable'"
    if not is_vy and vy_reg in ("genuine_plural", "polite_singular", "ambiguous"):
        return f"INCONSISTENT: source is not ви-form but vy_register='{vy_reg}'"

    valid_set = _VALID_PERSON_PRONOUN.get(person, set())
    if valid_set and pronoun not in valid_set:
        return f"INCONSISTENT: person={person} but pronoun='{pronoun}'"

    is_pro_drop = row.get("is_pro_drop")
    if is_pro_drop is True and "(IMPLIED)" not in source:
        return "INCONSISTENT: is_pro_drop=True but source_mapping missing '(IMPLIED)'"
    if is_pro_drop is False and "(IMPLIED)" in source:
        return "INCONSISTENT: is_pro_drop=False but source_mapping contains '(IMPLIED)'"

    return "OK"


                                                                                 
               
                                                                                 

def _year_from_date(val) -> Optional[int]:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if len(s) < 4:
        return None
    try:
        y = int(s[:4])
        return y if 1000 <= y <= 2100 else None
    except ValueError:
        return None


def load_data(
    layer1_path: Path,
    layer0_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load stanza data and build poem-level metadata lookup."""
    if not layer1_path.exists():
        raise FileNotFoundError(f"Stanza file not found: {layer1_path}")
    if not layer0_path.exists():
        raise FileNotFoundError(f"Poem file not found: {layer0_path}")

    stanzas = pd.read_csv(layer1_path, low_memory=False)
    for col in ("poem_id", "stanza_index", "stanza_text", "author"):
        if col not in stanzas.columns:
            raise ValueError(f"layer1 CSV missing column {col!r}")
    stanzas = stanzas[stanzas["stanza_text"].notna()].copy()

    usecols = [
        "poem_id", "Date posted", "Language",
        "Poem full text (copy and paste)",
        "Is repeat",
        "I.D. of original (if poem is a translation)",
    ]
    poems = pd.read_csv(layer0_path, usecols=usecols, low_memory=False)
    poems = poems.rename(columns={
        "Date posted": "date",
        "Language": "language",
        "Poem full text (copy and paste)": "full_poem_text",
        "Is repeat": "is_repeat_raw",
        "I.D. of original (if poem is a translation)": "original_id",
    })
    poems["year"] = poems["date"].map(_year_from_date)
    poems["temporal_period"] = poems["year"].map(_get_temporal_period)
    poems["is_repeat"] = poems["is_repeat_raw"].str.lower().eq("yes").fillna(False)
    poems["is_translation"] = poems["original_id"].notna()

    poem_meta = {}
    for _, row in poems.iterrows():
        pid = str(row["poem_id"])
        poem_meta[pid] = {
            "year":            row["year"],
            "temporal_period": row["temporal_period"],
            "language":        row.get("language", ""),
            "full_poem_text":  row.get("full_poem_text", ""),
            "is_repeat":       bool(row["is_repeat"]),
            "is_translation":  bool(row["is_translation"]),
        }

    return stanzas, poem_meta


                                                                                 
                              
                                                                                 

async def process_poem(
    pid_str: str,
    grp: pd.DataFrame,
    meta: dict,
) -> tuple[str, list[dict], dict]:
    """Annotate one poem."""
    usage_delta = {"prompt": 0, "completion": 0}

    stanza_texts: list[str] = []
    stanza_indices: list[int] = []
    for _, row in grp.iterrows():
        txt = str(row["stanza_text"]).strip()
        if not txt:
            continue
        stanza_texts.append(txt)
        stanza_indices.append(int(row["stanza_index"]))

    if not stanza_texts:
        return pid_str, [], usage_delta

    chunk_mapping = _build_chunk_mapping(stanza_texts, stanza_indices)
    chunk_texts = [t for _, t in chunk_mapping]

    analysis, usage = await analyze_stanza_batch(chunk_texts, pid_str)
    if usage:
        usage_delta["prompt"] += usage["prompt_tokens"]
        usage_delta["completion"] += usage["completion_tokens"]

    if not analysis:
        log.warning("poem_id=%s batch failed", pid_str)
        return pid_str, [], usage_delta

    sa_by_batch_idx: dict[int, StanzaAnalysis] = {
        sa.stanza_index: sa for sa in analysis.stanzas
    }

    poem_rows: list[dict] = []
    shakespeare_segs: list[str] = []
    author = grp.iloc[0]["author"] if "author" in grp.columns else ""

    for batch_idx, (orig_idx, chunk_text) in enumerate(chunk_mapping, start=1):
        sa = sa_by_batch_idx.get(batch_idx)
        if sa is None:
            log.warning(
                "poem_id=%s missing batch_index=%d (stanza %d)",
                pid_str,
                batch_idx,
                orig_idx,
            )
            continue

        en_seg = sa.shakespearean_segment or ""
        shakespeare_segs.append(en_seg)

        base = {
            "poem_id":          pid_str,
            "author":           author,
            "language":         meta.get("language", ""),
            "year":             meta.get("year"),
            "temporal_period":  meta.get("temporal_period", "unknown"),
            "is_repeat":        meta.get("is_repeat", False),
            "is_translation":   meta.get("is_translation", False),
            "stanza_index":     orig_idx,
            "stanza_ukr":       chunk_text,
            "stanza_en":        en_seg,
            "full_shakespeare_text": "",
        }

        if not sa.pronouns_found:
            poem_rows.append({
                **base,
                "pronoun_word":   None,
                "person":         None,
                "number":         None,
                "vy_register":    None,
                "is_pro_drop":    None,
                "source_mapping": None,
                "qa_flag":        "OK",
            })
        else:
            for pr in sa.pronouns_found:
                row_dict = {
                    **base,
                    "pronoun_word":   _normalize_pronoun(pr.english_pronoun),
                    "person":         pr.person,
                    "number":         pr.number,
                    "vy_register":    pr.vy_register,
                    "is_pro_drop":    pr.is_pro_drop,
                    "source_mapping": pr.source_mapping,
                    "qa_flag":        "",
                }
                row_dict["qa_flag"] = _validate_pronoun_row(row_dict)
                poem_rows.append(row_dict)

    full_text = " // ".join(s for s in shakespeare_segs if s)
    for r in poem_rows:
        r["full_shakespeare_text"] = full_text

    return pid_str, poem_rows, usage_delta


                                                                                 
       
                                                                                 

async def async_main():
    global _semaphore, MODEL

    parser = argparse.ArgumentParser(
        description="GPT stanza-level pronoun annotation (async).",
    )
    parser.add_argument("--layer1", type=Path, default=LAYER1_CSV)
    parser.add_argument("--layer0", type=Path, default=LAYER0_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"OpenAI model id (default: {MODEL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
        help=f"Max parallel API calls (default: {CONCURRENCY})",
    )
    args = parser.parse_args()
    MODEL = args.model

    if args.output_dir is not None:
        configure_output_dir(args.output_dir.resolve())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _semaphore = asyncio.Semaphore(args.concurrency)

    fh = logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    try:
        stanzas_df, poem_meta = load_data(args.layer1, args.layer0)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        return

    grouped = {
        pid: grp.sort_values("stanza_index")
        for pid, grp in stanzas_df.groupby("poem_id", sort=False)
    }
    poem_ids = list(grouped.keys())
    log.info("%d stanzas, %d poems", len(stanzas_df), len(poem_ids))

    done_ids = _load_checkpoint()
    todo = [pid for pid in poem_ids if str(pid) not in done_ids]
    log.info("checkpoint: %d done, %d to run", len(done_ids), len(todo))

    need_header = not OUTPUT_CSV.exists()
    usage_counter = {"prompt": 0, "completion": 0}

    tasks = {
        asyncio.create_task(
            process_poem(str(pid), grouped[pid], poem_meta.get(str(pid), {}))
        ): pid
        for pid in todo
    }

    done_count = 0
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="poems"):
        pid_str, rows, usage_delta = await coro
        usage_counter["prompt"] += usage_delta["prompt"]
        usage_counter["completion"] += usage_delta["completion"]

        if rows:
            _append_rows_to_csv(rows, write_header=need_header)
            need_header = False
            log.info("poem_id=%s wrote %d rows", pid_str, len(rows))

        _save_checkpoint(pid_str)
        done_count += 1

    total_tok = usage_counter["prompt"] + usage_counter["completion"]
    cost = _calc_cost(MODEL, usage_counter["prompt"], usage_counter["completion"])
    if cost is None:
        cost_line = "  Cost (USD):        N/A (pricing not configured for this model)"
    else:
        cost_line = f"  Cost (USD):        ${cost:.4f}"
    lines = [
        "=== Token usage ===",
        f"Run time (UTC):    {datetime.now(timezone.utc).isoformat()}",
        f"[{MODEL}]",
        f"  Prompt tokens:     {usage_counter['prompt']:,}",
        f"  Completion tokens: {usage_counter['completion']:,}",
        f"  Total tokens:      {total_tok:,}",
        cost_line,
        f"  Poems processed:   {done_count}",
    ]
    report = "\n".join(lines)
    log.info("\n" + report)
    with open(COST_REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(report + "\n\n")

    if OUTPUT_CSV.exists():
        result_df = pd.read_csv(OUTPUT_CSV)
        if "qa_flag" in result_df.columns:
            qa_issues = result_df[result_df["qa_flag"] != "OK"]
            total = len(result_df[result_df["pronoun_word"].notna()])
            if len(qa_issues) > 0:
                log.warning("QA issues %d/%d", len(qa_issues), total)
            else:
                log.info("QA ok %d rows", total)
        log.info("saved %s (%d rows)", OUTPUT_CSV, len(result_df))


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
