"""Batch GPT annotation: pronouns per sentence + poem-level perspective."""

import argparse
import csv
import json
import os
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field
from tqdm import tqdm
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

MANUAL_ANNOTATION_FILES = [
    _ROOT / "data" / "manual_annotation_result_Junyu.csv",
    _ROOT / "data" / "manual_annotation_result_junyu1.csv",
]

PUBLIC_PRONOUN_DETAILED = (
    _ROOT / "data" / "processed" / "ukrainian_pronouns_detailed_public_list.csv"
)

_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "01_pronouns_detection"
_PUBLIC_RUN_OUTPUT_DIR = _ROOT / "data" / "processed" / "gpt_annotation_public_run"

OUTPUT_DIR = _DEFAULT_OUTPUT_DIR
OUTPUT_CSV = OUTPUT_DIR / "gpt_annotation_detailed.csv"
OUTPUT_JSONL = OUTPUT_DIR / "gpt_annotation_raw.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_done_ids.txt"
COST_REPORT_FILE = OUTPUT_DIR / "token_usage_report.txt"


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSONL, CHECKPOINT_FILE, COST_REPORT_FILE
    OUTPUT_DIR = output_dir
    OUTPUT_CSV = OUTPUT_DIR / "gpt_annotation_detailed.csv"
    OUTPUT_JSONL = OUTPUT_DIR / "gpt_annotation_raw.jsonl"
    CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_done_ids.txt"
    COST_REPORT_FILE = OUTPUT_DIR / "token_usage_report.txt"


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

MODEL_SENTENCE = "gpt-4o-mini"
MODEL_PERSPECTIVE = "gpt-4o"

PRICE = {
    "gpt-4o":      {"input": 0.005,    "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015,  "output": 0.00060},
}

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
            "If the pronoun is pro-dropped in Ukrainian, render the INFERRED "
            "English pronoun explicitly (e.g., 'I', 'we') — NEVER write '(pro-drop)' here. "
            "ONLY personal pronouns: I me my mine myself / we us our ours ourselves / "
            "thou thee thy thine thyself / ye you your yours yourself yourselves / "
            "he him his himself / she her hers herself / it its itself / "
            "they them their theirs themselves. "
            "DO NOT include relative/demonstrative/indefinite pronouns: "
            "that, which, who (relative), whom (relative), whose (relative), "
            "none, what, whatever, whoever, whichever."
        ),
    )
    person: Literal["1st", "2nd", "3rd", "Impersonal"] = Field(
        ...,
        description=(
            "Impersonal ONLY for Ukrainian impersonal verb constructions "
            "(bezosovovi diyeslova) with no personal subject."
        ),
    )
    number: Literal["Singular", "Plural", "None"] = Field(
        ...,
        description="None only for genuinely impersonal constructions.",
    )
    is_pro_drop: bool = Field(
        ...,
        description=(
            "True if subject pronoun is absent in Ukrainian source "
            "and inferred solely from verb morphology. "
            "Possessives are NEVER pro-drop."
        ),
    )
    source_mapping: str = Field(
        ...,
        description=(
            "Ukrainian pronoun as dictionary/lemma form aligned with the source line "
            "(Cyrillic preferred; stay consistent). "
            "Append ' (IMPLIED)' if pro-drop. Append ' [RU]' if Russian source."
        ),
    )


class SentenceAnalysis(BaseModel):
    sentence_index: int = Field(
        ...,
        description="1-based index matching the input sentence numbering.",
    )
    shakespearean_segment: str = Field(
        ...,
        description="Shakespearean English translation of this sentence.",
    )
    pronouns_found: List[PronounAnalysis] = Field(
        ...,
        description=(
            "All PERSONAL pronouns in this sentence. "
            "Return an empty list if none are present."
        ),
    )


class PoemSentenceAnnotation(BaseModel):
    sentences: List[SentenceAnalysis] = Field(
        ...,
        description="One entry per input sentence, in the same order.",
    )


PerspectiveType = Literal[
    "1st person singular", "1st person plural",
    "2nd person singular", "2nd person plural",
    "3rd person singular", "3rd person plural",
    "Impersonal/Other",
]

WeInclusivity = Literal[
    "inclusive", "exclusive", "ambiguous", "not_applicable",
]

ReferentCategory = Literal[
    "SELF", "INTIMATE", "LOCAL", "NATION",
    "SUPRANATIONAL", "OTHER", "UNCERTAIN",
]

AddresseeType = Literal[
    "specific_individual", "collective_nation", "enemy_other",
    "europe_world", "absent_beloved", "lyric_self_2nd",
    "god_nature_abstract", "not_applicable",
]

PolyphonyType = Literal[
    "monologic", "mediated_polyphony", "choral_polyphony",
]


class PoemPerspective(BaseModel):
    poem_perspective_primary: PerspectiveType = Field(...)
    poem_perspective_secondary: Optional[PerspectiveType] = Field(
        default=None,
        description=(
            "Fill only if a second perspective occupies ≥30% of lines "
            "and marks a deliberate shift. Never equal to primary. "
            "Null otherwise."
        ),
    )
    we_inclusivity: WeInclusivity = Field(...)
    dominant_referent_category: ReferentCategory = Field(
        ...,
        description=(
            "Social collective or person that the lyric speaker's 1st-person "
            "pronouns (я, ми, нас, нам, …) primarily refer to. "
            "Ignore 3rd-person referents (e.g. enemy as 'вони'); those are not this field."
        ),
    )
    addressee_type: AddresseeType = Field(...)
    polyphony_type: PolyphonyType = Field(...)


SENTENCE_PROMPT = """\
You are a Ukrainian linguistics specialist. You will receive a numbered list of sentences
from a contemporary Ukrainian poem (Facebook, 2014-2025). For EACH sentence:

STEP 1 — Translate to Shakespearean English (dost, hath, art, etc.).
Preserve grammatical person/number. Render pro-drop subjects explicitly in translation.

2nd-person mapping (STRICT — apply before translating):
  ти / тебе / тобі / твій / твоя / твоє / твої  →  thou / thee / thy / thine / thyself  (SINGULAR)
  ви / вас / вам / вами / ваш*                  →  ye / you / your / yours / yourselves  (PLURAL)
  Polite "ви" to one addressee still maps to ye/you/your — NEVER thou/thee/thy.

STEP 2 — Identify ONLY PERSONAL PRONOUNS in your translation.

NOT PRONOUNS — NEVER tag these:
  Relative:      that  which  who  whom  whose  (when introducing a clause)
  Demonstrative: this  these  those
  Indefinite:    none  any  some  each  every  one  what  whatever  whoever

For each personal pronoun, provide:
- english_pronoun: exact token (lowercase)
- person: "1st" | "2nd" | "3rd" | "Impersonal"
  "Impersonal" = Ukrainian bezosovovo construction ONLY (no personal subject)
- number: "Singular" | "Plural" | "None" (None = Impersonal only)
- is_pro_drop: true if the Ukrainian subject pronoun is absent and inferred solely from verb morphology. Possessives are NEVER pro-drop.
- source_mapping: Ukrainian dictionary/lemma form for the pronoun (Cyrillic preferred). Append " (IMPLIED)" if pro-drop. Append " [RU]" if Russian source.

Return JSON matching PoemSentenceAnnotation. No extra text.\
"""

PERSPECTIVE_PROMPT = """\
You are an expert in Ukrainian poetry, Critical Discourse Analysis, and wartime
identity construction. Analyze the narrative perspective of an entire contemporary
Ukrainian poem (2014-2025, sourced from Facebook).

TASK 1 — poem_perspective_primary (ONE value only; "Mixed" is NOT valid):
  "1st person singular"  "1st person plural"
  "2nd person singular"  "2nd person plural"
  "3rd person singular"  "3rd person plural"
  "Impersonal/Other"

DECISION RULE:
  - Primary = the grammatical person of the LYRIC SPEAKER'S subject position
    that dominates ≥50% of content lines.
  - If no single perspective dominates by line count, choose the perspective
    of the opening and closing stanzas (framing effect).
  - You must commit to one value. "Mixed" is not an option.

TASK 2 — poem_perspective_secondary (optional):
  Same value set as Task 1, or null.
  Fill ONLY if a second perspective occupies ≥30% of lines AND marks a
  deliberate shift in lyric subject position (not brief pronoun alternation
  within a stanza). Never equal to poem_perspective_primary.

TASK 3 — we_inclusivity (evaluate only if 1st plural "ми/нас/нам" present):
  Step 1: Identify the addressee — is there a "ти/ви", a vocative
          ("друзі", "брати і сестри"), or an implied public audience?
    - If no addressee is identifiable (no 2nd person, no vocative, no clear
      implied audience), set we_inclusivity to "ambiguous" (handbook default
      when inclusive/exclusive cannot be grounded). Skip Step 2.
  Step 2: Only if Step 1 found an addressee: does "ми" invite that addressee into the group?
    - Yes → "inclusive"
      (cues: "ми всі", "разом з вами", calls to collective action,
       "давайте", "згадаймо")
    - No — "ми" is a subgroup contrasted with or separate from addressee
      → "exclusive"
      (cues: "ми стоїмо, а ви…", soldier vs civilian, "ми, поети")
    - Cannot resolve / deliberately open → "ambiguous"
  Step 3: No 1st plural at all → "not_applicable"

TASK 4 — dominant_referent_category:
  ONLY what the lyric speaker's 1st-person pronouns (я, ми, нас, нам, …)
  refer to — not 3rd-person referents (e.g. "вони" = occupiers is OUT OF SCOPE).
  STATE and ENEMY as referents of "ми/я" are extremely rare; do not use this field
  to encode the poem's main enemy or the state unless they are clearly the
  referent of the speaker's own 1st-person forms.

  Categories:
    "SELF"          – poet as individual person
    "INTIMATE"      – family, partner, close friends
                      (cues: "мама", "діти", "коханий/а", "ми з тобою")
    "LOCAL"         – city, community, military unit
                      (cues: place names + "ми", "наша рота", "наш батальйон")
    "NATION"        – Ukrainians as national collective
                      (cues: "українці", "народ", "ми – українці", "наша земля")
    "SUPRANATIONAL" – Europe, world, humanity
                      (cues: "Європа", "людство", "ми всі на цій планеті")
    "OTHER"         – identifiable group not covered above
                      ("ми, жінки"; "ми, біженці")
    "UNCERTAIN"     – unresolvable after reading full poem

  Decision procedure (1st-person referent only):
    1. Explicit national markers ("українці", "народ", "нація") → NATION
    2. Kinship/family terms → INTIMATE
    3. Local spatial markers or unit language → LOCAL
    4. Global/humanity frame → SUPRANATIONAL
    5. Predominantly singular 1st-person (я/мене/мій) in personal or introspective
       register with no collective reference → SELF
    6. Identifiable group not covered above (occupational, gender, refugee, etc.)
       → OTHER
    7. Cannot determine → UNCERTAIN

  Boundary rules:
    - "наші діти" in personal narrative → INTIMATE;
      in generational/national narrative → NATION
    - City name + national framing ("ми, кияни, захищаємо Україну") → NATION
    - City name only + no national markers → LOCAL

TASK 5 — addressee_type (only if 2nd person present; else "not_applicable"):
  "specific_individual"  – named or clearly identified single person
  "collective_nation"    – Ukrainian people addressed as a whole
  "enemy_other"          – occupiers, Russia addressed directly
  "europe_world"         – Europe, international community
  "absent_beloved"       – dead, missing, or separated intimate person
  "lyric_self_2nd"       – speaker addresses themselves in 2nd person
                           (apostrophe to self; "ти витримаєш, серце"
                            where "ти" = speaker's own self)
  "god_nature_abstract"  – divine, nature, abstract forces
  "not_applicable"       – no 2nd person in poem

  Decision rule: if multiple 2nd-person addressees exist, choose the one
  that receives the dominant emotional/rhetorical weight (most lines, climactic
  position, or strongest modal investment).
  If a named/identified individual is deceased or absent → "absent_beloved"
  takes priority over "specific_individual".

TASK 6 — polyphony_type:
  "monologic"          – single stable lyric voice throughout
  "mediated_polyphony" – central narrator quotes or collects other voices
                         (cues: reported speech, "вона сказала", "він писав")
  "choral_polyphony"   – multiple voices with equal status, no dominant
                         narrator mediating between them
  Note: "choral_polyphony" is rare in short lyric poems; most Facebook
  war poetry is "monologic". Use "choral_polyphony" only when there is
  genuinely no identifiable central narrator.

Return JSON matching PoemPerspective. No extra text.\
"""


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
        return "unknown"
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
    "pronoun_word", "person", "number", "is_pro_drop", "source_mapping", "qa_flag",
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


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICE.get(model, PRICE["gpt-4o"])
    return (prompt_tokens / 1000 * p["input"]
            + completion_tokens / 1000 * p["output"])


def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = fn(*args, **kwargs)
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            }
            return completion, usage
        except RateLimitError as e:
            wait = BACKOFF_BASE ** attempt
            log.warning("rate_limit %s/%s %.1fs %s", attempt, MAX_RETRIES, wait, e)
            time.sleep(wait)
        except APITimeoutError as e:
            wait = BACKOFF_BASE ** attempt
            log.warning("timeout %s/%s %.1fs %s", attempt, MAX_RETRIES, wait, e)
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                log.warning("server %s %s/%s %.1fs", e.status_code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                log.error("api %s %s", e.status_code, e)
                return None, None
        except Exception as e:
            log.error("%s", e)
            return None, None
    log.error("retries exhausted")
    return None, None


def analyze_poem_sentences_with_gpt(
    sentences: list[str],
    poem_id: str,
) -> tuple[Optional[PoemSentenceAnnotation], Optional[dict]]:
    if not sentences:
        return None, None

    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sentences))

    def _call():
        return client.beta.chat.completions.parse(
            model=MODEL_SENTENCE,
            messages=[
                {"role": "system", "content": SENTENCE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze ALL sentences below. "
                        "Return one SentenceAnalysis per sentence, "
                        "preserving the sentence_index numbers.\n\n"
                        + numbered
                    ),
                },
            ],
            response_format=PoemSentenceAnnotation,
            temperature=0.1,
        )

    completion, usage = _call_with_retry(_call)
    if completion is None:
        return None, None

    parsed = completion.choices[0].message.parsed
    raw = completion.choices[0].message.model_dump(warnings=False)

    _append_raw_jsonl({
        "type": "sentence_batch",
        "poem_id": str(poem_id),
        "sentence_count": len(sentences),
        "raw_response": raw,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if parsed is None:
        log.error("poem_id=%s sentence parse None", poem_id)
        return None, usage

    return parsed, usage


def analyze_perspective_with_gpt(
    full_poem_text: str,
    poem_id: str,
) -> tuple[Optional[PoemPerspective], Optional[dict]]:
    def _call():
        return client.beta.chat.completions.parse(
            model=MODEL_PERSPECTIVE,
            messages=[
                {"role": "system", "content": PERSPECTIVE_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze the perspective of this Ukrainian poem:\n\n{full_poem_text}",
                },
            ],
            response_format=PoemPerspective,
            temperature=0.1,
        )

    completion, usage = _call_with_retry(_call)
    if completion is None:
        return None, None

    parsed = completion.choices[0].message.parsed
    raw = completion.choices[0].message.model_dump(warnings=False)

    _append_raw_jsonl({
        "type": "perspective",
        "poem_id": str(poem_id),
        "raw_response": raw,
        "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if parsed is None:
        log.error("poem_id=%s perspective parse None", poem_id)
        return None, usage

    return parsed, usage


_UKR_TOKEN = re.compile(r"[а-яіїєґ']+", re.IGNORECASE)
_THOU_FORMS = frozenset({"thou", "thee", "thy", "thine", "thyself"})


def _source_indicates_plural_vy(source: str) -> bool:
    s = source.strip().lower()
    for cut in (" (implied)", " [ru]"):
        if cut in s:
            s = s.split(cut)[0]
    tokens = _UKR_TOKEN.findall(s.replace("'", "’"))
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
    pronoun = _normalize_pronoun(str(raw_pronoun))

    if "pro" in pronoun and "drop" in pronoun:
        return (
            "ERROR: english_pronoun contains pro-drop placeholder; "
            "model failed to render explicit pronoun"
        )

    source = str(row.get("source_mapping") or "")
    if _source_indicates_plural_vy(source) and pronoun in _THOU_FORMS:
        return (
            "INCONSISTENT: source is ви-form (plural) but english_pronoun is "
            "singular thou-form"
        )

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


def load_annotation_sentences(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        if path.exists():
            dfs.append(pd.read_csv(path, low_memory=False))
    if not dfs:
        missing = ", ".join(str(p) for p in paths)
        raise FileNotFoundError(f"Input file(s) not found: {missing}")
    df = pd.concat(dfs, ignore_index=True)
    for col in ("ID", "context", "author", "text"):
        if col not in df.columns:
            raise ValueError(
                f"CSV missing required column {col!r} (have: {list(df.columns)})"
            )
    unique = df.drop_duplicates(subset=["ID", "context"]).copy()
    unique = unique[
        unique["context"].notna() & (unique["context"].astype(str).str.strip() != "")
    ]
    if "year" not in unique.columns and "date" in unique.columns:
        unique = unique.copy()
        unique["year"] = unique["date"].map(_year_from_date)
    elif "year" not in unique.columns:
        unique = unique.copy()
        unique["year"] = None
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="GPT pronoun + poem perspective annotation (manual or public list)."
    )
    env_src = os.environ.get("GPT_ANNOTATION_SOURCE", "manual")
    parser.add_argument(
        "--source",
        choices=("manual", "public"),
        default=env_src if env_src in ("manual", "public") else "manual",
        help=(
            "manual: annotator CSVs; public: ukrainian_pronouns_detailed_public_list.csv. "
            "Override with env GPT_ANNOTATION_SOURCE."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: manual→outputs/01_pronouns_detection, "
            "public→data/processed/gpt_annotation_public_run)."
        ),
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        configure_output_dir(args.output_dir.resolve())
    elif args.source == "public":
        configure_output_dir(_PUBLIC_RUN_OUTPUT_DIR)
    else:
        configure_output_dir(_DEFAULT_OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    if args.source == "public":
        input_paths = [PUBLIC_PRONOUN_DETAILED]
        log.info("source=public %s", PUBLIC_PRONOUN_DETAILED)
    else:
        input_paths = list(MANUAL_ANNOTATION_FILES)
        log.info("source=manual %s", input_paths)
    log.info("output_dir=%s", OUTPUT_DIR)

    try:
        df_sentences = load_annotation_sentences(input_paths)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        return

    poems = df_sentences.drop_duplicates(subset=["ID"])[["ID", "author", "text"]].copy()

    year_by_id: dict = {}
    if "year" in df_sentences.columns:
        year_by_id = (
            df_sentences.drop_duplicates(subset=["ID"])
            .set_index("ID")["year"]
            .to_dict()
        )

    log.info("%d sentences, %d poems", len(df_sentences), len(poems))

    done_ids = _load_checkpoint()
    poems_to_run = poems[~poems["ID"].astype(str).isin(done_ids)]
    log.info("checkpoint: %d done, %d to run", len(done_ids), len(poems_to_run))

    need_header = not OUTPUT_CSV.exists()

    usage_counter: dict[str, dict[str, int]] = {
        MODEL_SENTENCE: {"prompt": 0, "completion": 0},
        MODEL_PERSPECTIVE: {"prompt": 0, "completion": 0},
    }

    log.info("=== PHASE 1: perspective ===")
    perspective_by_id: dict = {}
    perspective_attempted_ids: set[str] = set()

    for _, row in tqdm(poems_to_run.iterrows(), total=len(poems_to_run), desc="perspective"):
        pid = str(row["ID"])
        text = row.get("text", "")
        if pd.isna(text) or not str(text).strip():
            continue
        perspective_attempted_ids.add(pid)
        p, usage = analyze_perspective_with_gpt(str(text), pid)
        if usage:
            usage_counter[MODEL_PERSPECTIVE]["prompt"] += usage["prompt_tokens"]
            usage_counter[MODEL_PERSPECTIVE]["completion"] += usage["completion_tokens"]
        if p:
            perspective_by_id[pid] = {
                "primary":                    p.poem_perspective_primary or "",
                "secondary":                  p.poem_perspective_secondary,
                "we_inclusivity":             p.we_inclusivity or "",
                "dominant_referent_category": p.dominant_referent_category or "",
                "addressee_type":             p.addressee_type or "",
                "polyphony_type":             p.polyphony_type or "",
            }

    log.info("=== PHASE 2: sentences ===")

    for pid_str in tqdm(poems_to_run["ID"].astype(str).tolist(), desc="poems"):
        pid_str = str(pid_str)
        if (
            pid_str in perspective_attempted_ids
            and pid_str not in perspective_by_id
        ):
            log.warning("poem_id=%s no perspective, skip phase2 (no checkpoint)", pid_str)
            continue

        poem_sentences = df_sentences[df_sentences["ID"].astype(str) == pid_str]
        pers = perspective_by_id.get(pid_str, {})

        try:
            raw_id = int(float(pid_str))
        except (ValueError, TypeError):
            raw_id = pid_str
        raw_year = year_by_id.get(raw_id)
        try:
            year_int = int(raw_year) if raw_year and not pd.isna(raw_year) else None
        except (ValueError, TypeError):
            year_int = None
        temporal_period = _get_temporal_period(year_int)

        sentence_list: list[str] = []
        sentence_meta: list[dict] = []
        for _, row in poem_sentences.iterrows():
            ctx = row["context"]
            if pd.isna(ctx) or not str(ctx).strip():
                continue
            sentence_list.append(str(ctx).strip())
            sentence_meta.append({
                "author": row.get("author", ""),
                "context_ukr": str(ctx).strip(),
            })

        if not sentence_list:
            _save_checkpoint(pid_str)
            continue

        analysis, usage = analyze_poem_sentences_with_gpt(sentence_list, pid_str)
        if usage:
            usage_counter[MODEL_SENTENCE]["prompt"] += usage["prompt_tokens"]
            usage_counter[MODEL_SENTENCE]["completion"] += usage["completion_tokens"]

        if not analysis:
            log.warning("poem_id=%s sentence batch failed", pid_str)
            _save_checkpoint(pid_str)
            continue

        sa_by_index: dict[int, SentenceAnalysis] = {
            sa.sentence_index: sa for sa in analysis.sentences
        }

        poem_rows: list[dict] = []
        shakespeare_segs: list[str] = []

        for i, meta in enumerate(sentence_meta):
            idx = i + 1
            sa = sa_by_index.get(idx)

            if sa is None:
                log.warning("poem_id=%s no sentence_index=%s", pid_str, idx)
                continue

            en_seg = sa.shakespearean_segment or ""
            shakespeare_segs.append(en_seg)

            base = {
                "original_id":                   pid_str,
                "author":                        meta["author"],
                "year":                          year_int,
                "temporal_period":               temporal_period,
                "full_shakespeare_text":         "",
                "context_sentence_en":           en_seg,
                "context_sentence_ukr":          meta["context_ukr"],
                "poem_perspective_primary":      pers.get("primary", ""),
                "poem_perspective_secondary":    pers.get("secondary"),
                "we_inclusivity":                pers.get("we_inclusivity", ""),
                "dominant_referent_category":    pers.get("dominant_referent_category", ""),
                "addressee_type":                pers.get("addressee_type", ""),
                "polyphony_type":                pers.get("polyphony_type", ""),
            }

            if not sa.pronouns_found:
                poem_rows.append({
                    **base,
                    "pronoun_word":    None,
                    "person":          None,
                    "number":          None,
                    "is_pro_drop":     None,
                    "source_mapping":  None,
                    "qa_flag":         "OK",
                })
            else:
                for pr in sa.pronouns_found:
                    row_dict = {
                        **base,
                        "pronoun_word":    _normalize_pronoun(pr.english_pronoun),
                        "person":          pr.person,
                        "number":          pr.number,
                        "is_pro_drop":     pr.is_pro_drop,
                        "source_mapping":  pr.source_mapping,
                        "qa_flag":         "",
                    }
                    row_dict["qa_flag"] = _validate_pronoun_row(row_dict)
                    poem_rows.append(row_dict)

        full_text = " ".join(s for s in shakespeare_segs if s)
        for r in poem_rows:
            r["full_shakespeare_text"] = full_text

        _append_rows_to_csv(poem_rows, write_header=need_header)
        need_header = False
        _save_checkpoint(pid_str)
        log.info("poem_id=%s wrote %d rows", pid_str, len(poem_rows))

    total_cost = sum(
        _calc_cost(model, v["prompt"], v["completion"])
        for model, v in usage_counter.items()
    )
    lines = [
        "=== Token usage ===",
        f"Run time (UTC):    {datetime.now(timezone.utc).isoformat()}",
    ]
    for model, v in usage_counter.items():
        total_tok = v["prompt"] + v["completion"]
        cost = _calc_cost(model, v["prompt"], v["completion"])
        lines += [
            f"[{model}]",
            f"  Prompt tokens:     {v['prompt']:,}",
            f"  Completion tokens: {v['completion']:,}",
            f"  Total tokens:      {total_tok:,}",
            f"  Cost (USD):        ${cost:.4f}",
        ]
    lines.append(f"Total est. cost (USD): ${total_cost:.4f}")
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
        log.info("raw jsonl %s", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
