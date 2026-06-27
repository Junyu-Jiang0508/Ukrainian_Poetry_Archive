"""Source-side 1st/2nd-person pronoun detection for Ukrainian/Russian poetry.

Detects the four morphological cells ``{1sg, 1pl, 2sg, 2pl}`` DIRECTLY from the
Ukrainian/Russian source text, with no English-translation intermediary. Replacement
for the translate-then-tag GPT pipeline in ``src/01_annotation_gpt_annotation.py``
(see docs/methodology_source_pronoun_detection.md).

Two engines, selectable with ``--mode`` / ``--compare-modes``:

  v1  Stanza ``tokenize,pos,lemma`` + sentence-level pro-drop suppression heuristic.
      Past tense is always excluded (AMBIGUOUS_PAST). Kept as the baseline.

  v2  Stanza ``tokenize,pos,lemma,depparse`` — highest precision:
        * nsubj attribution: a finite verb is pro-drop only when it has NO ``nsubj``
          child (true dependency edge, not a sentence-level guess).
        * explicit pron policy: personal pronouns are counted in ALL roles
          (subject/object/oblique), each tagged with its ``syntactic_role`` (deprel).
        * past tense layered: excluded by default, but PROMOTED to a cell when a
          coordinated (``conj``) clause supplies the person — flagged
          ``PAST_PROMOTED_<rule>``. No silent guessing.
        * imperative number fix: ``-те/-іть/-йте`` surface suffix overrides Stanza's
          often-wrong imperative ``Number`` (``IMPERATIVE_NUMBER_FIXED``).
        * false-positive filter: finite verbs in nominal deprel positions
          (noun mis-tagged as verb) are rejected (``FILTERED_NONPREDICATE``).

  gpt The existing GPT annotation, used only as a comparison column (NOT ground truth).

Possessives in both engines: cell from the *lemma* (мій/наш/твій/ваш | мой/твой),
never from the DET ``Number`` feat (which agrees with the possessed noun).

Scope: ``language in {Ukrainian, Russian}`` only (pipeline chosen by the ``language``
value, never the column name — the Russian source text lives in ``stanza_ukr``).
``ви/вы`` is always morphological ``2pl`` (no polite-singular split here). No 3rd
person, referent, inclusivity. No LLM/translation anywhere in the detection path.

Dependencies::

    python -c "import stanza; stanza.download('uk'); stanza.download('ru')"

Run::

    PYTHONPATH=src python src/01_annotation_source_pronoun_detection.py \
        --sample 50 --seed 42 --compare-modes v1,v2,gpt

Outputs under ``outputs/01_annotation_source_pronoun_detection/``:
    tokens_<mode>_<lang>.csv   token-level detections (v1/v2)
    stanza_cell_counts_v2.csv  per-stanza 4-cell counts (and per requested mode)
    diff_v1_vs_v2.csv          per-stanza v1 vs v2 deltas
    diff_v2_vs_gpt.csv         per-stanza v2 vs GPT deltas
    ambiguous_past_audit.csv   every past-tense verb: excluded vs promoted + evidence
    error_analysis.csv         v2 rows the filters rejected / corrected
    eval_gold_metrics.csv      P/R/F1 vs the hand gold (if data/gold/ file present)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from utils.stage_io import stage_output_dir, write_csv_artifact
from utils.workspace import canonical_pronoun_annotation_csv, prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)
STAGE_DIR = stage_output_dir("01_annotation_source_pronoun_detection", root=ROOT)

GPT_ANNOTATION_CSV = ROOT / "data" / "Annotated_GPT" / "pronoun_annotation.csv"
# Current canonical GPT annotation = source of stanza TEXT/metadata for --full
# (this is an input to v2, not the v2 output — keep it a literal, not the canonical
# helper, which now resolves to the v2 table).
GPT_RERUN_ANNOTATION_CSV = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
GOLD_CSV = ROOT / "data" / "gold" / "pronoun_gold_25_stanzas.csv"

# Full-corpus v2 outputs land here (the new canonical source-side annotation).
SOURCE_ANNOTATION_DIR = ROOT / "data" / "Annotated_Source"
FULL_TOKENS_CSV = SOURCE_ANNOTATION_DIR / "tokens_v2_full.csv"

IN_SCOPE_LANGUAGES = ("Ukrainian", "Russian")
LANG_TO_STANZA = {"Ukrainian": "uk", "Russian": "ru"}
PRIMARY_CELLS = ("1sg", "1pl", "2sg", "2pl")

# Possessive lemma -> cell. Possessor number is lexical (in the lemma); the DET's
# Number feat agrees with the possessed noun, so feats are deliberately ignored here.
POSSESSIVE_LEMMA_CELL: dict[str, str] = {
    "мій": "1sg", "твій": "2sg",      # Ukrainian
    "мой": "1sg", "твой": "2sg",      # Russian
    "наш": "1pl", "ваш": "2pl",       # shared uk/ru spelling
}

# Personal-pronoun lemma -> cell (number is lexical: я/ты/ти singular, мы/ми/вы/ви
# plural). Used by v2 as a fallback when Stanza omits PronType/Person/Number feats.
PERSONAL_LEMMA_CELL: dict[str, str] = {
    "я": "1sg", "ми": "1pl", "мы": "1pl",
    "ти": "2sg", "ты": "2sg", "ви": "2pl", "вы": "2pl",
}

# Dependency labels that a genuine finite predicate may occupy. Anything nominal
# (nmod/obl/appos/...) carrying a "verb" tag is treated as a tagger false positive.
PREDICATE_DEPRELS = {
    "root", "conj", "ccomp", "xcomp", "advcl", "acl", "parataxis", "csubj",
}

# Plural-imperative surface endings (uk + ru) used to override Stanza's number.
# UA uses -те and -іть/-їть (несіть, ідіть); RU uses -те. The reflexive particle
# -ся/-сь is stripped first so дивуйтеся -> дивуйте still reads as plural.
IMPERATIVE_PLURAL_SUFFIXES = ("те", "іть", "їть")
IMPERATIVE_REFLEXIVE_SUFFIXES = ("ся", "сь")

_CELL_TO_GPT_PERSON = {"1": "1st", "2": "2nd"}
_NUMBER_TO_GPT = {"sg": "Singular", "pl": "Plural"}

# Stanza pipeline cache keyed by (lang_code, with_depparse).
_PIPELINES: dict[tuple[str, bool], object] = {}


def get_pipeline(lang_code: str, *, depparse: bool):
    key = (lang_code, depparse)
    if key not in _PIPELINES:
        import stanza

        procs = "tokenize,pos,lemma,depparse" if depparse else "tokenize,pos,lemma"
        _PIPELINES[key] = stanza.Pipeline(
            lang=lang_code, processors=procs, download_method=None, verbose=False
        )
    return _PIPELINES[key]


def parse_feats(feats_str: str | None) -> dict[str, str]:
    if not feats_str:
        return {}
    out: dict[str, str] = {}
    for item in feats_str.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def person_number_to_cell(person: str | None, number: str | None) -> str | None:
    if person not in ("1", "2"):
        return None
    if number == "Sing":
        return f"{person}sg"
    if number == "Plur":
        return f"{person}pl"
    return None


def is_finite_verb(feats: dict[str, str], upos: str) -> bool:
    """Finite uk/ru verb form (incl. past + imperative); excludes inf/conv/part."""
    if upos != "VERB":
        return False
    if feats.get("VerbForm") in ("Inf", "Conv", "Part"):
        return False
    if "Person" in feats:
        return True
    if feats.get("Tense") == "Past" and "Number" in feats:
        return True
    if feats.get("Mood") == "Imp":
        return True
    return feats.get("VerbForm") == "Fin"


def _gpt_style(cell: str) -> tuple[str, str]:
    if not cell or len(cell) != 3:
        return "", ""
    return _CELL_TO_GPT_PERSON.get(cell[0], ""), _NUMBER_TO_GPT.get(cell[1:], "")


def _past_candidates(number: str | None) -> str:
    if number == "Sing":
        return "1sg|2sg|3sg"
    if number == "Plur":
        return "1pl|2pl|3pl"
    return "ambiguous"


def classify_role(deprel: str) -> str:
    base = (deprel or "").split(":")[0]
    return {
        "nsubj": "subject",
        "obj": "object_direct",
        "iobj": "object_indirect",
        "obl": "oblique",
    }.get(base, base or "unknown")


def fix_imperative_number(surface: str, stanza_number: str | None) -> tuple[str, bool]:
    """Return (corrected Sing/Plur, changed?) from the imperative surface suffix,
    stripping the reflexive particle first (дивуйтеся -> дивуйте)."""
    s = surface.lower().strip().strip(".,!?;:—-–…»«\"'")
    for refl in IMPERATIVE_REFLEXIVE_SUFFIXES:
        if s.endswith(refl):
            s = s[: -len(refl)]
            break
    is_plural = s.endswith(IMPERATIVE_PLURAL_SUFFIXES)
    corrected = "Plur" if is_plural else "Sing"
    return corrected, (corrected != (stanza_number or ""))


@dataclass
class Detection:
    detection_method: str  # explicit_pron | explicit_poss | implied_verb
    surface_form: str
    lemma: str
    cell: str  # "" when excluded
    person: str
    number: str
    is_pro_drop: bool
    governing_verb: str = ""
    verb_tense: str = ""
    verb_deprel: str = ""
    nsubj_lemma: str = ""
    nsubj_deprel: str = ""
    syntactic_role: str = ""
    candidates: str = ""
    promotion_rule: str = ""
    qa_flag: str = "OK"


# --------------------------------------------------------------------------- v1

def detect_stanza_v1(text: str, lang_code: str) -> list[Detection]:
    """Baseline: tokenize/pos/lemma + sentence-level subject-suppression heuristic."""
    if not isinstance(text, str) or not text.strip():
        return []
    nlp = get_pipeline(lang_code, depparse=False)
    doc = nlp(text)
    detections: list[Detection] = []
    for sent in doc.sentences:
        words = sent.words
        explicit_subject_cells: set[str] = set()
        for w in words:
            feats = parse_feats(w.feats)
            if (
                w.upos == "PRON"
                and feats.get("PronType") == "Prs"
                and feats.get("Case") == "Nom"
            ):
                cell = person_number_to_cell(feats.get("Person"), feats.get("Number"))
                if cell:
                    explicit_subject_cells.add(cell)
        for w in words:
            feats = parse_feats(w.feats)
            upos = w.upos
            if upos == "PRON" and feats.get("PronType") == "Prs":
                cell = person_number_to_cell(feats.get("Person"), feats.get("Number"))
                if cell:
                    p, n = _gpt_style(cell)
                    detections.append(Detection("explicit_pron", w.text, w.lemma, cell, p, n, False))
                    continue
            if upos in ("DET", "ADJ") and feats.get("Poss") == "Yes" and feats.get("Person") in ("1", "2"):
                cell = POSSESSIVE_LEMMA_CELL.get(w.lemma.lower())
                if cell is None:
                    pr = feats.get("Person")
                    cell = f"{pr}sg" if pr in ("1", "2") else None
                if cell:
                    p, n = _gpt_style(cell)
                    detections.append(Detection("explicit_poss", w.text, w.lemma, cell, p, n, False))
                    continue
            if is_finite_verb(feats, upos):
                tense = feats.get("Tense", "")
                mood = feats.get("Mood", "")
                person = feats.get("Person")
                if person not in ("1", "2"):
                    if tense == "Past":
                        number = feats.get("Number", "")
                        detections.append(Detection(
                            "implied_verb", w.text, w.lemma, "", "",
                            {"Sing": "Singular", "Plur": "Plural"}.get(number, ""), True,
                            governing_verb=w.text, verb_tense="Past",
                            candidates=_past_candidates(number),
                            qa_flag=f"AMBIGUOUS_PAST(gender={feats.get('Gender', '?')})",
                        ))
                    continue
                cell = person_number_to_cell(person, feats.get("Number"))
                if cell is None:
                    continue
                if cell in explicit_subject_cells:
                    continue
                qa = "IMPERATIVE_NUMBER_UNCERTAIN" if mood == "Imp" else "OK"
                p, n = _gpt_style(cell)
                detections.append(Detection(
                    "implied_verb", w.text, w.lemma, cell, p, n, True,
                    governing_verb=w.text, verb_tense=tense or mood, qa_flag=qa,
                ))
    return detections


# --------------------------------------------------------------------------- v2

def _try_promote_past(verb, feats, id2word, children):
    """Evidence-based promotion of an ambiguous past verb. Returns
    (cell, rule, src_lemma, src_deprel) or None. Only coordinated (conj) evidence,
    and only when NUMBER AGREES between the past verb and the evidence source —
    a mismatch (e.g. neuter-singular impersonal ``було/было`` under a plural ``ми``)
    means they do not share a subject, so we refuse to promote."""
    own_number = feats.get("Number")
    if own_number not in ("Sing", "Plur"):
        return None
    if (verb.deprel or "").split(":")[0] != "conj":
        return None
    head = id2word.get(verb.head)
    if head is None or head.upos != "VERB":
        return None
    hfeats = parse_feats(head.feats)
    # (a) head is a present/future/imperative verb with explicit Person + Number.
    if hfeats.get("Person") in ("1", "2") and hfeats.get("Number") == own_number:
        cell = person_number_to_cell(hfeats["Person"], own_number)
        if cell:
            return cell, "CONJ_PERSON", head.lemma, "conj"
    # (b) head is itself past but carries an explicit 1/2-person pronoun nsubj whose
    #     number agrees with this verb.
    for c in children.get(head.id, []):
        if (c.deprel or "").split(":")[0] == "nsubj" and c.upos == "PRON":
            cfeats = parse_feats(c.feats)
            if (
                cfeats.get("PronType") == "Prs"
                and cfeats.get("Person") in ("1", "2")
                and cfeats.get("Number") == own_number
            ):
                cell = person_number_to_cell(cfeats["Person"], own_number)
                if cell:
                    return cell, "CONJ_NSUBJ", c.lemma, "conj"
    return None


def detect_stanza_v2(text: str, lang_code: str) -> list[Detection]:
    """Highest-precision engine using dependency parses."""
    if not isinstance(text, str) or not text.strip():
        return []
    nlp = get_pipeline(lang_code, depparse=True)
    doc = nlp(text)
    detections: list[Detection] = []
    for sent in doc.sentences:
        words = sent.words
        id2word = {w.id: w for w in words}
        children: dict[int, list] = defaultdict(list)
        for w in words:
            children[w.head].append(w)

        for w in words:
            feats = parse_feats(w.feats)
            upos = w.upos
            deprel = w.deprel or ""
            lemma_l = w.lemma.lower()

            # explicit personal pronoun (counted in ALL syntactic roles). Trigger on
            # PronType=Prs or a known personal lemma (number is lexical in the lemma:
            # я/ты/ти=sg, мы/ми/вы/ви=pl) so feats-poor tokens are still caught.
            if upos == "PRON" and (feats.get("PronType") == "Prs" or lemma_l in PERSONAL_LEMMA_CELL):
                cell = person_number_to_cell(feats.get("Person"), feats.get("Number")) \
                    or PERSONAL_LEMMA_CELL.get(lemma_l)
                if cell:
                    p, n = _gpt_style(cell)
                    detections.append(Detection(
                        "explicit_pron", w.text, w.lemma, cell, p, n, False,
                        syntactic_role=classify_role(deprel), verb_deprel=deprel,
                    ))
                    continue

            # explicit possessive determiner. Cell by lemma; Person feat is OPTIONAL
            # because Russian мой/наш/твой carry no Person feat in Stanza.
            poss_cell = POSSESSIVE_LEMMA_CELL.get(lemma_l)
            if upos in ("DET", "ADJ", "PRON") and (
                poss_cell or (feats.get("Poss") == "Yes" and feats.get("Person") in ("1", "2"))
            ):
                cell = poss_cell
                if cell is None:
                    pr = feats.get("Person")
                    cell = f"{pr}sg" if pr in ("1", "2") else None
                if cell:
                    p, n = _gpt_style(cell)
                    detections.append(Detection(
                        "explicit_poss", w.text, w.lemma, cell, p, n, False,
                        syntactic_role="possessive", verb_deprel=deprel,
                    ))
                    continue

            # finite predicate -> pro-drop path. VERB, plus AUX carrying analytic-future
            # person (буду/будемо ... + infinitive), which Stanza tags upos=AUX deprel=aux.
            is_aux_pred = upos == "AUX" and feats.get("Person") in ("1", "2")
            if is_finite_verb(feats, upos) or is_aux_pred:
                # false-positive filter (full verbs only): a real predicate occupies a
                # predicate deprel; a noun mis-tagged VERB sits in a nominal slot.
                if upos == "VERB" and deprel.split(":")[0] not in PREDICATE_DEPRELS:
                    detections.append(Detection(
                        "implied_verb", w.text, w.lemma, "", "", "", False,
                        governing_verb=w.text, verb_deprel=deprel,
                        qa_flag="FILTERED_NONPREDICATE",
                    ))
                    continue

                tense = feats.get("Tense", "")
                mood = feats.get("Mood", "")
                person = feats.get("Person")

                # nsubj attaches to the verb itself, or (for an AUX) to the verb it supports.
                owner = w if upos == "VERB" else id2word.get(w.head, w)
                nsubj = next(
                    (c for c in (children.get(w.id, []) + children.get(owner.id, []))
                     if (c.deprel or "").split(":")[0] == "nsubj"),
                    None,
                )

                # present / future / imperative with explicit person
                if person in ("1", "2"):
                    number = feats.get("Number")
                    qa = "OK"
                    if mood == "Imp":
                        number, changed = fix_imperative_number(w.text, number)
                        qa = "IMPERATIVE_NUMBER_FIXED" if changed else "IMPERATIVE_OK"
                    cell = person_number_to_cell(person, number)
                    if cell is None:
                        detections.append(Detection(
                            "implied_verb", w.text, w.lemma, "",
                            _CELL_TO_GPT_PERSON.get(person, ""), "", False,
                            governing_verb=w.text, verb_tense=tense or mood,
                            verb_deprel=deprel, qa_flag="NO_VERB_NUMBER",
                        ))
                        continue
                    if nsubj is not None:
                        # overt subject -> NOT pro-drop (pronoun, if any, counted itself)
                        continue
                    p, n = _gpt_style(cell)
                    detections.append(Detection(
                        "implied_verb", w.text, w.lemma, cell, p, n, True,
                        governing_verb=w.text, verb_tense=tense or mood,
                        verb_deprel=deprel, qa_flag=qa,
                    ))
                    continue

                # past tense (no Person) -> exclude unless promotable, never guess
                if tense == "Past":
                    number = feats.get("Number")
                    if nsubj is not None:
                        # subject is overt; not a dropped-subject event.
                        continue
                    promo = _try_promote_past(w, feats, id2word, children)
                    if promo:
                        cell, rule, src_lemma, src_deprel = promo
                        p, n = _gpt_style(cell)
                        detections.append(Detection(
                            "implied_verb", w.text, w.lemma, cell, p, n, True,
                            governing_verb=w.text, verb_tense="Past", verb_deprel=deprel,
                            nsubj_lemma=src_lemma, nsubj_deprel=src_deprel,
                            promotion_rule=rule, qa_flag=f"PAST_PROMOTED_{rule}",
                        ))
                    else:
                        detections.append(Detection(
                            "implied_verb", w.text, w.lemma, "", "",
                            {"Sing": "Singular", "Plur": "Plural"}.get(number, ""), True,
                            governing_verb=w.text, verb_tense="Past", verb_deprel=deprel,
                            candidates=_past_candidates(number),
                            qa_flag=f"AMBIGUOUS_PAST(gender={feats.get('Gender', '?')})",
                        ))
                    continue
                # other finite verb (3rd person etc.) -> out of scope
    return detections


DETECTORS = {"v1": detect_stanza_v1, "v2": detect_stanza_v2}


# ------------------------------------------------------------------- data + counts

def load_stanza_table(input_csv: Path) -> pd.DataFrame:
    usecols = [
        "poem_id", "author", "language", "year", "temporal_period",
        "stanza_index", "stanza_ukr", "person", "number", "is_pro_drop",
    ]
    df = pd.read_csv(input_csv, usecols=usecols, low_memory=False)
    df["language"] = df["language"].astype(str).str.strip()
    df = df[df["language"].isin(IN_SCOPE_LANGUAGES)].copy()
    df["stanza_index"] = pd.to_numeric(df["stanza_index"], errors="coerce")
    df = df.dropna(subset=["stanza_index"])
    df["stanza_index"] = df["stanza_index"].astype(int)
    return df


def unique_stanzas(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["poem_id", "stanza_index", "language", "author", "year", "temporal_period", "stanza_ukr"]
    uniq = df[cols].drop_duplicates(subset=["poem_id", "stanza_index"]).reset_index(drop=True)
    return uniq.rename(columns={"stanza_ukr": "text"})


def stratified_sample(uniq: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per_lang = max(1, n // len(IN_SCOPE_LANGUAGES))
    parts = []
    for lang in IN_SCOPE_LANGUAGES:
        sub = uniq[uniq["language"] == lang]
        parts.append(sub.sample(min(per_lang, len(sub)), random_state=seed))
    out = pd.concat(parts).reset_index(drop=True)
    return out.head(n) if len(out) > n else out


@dataclass
class StanzaJob:
    poem_id: str
    stanza_index: int
    language: str
    author: str
    year: object
    temporal_period: object
    text: str


def _detection_to_row(job: "StanzaJob", d: Detection, mode: str) -> dict:
    """One token-level output row (shared by sample + full-corpus paths)."""
    return {
        "poem_id": job.poem_id, "author": job.author, "language": job.language,
        "year": job.year, "temporal_period": job.temporal_period,
        "stanza_index": job.stanza_index, "mode": mode,
        "detection_method": d.detection_method, "surface_form": d.surface_form,
        "lemma": d.lemma, "cell": d.cell, "person": d.person, "number": d.number,
        "is_pro_drop": d.is_pro_drop, "governing_verb": d.governing_verb,
        "verb_tense": d.verb_tense, "verb_deprel": d.verb_deprel,
        "nsubj_lemma": d.nsubj_lemma, "nsubj_deprel": d.nsubj_deprel,
        "syntactic_role": d.syntactic_role, "candidates": d.candidates,
        "promotion_rule": d.promotion_rule, "qa_flag": d.qa_flag,
    }


def detect_jobs(jobs: list[StanzaJob], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (token rows, per-stanza count rows) for one engine mode."""
    detect = DETECTORS[mode]
    token_rows: list[dict] = []
    count_rows: list[dict] = []
    for job in jobs:
        dets = detect(job.text, LANG_TO_STANZA[job.language])
        cell_count = {c: 0 for c in PRIMARY_CELLS}
        expl_count = {c: 0 for c in PRIMARY_CELLS}
        impl_count = {c: 0 for c in PRIMARY_CELLS}
        for d in dets:
            token_rows.append(_detection_to_row(job, d, mode))
            if d.cell in cell_count:
                cell_count[d.cell] += 1
                if d.is_pro_drop:
                    impl_count[d.cell] += 1
                else:
                    expl_count[d.cell] += 1
        rec = {"poem_id": job.poem_id, "stanza_index": job.stanza_index, "language": job.language}
        for c in PRIMARY_CELLS:
            rec[f"{mode}_{c}"] = cell_count[c]
            rec[f"{mode}_expl_{c}"] = expl_count[c]
            rec[f"{mode}_impl_{c}"] = impl_count[c]
        rec[f"{mode}_ambiguous_past"] = sum(1 for d in dets if d.qa_flag.startswith("AMBIGUOUS_PAST"))
        rec[f"{mode}_promoted_past"] = sum(1 for d in dets if d.promotion_rule)
        count_rows.append(rec)
    return pd.DataFrame(token_rows), pd.DataFrame(count_rows)


def run_full(jobs: list[StanzaJob], mode: str, out_csv: Path,
             *, checkpoint_every: int = 300) -> Path:
    """Full-corpus detection with resumable checkpointing.

    Appends token rows to ``out_csv`` every ``checkpoint_every`` stanzas; on restart,
    stanzas already present in ``out_csv`` are skipped. Stanzas with zero detections
    leave no row (and are harmlessly reprocessed on resume — detection is pure)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, usecols=["poem_id", "stanza_index"], low_memory=False)
        done = set(zip(prev["poem_id"].astype(str), prev["stanza_index"].astype(int)))
        print(f"[full] resume: {len(done)} stanzas already in {out_csv.name}")
    pending = [j for j in jobs if (j.poem_id, j.stanza_index) not in done]
    print(f"[full] processing {len(pending)} of {len(jobs)} stanzas (mode={mode})")
    detect = DETECTORS[mode]
    header_written = out_csv.exists()
    buf: list[dict] = []

    def flush():
        nonlocal buf, header_written
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not header_written,
                                 index=False, encoding="utf-8")
        header_written = True
        buf = []

    try:
        from tqdm import tqdm
        it = tqdm(pending, desc=f"v2 full ({mode})", unit="stanza")
    except Exception:
        it = pending
    for i, job in enumerate(pending, 1):
        for d in detect(job.text, LANG_TO_STANZA[job.language]):
            buf.append(_detection_to_row(job, d, mode))
        if i % checkpoint_every == 0:
            flush()
    flush()
    n = pd.read_csv(out_csv, usecols=["poem_id"], low_memory=False).shape[0] if out_csv.exists() else 0
    print(f"[full] done: {n} token rows in {out_csv}")
    return out_csv


def gpt_counts(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    from utils.pronoun_encoding import pronoun_class_sixway_column

    merged = df.merge(keys[["poem_id", "stanza_index"]], on=["poem_id", "stanza_index"]).copy()
    merged["cell"] = pronoun_class_sixway_column(merged)
    merged = merged[merged["cell"].isin(PRIMARY_CELLS)]
    drop = merged["is_pro_drop"].astype(str).str.lower().isin(["true", "1", "1.0"])
    merged["impl"] = drop
    rows = []
    for (pid, si), grp in merged.groupby(["poem_id", "stanza_index"]):
        rec = {"poem_id": pid, "stanza_index": si}
        for c in PRIMARY_CELLS:
            m = grp["cell"] == c
            rec[f"gpt_{c}"] = int(m.sum())
            rec[f"gpt_expl_{c}"] = int((m & ~grp["impl"]).sum())
            rec[f"gpt_impl_{c}"] = int((m & grp["impl"]).sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        cols = ["poem_id", "stanza_index"] + [f"gpt_{p}{c}" for p in ("", "expl_", "impl_") for c in PRIMARY_CELLS]
        out = pd.DataFrame(columns=cols)
    return out


def build_pair_diff(left: pd.DataFrame, right: pd.DataFrame, lname: str, rname: str) -> pd.DataFrame:
    keys = ["poem_id", "stanza_index"]
    base = left[keys + (["language"] if "language" in left.columns else [])]
    diff = base.merge(left, on=base.columns.tolist(), how="left").merge(right, on=keys, how="outer")
    for c in PRIMARY_CELLS:
        lc, rc = f"{lname}_{c}", f"{rname}_{c}"
        for col in (lc, rc):
            if col not in diff.columns:
                diff[col] = 0
            diff[col] = diff[col].fillna(0).astype(int)
        diff[f"d_{c}"] = diff[rc] - diff[lc]
    ordered = keys + (["language"] if "language" in diff.columns else [])
    for c in PRIMARY_CELLS:
        ordered += [f"{lname}_{c}", f"{rname}_{c}", f"d_{c}"]
    ordered += [x for x in diff.columns if x not in ordered]
    return diff[ordered]


# ------------------------------------------------------------------- gold metrics

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def gold_metrics(gold: pd.DataFrame, count_tables: dict[str, pd.DataFrame],
                 gpt_table: pd.DataFrame) -> pd.DataFrame:
    """Micro count-based P/R/F1 of each mode vs the hand gold, by cell/role/language."""
    keys = ["poem_id", "stanza_index"]
    gold = gold.copy()
    gold["poem_id"] = gold["poem_id"].astype(str)
    g = gold.set_index(keys)

    # Build per-mode predicted lookups: mode -> {(pid,si): {cell:{'expl':n,'impl':n}}}
    preds: dict[str, pd.DataFrame] = {}
    for mode, tab in count_tables.items():
        t = tab.copy()
        t["poem_id"] = t["poem_id"].astype(str)
        preds[mode] = t.set_index(keys)
    gpt = gpt_table.copy()
    if not gpt.empty:
        gpt["poem_id"] = gpt["poem_id"].astype(str)
        gpt = gpt.set_index(keys)
    preds["gpt"] = gpt

    rows = []

    def metric(mode: str, stratum: str, langfilter: str | None):
        tp = pred_tot = gold_tot = 0
        for idx, grow in g.iterrows():
            lang = grow.get("language", "")
            if langfilter and lang != langfilter:
                continue
            ptab = preds.get(mode)
            prow = ptab.loc[idx] if (ptab is not None and idx in ptab.index) else None
            for c in PRIMARY_CELLS:
                for kind in (("expl", "impl") if stratum == "total" else (stratum,)):
                    gval = int(grow.get(f"gold_{kind}_{c}", 0) or 0)
                    if prow is None:
                        pval = 0
                    else:
                        col = f"{'gpt' if mode == 'gpt' else mode}_{kind}_{c}"
                        pval = int(prow.get(col, 0) or 0)
                    tp += min(pval, gval)
                    pred_tot += pval
                    gold_tot += gval
        prec = _safe_div(tp, pred_tot)
        rec = _safe_div(tp, gold_tot)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        rows.append({
            "mode": mode, "stratum": stratum, "language": langfilter or "all",
            "tp": tp, "pred": pred_tot, "gold": gold_tot,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        })

    modes = list(count_tables.keys()) + (["gpt"] if not gpt_table.empty else [])
    for mode in modes:
        for stratum in ("total", "expl", "impl"):
            metric(mode, stratum, None)
            for lang in IN_SCOPE_LANGUAGES:
                metric(mode, stratum, lang)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- audits

def ambiguous_past_audit(tokens_v2: pd.DataFrame) -> pd.DataFrame:
    if tokens_v2.empty:
        return tokens_v2
    m = tokens_v2["qa_flag"].str.startswith("AMBIGUOUS_PAST") | (tokens_v2["promotion_rule"] != "")
    cols = ["poem_id", "stanza_index", "language", "surface_form", "lemma", "verb_deprel",
            "candidates", "promotion_rule", "nsubj_lemma", "cell", "qa_flag"]
    return tokens_v2.loc[m, cols].reset_index(drop=True)


def error_analysis(tokens_v2: pd.DataFrame) -> pd.DataFrame:
    if tokens_v2.empty:
        return tokens_v2
    flags = ("FILTERED_NONPREDICATE", "IMPERATIVE_NUMBER_FIXED", "NO_VERB_NUMBER")
    m = tokens_v2["qa_flag"].isin(flags)
    cols = ["poem_id", "stanza_index", "language", "surface_form", "lemma",
            "verb_deprel", "cell", "number", "qa_flag"]
    return tokens_v2.loc[m, cols].reset_index(drop=True)


# --------------------------------------------------------------------------- main

def main() -> None:
    """Prototype CLI. Full-corpus scaling: v2 (+depparse) ~2x v1; checkpoint per
    poem_id to parquet shards, one Stanza pipeline per worker process, optional GPU."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--language", choices=[*IN_SCOPE_LANGUAGES, "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", type=Path, default=GPT_ANNOTATION_CSV)
    ap.add_argument("--mode", choices=["v1", "v2"], default="v2",
                    help="single-engine output mode (ignored if --compare-modes given)")
    ap.add_argument("--compare-modes", type=str, default="",
                    help="comma list from {v1,v2,gpt}, e.g. v1,v2,gpt")
    ap.add_argument("--gold", type=Path, default=GOLD_CSV)
    ap.add_argument("--dump-candidates", type=Path, default=None,
                    help="write the sampled stanzas' text to this CSV and exit (gold prep)")
    ap.add_argument("--full", action="store_true",
                    help="process ALL in-scope stanzas with checkpoint/resume -> --out")
    ap.add_argument("--out", type=Path, default=FULL_TOKENS_CSV,
                    help="full-corpus token CSV output (default: data/Annotated_Source/tokens_v2_full.csv)")
    ap.add_argument("--checkpoint-every", type=int, default=300)
    args = ap.parse_args()

    modes = [m.strip() for m in args.compare_modes.split(",") if m.strip()] or [args.mode]
    engine_modes = [m for m in modes if m in DETECTORS]
    want_gpt = "gpt" in modes

    print(f"[INFO] Loading stanza table from {args.input}")
    df = load_stanza_table(args.input)
    uniq = unique_stanzas(df)
    if args.language != "both":
        uniq = uniq[uniq["language"] == args.language]
    print(f"[INFO] {len(uniq)} unique in-scope stanzas available")

    if args.full:
        all_jobs = [
            StanzaJob(str(r.poem_id), int(r.stanza_index), r.language, r.author, r.year,
                      r.temporal_period, r.text)
            for r in uniq.itertuples(index=False)
        ]
        run_mode = args.mode if not engine_modes else engine_modes[-1]
        run_full(all_jobs, run_mode, args.out, checkpoint_every=args.checkpoint_every)
        return

    sample = uniq.reset_index(drop=True) if args.full else stratified_sample(uniq, args.sample, args.seed)
    print(f"[INFO] {len(sample)} stanzas sampled ({sample['language'].value_counts().to_dict()})")

    if args.dump_candidates is not None:
        write_csv_artifact(sample, args.dump_candidates)
        print(f"[INFO] wrote candidate texts -> {args.dump_candidates}")
        return

    jobs = [
        StanzaJob(str(r.poem_id), int(r.stanza_index), r.language, r.author, r.year,
                  r.temporal_period, r.text)
        for r in sample.itertuples(index=False)
    ]

    count_tables: dict[str, pd.DataFrame] = {}
    token_tables: dict[str, pd.DataFrame] = {}
    for mode in engine_modes:
        print(f"[INFO] running engine '{mode}' ...")
        tokens, counts = detect_jobs(jobs, mode)
        token_tables[mode] = tokens
        count_tables[mode] = counts
        for lang in IN_SCOPE_LANGUAGES:
            sub = tokens[tokens["language"] == lang] if not tokens.empty else tokens
            if not sub.empty:
                path = STAGE_DIR / f"tokens_{mode}_{LANG_TO_STANZA[lang]}.csv"
                write_csv_artifact(sub, path)
                print(f"[INFO]   wrote {len(sub)} token rows -> {path.name}")
        write_csv_artifact(counts, STAGE_DIR / f"stanza_cell_counts_{mode}.csv")

    gpt_table = gpt_counts(df, sample[["poem_id", "stanza_index"]]) if (want_gpt or args.gold.exists()) else pd.DataFrame()

    # diffs
    if "v1" in count_tables and "v2" in count_tables:
        d = build_pair_diff(count_tables["v1"], count_tables["v2"], "v1", "v2")
        write_csv_artifact(d, STAGE_DIR / "diff_v1_vs_v2.csv")
        print("[INFO] wrote diff_v1_vs_v2.csv")
    if "v2" in count_tables and not gpt_table.empty:
        d = build_pair_diff(count_tables["v2"], gpt_table, "v2", "gpt")
        write_csv_artifact(d, STAGE_DIR / "diff_v2_vs_gpt.csv")
        print("[INFO] wrote diff_v2_vs_gpt.csv")

    # audits (v2)
    if "v2" in token_tables:
        write_csv_artifact(ambiguous_past_audit(token_tables["v2"]), STAGE_DIR / "ambiguous_past_audit.csv")
        write_csv_artifact(error_analysis(token_tables["v2"]), STAGE_DIR / "error_analysis.csv")
        print("[INFO] wrote ambiguous_past_audit.csv + error_analysis.csv")

    # gold evaluation — always scored on the GOLD stanzas themselves (independent of
    # --sample), running BOTH engines + GPT so the v1/v2/gpt comparison is complete.
    if args.gold.exists():
        gold = pd.read_csv(args.gold)
        gold["poem_id"] = gold["poem_id"].astype(str)
        gkeys = gold[["poem_id", "stanza_index"]].drop_duplicates()
        guniq = uniq.copy()
        guniq["poem_id"] = guniq["poem_id"].astype(str)
        gjoin = gkeys.merge(guniq, on=["poem_id", "stanza_index"], how="left")
        missing = gjoin["text"].isna().sum()
        if missing:
            print(f"[WARN] {missing} gold stanzas not found in corpus; scored as 0 prediction")
        gold_jobs = [
            StanzaJob(str(r.poem_id), int(r.stanza_index), r.language, r.author,
                      r.year, r.temporal_period, r.text)
            for r in gjoin.dropna(subset=["text"]).itertuples(index=False)
        ]
        gold_count_tables = {m: detect_jobs(gold_jobs, m)[1] for m in DETECTORS}
        gtab = gpt_counts(df, gkeys)
        metrics = gold_metrics(gold, gold_count_tables, gtab)
        write_csv_artifact(metrics, STAGE_DIR / "eval_gold_metrics.csv")
        print(f"\n=== Gold P/R/F1 ({len(gold)} stanzas) — stratum=total, language=all ===")
        show = metrics[(metrics.stratum == "total") & (metrics.language == "all")]
        print(show[["mode", "precision", "recall", "f1", "tp", "pred", "gold"]].to_string(index=False))
        print("\n=== by explicit/implied (all languages) ===")
        show2 = metrics[(metrics.stratum != "total") & (metrics.language == "all")]
        print(show2[["mode", "stratum", "precision", "recall", "f1"]].to_string(index=False))
    else:
        print(f"[WARN] gold file not found at {args.gold}; skipping P/R/F1")

    # quick console summary
    if "v2" in count_tables:
        v2 = count_tables["v2"]
        print(f"\n[v2] promoted-past events={int(v2.get('v2_promoted_past', pd.Series(dtype=int)).sum())}; "
              f"ambiguous-past excluded={int(v2.get('v2_ambiguous_past', pd.Series(dtype=int)).sum())}")


if __name__ == "__main__":
    main()
