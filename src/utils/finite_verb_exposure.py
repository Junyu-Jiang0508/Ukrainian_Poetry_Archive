"""Finite-verb exposure helpers for poem-level offset construction.

Stanza parsing is expensive; callers should run ``00e_compute_finite_verb_exposure.py``
once and join ``stanza_finite_verb_counts.csv`` via ``load_finite_verb_counts``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_STANZA_PIPELINE: Any | None = None
_STANZA_INIT_FAILED = False


def _parse_feats(feats_str: str | None) -> dict[str, str]:
    """Parse Stanza feature string into key/value pairs."""
    if not feats_str:
        return {}
    out: dict[str, str] = {}
    for item in feats_str.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def is_finite_verb(word: Any, *, exclude_imperative: bool = False) -> bool:
    """Return True when a Stanza word is a finite Ukrainian verb form."""
    if getattr(word, "upos", None) != "VERB":
        return False
    feats = _parse_feats(getattr(word, "feats", None))
    if exclude_imperative and feats.get("Mood") == "Imp":
        return False
    verb_form = feats.get("VerbForm", "")

    if verb_form in ("Inf", "Conv", "Part"):
        return False
    if "Person" in feats:
        return True
    if feats.get("Tense") == "Past" and "Number" in feats:
        return True
    if feats.get("Mood") == "Imp":
        return True
    if verb_form == "Fin":
        return True
    return False


def _get_stanza_pipeline(*, raise_on_failure: bool = False) -> Any:
    """Initialize Stanza pipeline once; raise or return None on failure."""
    global _STANZA_PIPELINE, _STANZA_INIT_FAILED
    if _STANZA_PIPELINE is not None:
        return _STANZA_PIPELINE
    if _STANZA_INIT_FAILED and not raise_on_failure:
        return None
    try:
        import stanza  # imported lazily; optional dependency outside annotation path

        _STANZA_PIPELINE = stanza.Pipeline(
            lang="uk",
            processors="tokenize,pos,lemma",
            download_method=None,
            verbose=False,
        )
        return _STANZA_PIPELINE
    except Exception as exc:
        _STANZA_INIT_FAILED = True
        msg = f"Could not initialize Stanza for finite-verb offset: {exc}"
        if raise_on_failure:
            raise RuntimeError(msg) from exc
        log.warning(msg)
        return None


def count_finite_verbs_in_stanza(
    text: object,
    nlp_stanza: Any,
    *,
    exclude_imperative: bool = False,
) -> int:
    """Count finite verbs in one stanza text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    try:
        doc = nlp_stanza(s)
        return int(
            sum(
                1
                for sent in doc.sentences
                for word in sent.words
                if is_finite_verb(word, exclude_imperative=exclude_imperative)
            )
        )
    except Exception as exc:  # pragma: no cover - parser failure branch
        log.warning("Stanza parse failed in finite-verb counting: %s", exc)
        return 0


def stanza_text_hash(text: object) -> str:
    """Stable short hash of stanza text for cache lineage."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    raw = str(text).strip().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def compute_finite_verb_counts_table(
    df_stanzas: pd.DataFrame,
    nlp_stanza: Any,
    *,
    text_col: str = "stanza_text",
) -> pd.DataFrame:
    """Return one row per (poem_id, stanza_index) with FV counts and token counts.

    Expects columns ``poem_id``, ``stanza_index``, and ``text_col``.
    """
    need = {"poem_id", "stanza_index", text_col}
    miss = need - set(df_stanzas.columns)
    if miss:
        raise ValueError(f"compute_finite_verb_counts_table: missing columns {sorted(miss)}")

    rows: list[dict[str, object]] = []
    try:
        from tqdm import tqdm  # type: ignore

        row_iter = tqdm(df_stanzas.iterrows(), total=len(df_stanzas), desc="Stanza FV counts", unit="stanza")
    except Exception:
        row_iter = df_stanzas.iterrows()
    for _, r in row_iter:
        pid = str(r["poem_id"]).strip()
        si = pd.to_numeric(r["stanza_index"], errors="coerce")
        if pd.isna(si):
            continue
        stanza_idx = int(si)
        txt = r[text_col]
        s = str(txt).strip() if txt is not None and not (isinstance(txt, float) and pd.isna(txt)) else ""
        n_tok = len(s.split()) if s else 0
        n_fv = count_finite_verbs_in_stanza(s, nlp_stanza, exclude_imperative=False)
        n_fv_ex = count_finite_verbs_in_stanza(s, nlp_stanza, exclude_imperative=True)
        rows.append(
            {
                "poem_id": pid,
                "stanza_index": stanza_idx,
                "n_tokens_stanza": n_tok,
                "n_finite_verbs": n_fv,
                "n_finite_verbs_excl_imperative": n_fv_ex,
                "stanza_hash": stanza_text_hash(txt),
            }
        )
    return pd.DataFrame(rows)


def init_stanza_finite_verb_pipeline() -> Any:
    """Initialize the morph-only Stanza pipeline or raise ``RuntimeError``."""
    return _get_stanza_pipeline(raise_on_failure=True)


def load_finite_verb_counts(path: Path | str) -> pd.DataFrame | None:
    """Load precomputed stanza-level FV counts; return None if missing or empty."""
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p, low_memory=False)
    if df.empty:
        return None
    for c in ("poem_id", "stanza_index", "n_finite_verbs", "n_finite_verbs_excl_imperative"):
        if c not in df.columns:
            raise ValueError(f"load_finite_verb_counts: {p} missing required column {c!r}")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["stanza_index"] = pd.to_numeric(df["stanza_index"], errors="coerce")
    return df


def require_finite_verb_counts(path: Path | str) -> pd.DataFrame:
    """Load FV counts or raise SystemExit with remediation hint."""
    df = load_finite_verb_counts(path)
    if df is None:
        raise SystemExit(
            f"Finite-verb exposure file missing or empty: {path}\n"
            "Run: PYTHONPATH=src python src/00e_compute_finite_verb_exposure.py"
        )
    return df


def default_finite_verb_counts_path(root: Path) -> Path:
    return root / "data" / "To_run" / "00_filtering" / "stanza_finite_verb_counts.csv"


def resolve_finite_verb_counts_for_modeling(
    root: Path,
    *,
    exposure_type: str,
    finite_verb_csv: Path | None = None,
) -> pd.DataFrame | None:
    """Require non-empty FV table for FV exposure modes; otherwise load if present."""
    path = Path(finite_verb_csv) if finite_verb_csv is not None else default_finite_verb_counts_path(root)
    if exposure_type in ("n_finite_verbs", "n_finite_verbs_excl_imperative"):
        return require_finite_verb_counts(path)
    return load_finite_verb_counts(path)
