"""Finite-verb exposure helpers for poem-level offset construction."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

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


def is_finite_verb(word: Any) -> bool:
    """Return True when a Stanza word is a finite Ukrainian verb form."""
    if getattr(word, "upos", None) != "VERB":
        return False
    feats = _parse_feats(getattr(word, "feats", None))
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


def _get_stanza_pipeline() -> Any | None:
    """Initialize Stanza pipeline once; return None if unavailable."""
    global _STANZA_PIPELINE, _STANZA_INIT_FAILED
    if _STANZA_PIPELINE is not None:
        return _STANZA_PIPELINE
    if _STANZA_INIT_FAILED:
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
    except Exception as exc:  # pragma: no cover - environment-dependent
        _STANZA_INIT_FAILED = True
        log.warning("Could not initialize Stanza for finite-verb offset: %s", exc)
        return None


@lru_cache(maxsize=100_000)
def _count_finite_verbs_cached(text: str) -> int:
    nlp_stanza = _get_stanza_pipeline()
    if nlp_stanza is None:
        return 0
    try:
        doc = nlp_stanza(text)
        return int(sum(1 for sent in doc.sentences for word in sent.words if is_finite_verb(word)))
    except Exception as exc:  # pragma: no cover - parser failure branch
        log.warning("Stanza parse failed in finite-verb counting: %s", exc)
        return 0


def count_finite_verbs_in_text(text: object) -> int:
    """Count finite verbs in one stanza text (returns 0 on empty/parse-failure)."""
    if text is None:
        return 0
    s = str(text).strip()
    if not s:
        return 0
    return _count_finite_verbs_cached(s)
