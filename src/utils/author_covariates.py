"""Time-anchored author-level covariate loader (P2-3, post-refactor).

Author-level predictors (gender, generation, birth-region, language history,
bilingual-switcher) are not derivable from the poem-level corpus. They are
supplied as a manually curated CSV at :data:`DEFAULT_COVARIATE_PATH`. This
module loads and validates the CSV against the schema in :data:`SCHEMA` and
merges it onto poem-level tables by ``author`` (left join).

Time-anchoring discipline (post-refactor)
-----------------------------------------
The corpus we model spans 2014--2025. Several biographical attributes are
*not* time-invariant across that window: a poet who lived in Kyiv in 2020
and moved to Berlin in 2023 has different "location" states for the
period-1 (2014--2021) and period-2 (post-2022) poems. Using
``Current location`` as a single static covariate in a within-author
period contrast risks **post-treatment bias**: emigration *is itself a
consequence* of the war, so adjusting on it bleeds the period effect out
of the regression.

We therefore split every potentially time-varying field into columns with
an *explicit time anchor*:

* Stable across the study window (safe to use as fixed effects or as
  group-level predictors in random-slope models):
  ``gender``, ``birth_year``, ``generation_cohort``, ``region_of_birth``.

* Single-time-point snapshots (use with caution; never as static
  "control" variables in within-author period models):
  ``region_jan2022``, ``region_at_archive_freeze``,
  ``language_xlsx_primary_at_freeze``.

* Empirically derived from the corpus *by period* (best controls for
  this study because the time anchor matches the inferential unit):
  ``language_corpus_p1``, ``language_corpus_p2``,
  ``bilingual_switcher_corpus``.

Recommended usage in 02c group-level predictors::

    period_slope_i ~ gender + generation_cohort + region_of_birth +
                     language_corpus_p1

Do **not** include ``region_at_archive_freeze`` as a "control" — it is a
descendant of the period effect, not an antecedent.

Schema columns (canonical order)
--------------------------------
* ``author`` (string, primary key)
* ``gender``  ∈  {``F``, ``M``, ``NB``}  (blank if unknown)
* ``birth_year`` (4-digit string year, e.g. ``"1975"``; blank if unknown)
* ``generation_cohort``  ∈
  {``pre_1970``, ``1970s``, ``1980s``, ``1990s``, ``2000s_plus``}  (blank if unknown)
* ``region_of_birth``  ∈  {``east_ukraine``, ``west_ukraine``,
  ``central_ukraine``, ``south_ukraine``, ``kyiv``, ``crimea``,
  ``born_abroad``, ``other``}  (blank if unknown)
* ``region_jan2022``  (same vocabulary as ``region_of_birth`` plus
  ``diaspora``; documented as a pre-invasion snapshot, only ~21/281
  authors have this from the xlsx)
* ``region_at_archive_freeze``  (same vocabulary; **post-2022 snapshot**,
  do not use as a static control)
* ``language_xlsx_primary_at_freeze``  ∈  {``Ukrainian``, ``Russian``,
  ``bilingual``, ``other``}  (blank if unknown; xlsx Primary language)
* ``language_corpus_p1``  (empirical dominant language in P1: 2014--2021;
  same vocabulary; blank if unknown)
* ``language_corpus_p2``  (empirical dominant language in P2: post-2022;
  blank if unknown)
* ``bilingual_switcher_corpus``  ∈  {``yes``, ``no``}  (blank if unknown)
  (empirical, derived from the corpus by the roster freeze stage)
* ``notes`` (free text, optional)
* ``references`` (pipe-separated source URLs; optional; not a model predictor)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.workspace import repository_root

log = logging.getLogger(__name__)


DEFAULT_COVARIATE_PATH = repository_root() / "data" / "author_covariates.csv"


@dataclass(frozen=True)
class CovariateColumn:
    name: str
    allowed: frozenset[str] | None  # None == free text / numeric-like
    default_unknown: str = ""  # blank cell = missing / not yet researched
    time_anchor: str = "time_invariant"  # documentation-only field


# Region vocabulary shared by the three time-anchored region columns. We add
# ``born_abroad`` for region_of_birth (e.g. born in Russia, Tajikistan, Poland)
# and ``diaspora`` for the two snapshot columns (post-emigration). Both are
# documented as ``other``'s siblings. Blank = missing (not a coded category).
_REGION_VOCAB = frozenset(
    {
        "east_ukraine",
        "west_ukraine",
        "central_ukraine",
        "south_ukraine",
        "kyiv",
        "crimea",
        "born_abroad",
        "diaspora",
        "other",
    }
)

_LANGUAGE_VOCAB = frozenset(
    {"Ukrainian", "Russian", "bilingual", "other"}
)


SCHEMA: tuple[CovariateColumn, ...] = (
    CovariateColumn("author", None, time_anchor="key"),
    CovariateColumn(
        "gender",
        frozenset({"F", "M", "NB"}),
        time_anchor="time_invariant",
    ),
    # birth_year is stored as a string so blanks coerce predictably; downstream
    # consumers can ``pd.to_numeric(..., errors="coerce")``.
    CovariateColumn("birth_year", None, time_anchor="time_invariant"),
    CovariateColumn(
        "generation_cohort",
        frozenset({"pre_1970", "1970s", "1980s", "1990s", "2000s_plus"}),
        time_anchor="time_invariant",
    ),
    CovariateColumn("region_of_birth", _REGION_VOCAB, time_anchor="time_invariant"),
    CovariateColumn(
        "region_jan2022",
        _REGION_VOCAB,
        time_anchor="snapshot_pre_invasion_2022_01",
    ),
    CovariateColumn(
        "region_at_archive_freeze",
        _REGION_VOCAB,
        time_anchor="snapshot_post_2022_archive_freeze",
    ),
    CovariateColumn(
        "language_xlsx_primary_at_freeze",
        _LANGUAGE_VOCAB,
        time_anchor="snapshot_archive_freeze_year_not_documented",
    ),
    CovariateColumn(
        "language_corpus_p1",
        _LANGUAGE_VOCAB,
        time_anchor="empirical_period_1_2014_2021",
    ),
    CovariateColumn(
        "language_corpus_p2",
        _LANGUAGE_VOCAB,
        time_anchor="empirical_period_2_post_2022",
    ),
    CovariateColumn(
        "bilingual_switcher_corpus",
        frozenset({"yes", "no"}),
        time_anchor="empirical_period_1_vs_period_2_contrast",
    ),
    CovariateColumn("notes", None, time_anchor="free_text"),
    CovariateColumn("references", None, time_anchor="free_text"),
)


# Convenience subsets for downstream modelling.
TIME_INVARIANT_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in SCHEMA if c.time_anchor == "time_invariant"
)
SNAPSHOT_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in SCHEMA if c.time_anchor.startswith("snapshot")
)
EMPIRICAL_BY_PERIOD_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in SCHEMA if c.time_anchor.startswith("empirical")
)
SAFE_FOR_PERIOD_SLOPE_PREDICTORS: tuple[str, ...] = (
    "gender",
    "generation_cohort",
    "region_of_birth",
    "language_corpus_p1",
)


def schema_columns() -> list[str]:
    return [c.name for c in SCHEMA]


def is_covariate_missing(series: pd.Series) -> pd.Series:
    """True where a covariate cell is blank or legacy ``unknown``."""
    s = series.astype(str).str.strip()
    return s.eq("") | s.str.casefold().eq("unknown")


def _validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce columns to canonical strings; record any out-of-vocabulary values."""
    warnings_list: list[str] = []
    out = df.copy()
    for col in SCHEMA:
        if col.name not in out.columns:
            if col.name in ("notes", "references"):
                out[col.name] = ""
            elif col.name == "birth_year":
                out[col.name] = ""
            else:
                out[col.name] = col.default_unknown
            warnings_list.append(f"column {col.name!r} absent; defaulted to blank")
            continue
        out[col.name] = out[col.name].fillna("").astype(str).str.strip()
        # Legacy CSV rows may still contain the literal token ``unknown``.
        out.loc[out[col.name].str.casefold().eq("unknown"), col.name] = ""
        if col.allowed is None:
            continue
        filled = out[col.name].ne("")
        out_of_set = filled & ~out[col.name].isin(col.allowed)
        if out_of_set.any():
            bad = sorted(out.loc[out_of_set, col.name].unique().tolist())
            warnings_list.append(
                f"column {col.name!r} has values outside the allowed set "
                f"{sorted(col.allowed)!r}: {bad!r}; coerced to blank"
            )
            out.loc[out_of_set, col.name] = ""
    return out, warnings_list


def load_author_covariates(
    path: Path | None = None,
    *,
    warn_on_missing: bool = True,
) -> pd.DataFrame:
    """Load and validate the covariates CSV. Returns an empty schema if absent."""
    p = (path or DEFAULT_COVARIATE_PATH).resolve()
    if not p.is_file():
        if warn_on_missing:
            log.warning(
                "Author covariates file %s not found. Downstream models will "
                "fall back to blank for every field. See utils.author_covariates.SCHEMA "
                "for the schema and time-anchor documentation.",
                p,
            )
        return pd.DataFrame(columns=schema_columns())
    raw = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False, comment="#")
    if "author" in raw.columns:
        raw = raw.loc[raw["author"].astype(str).str.strip().ne("")].copy()
    coerced, warnings_list = _validate(raw)
    for w in warnings_list:
        log.warning("author_covariates schema: %s", w)
    return coerced


def merge_onto_poem_table(
    poem_df: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join the covariates onto a poem-level DataFrame keyed on ``author``."""
    if covariates is None:
        covariates = load_author_covariates()
    if covariates.empty:
        for col in SCHEMA:
            if col.name == "author":
                continue
            if col.name not in poem_df.columns:
                default = (
                    ""
                    if col.name in ("notes", "references", "birth_year")
                    else col.default_unknown
                )
                poem_df = poem_df.assign(**{col.name: default})
        return poem_df
    out = poem_df.merge(covariates, on="author", how="left")
    for col in SCHEMA:
        if col.name in ("author", "notes", "references", "birth_year"):
            continue
        out[col.name] = out[col.name].fillna("")
        out.loc[out[col.name].astype(str).str.casefold().eq("unknown"), col.name] = ""
    return out


def emit_schema_template(path: Path | None = None) -> Path:
    """Write a header-only template CSV at ``path`` (preserves existing file)."""
    p = (path or DEFAULT_COVARIATE_PATH).resolve()
    if p.is_file():
        log.info("Covariates file already exists at %s — not overwriting", p)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    header = pd.DataFrame(columns=schema_columns())
    header.to_csv(p, index=False, encoding="utf-8-sig")
    log.info("Wrote empty covariates template to %s", p)
    return p
