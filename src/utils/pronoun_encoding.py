"""Shared person × number encodings for UD-style GPT annotation rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Poem-level count columns (order preserved for CSVs) ---
CELL_2PL_LEGACY = "2pl"
CELL_2PL_VY_POLITE_SINGULAR = "2pl_vy_polite_singular"
CELL_2PL_VY_TRUE_PLURAL = "2pl_vy_true_plural"

POEM_COUNT_CELL_COLUMNS: tuple[str, ...] = (
    "1sg",
    "1pl",
    "2sg",
    CELL_2PL_VY_POLITE_SINGULAR,
    CELL_2PL_VY_TRUE_PLURAL,
    CELL_2PL_LEGACY,
)

# Frequentist inference cell set (Q1, Q1b, Q1c, robustness): 4-cell. Polite-singular
# is dropped from frequentist inference because of corpus sparsity (23 events /
# 18 poems / 13 authors; only 3 authors have events in both P1 and P2, causing
# separation in FE designs and degenerate cluster-robust covariance in main GLM).
PRIMARY_GLM_CELLS_FREQUENTIST: tuple[str, ...] = (
    "1sg",
    "1pl",
    "2sg",
    CELL_2PL_VY_TRUE_PLURAL,
)

# Bayesian inference cell set (Q2 hierarchical): 5-cell. Negative-binomial random-
# slope shrinkage produces meaningful (if wide) HDIs even when the cell is sparse,
# so polite-singular is retained on the Bayesian path. The HDI for that cell will
# be wide; the inference is honest because the prior pulls toward zero.
PRIMARY_GLM_CELLS_BAYESIAN: tuple[str, ...] = (
    "1sg",
    "1pl",
    "2sg",
    CELL_2PL_VY_POLITE_SINGULAR,
    CELL_2PL_VY_TRUE_PLURAL,
)

# Back-compat alias: existing scripts that say `PRIMARY_GLM_CELLS` get the
# frequentist 4-cell set. Q2 explicitly imports `PRIMARY_GLM_CELLS_BAYESIAN`.
PRIMARY_GLM_CELLS: tuple[str, ...] = PRIMARY_GLM_CELLS_FREQUENTIST

ENABLED_CELLS = PRIMARY_GLM_CELLS

VY_REGISTER_POLITE_SINGULAR = "polite_singular"

# Count-construction set: 5-cell (includes polite-singular). The cell-counts CSV
# emits all five columns even though frequentist inference iterates a subset.
COUNT_INPUT_CELLS: tuple[str, ...] = PRIMARY_GLM_CELLS_BAYESIAN

# Stable 5-cell sum used to compute `n_total` per poem, decoupled from any
# inference loop. Downstream gates (e.g. `--min-total-per-poem`) compare against
# this sum, so the gate semantics do not shift when an inference cell-set changes.
N_TOTAL_CELLS: tuple[str, ...] = PRIMARY_GLM_CELLS_BAYESIAN


def poem_person_cell_column(df: pd.DataFrame) -> pd.Series:
    """Map each row to a poem aggregation cell including split 2nd-plural ви-cells.

    Non-2pl rows match ``pronoun_class_sixway_column``. Rows with morphological ``2pl`` split:
    ``polite_singular`` vy_register → ``2pl_vy_polite_singular``; all other vy values → ``2pl_vy_true_plural``.

    Optional future columns (not required): ``addressee_is_singular`` (bool)—when present with no NaN,
    polite-singular is further restricted to that flag; rows with missing refinement stay on vy_register-only rule.
    """
    pn = pronoun_class_sixway_column(df)
    out = pn.astype(object).copy()
    is_2pl = pn.eq("2pl")

    if "vy_register" in df.columns:
        vy = df["vy_register"].fillna("").astype(str).str.strip()
    else:
        vy = pd.Series("", index=df.index, dtype=object)

    polite_core = vy.eq(VY_REGISTER_POLITE_SINGULAR)
    if "addressee_is_singular" in df.columns:
        ads = df["addressee_is_singular"]
        defined = ads.notna()
        singular_addressee = ads.astype(bool)
        polite = polite_core & (~defined | singular_addressee)
    else:
        polite = polite_core

    out.loc[is_2pl & polite] = CELL_2PL_VY_POLITE_SINGULAR
    out.loc[is_2pl & ~polite] = CELL_2PL_VY_TRUE_PLURAL
    return out


def normalize_annotation_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def pronoun_class_sixway_column(df: pd.DataFrame) -> pd.Series:
    """Vector map of ``person`` × ``number`` to ``{1sg,1pl,2sg,2pl,3sg,3pl}``."""
    if "person" not in df.columns or "number" not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    person = df["person"].fillna("").astype(str).str.strip()
    number = df["number"].fillna("").astype(str).str.strip()
    has_sing = number.str.contains("Sing", na=False, regex=False)
    has_plur = number.str.contains("Plur", na=False, regex=False)
    p1 = person.eq("1st")
    p2 = person.eq("2nd")
    p3 = person.eq("3rd")
    conds = [p1 & has_sing, p1 & has_plur, p2 & has_sing, p2 & has_plur, p3 & has_sing, p3 & has_plur]
    labels = ("1sg", "1pl", "2sg", "2pl", "3sg", "3pl")
    return pd.Series(np.select(conds, labels, default=""), index=df.index, dtype=object)


def pronoun_class_sixway(row: pd.Series) -> str | None:
    """Map person (1st/2nd/3rd) + number (Singular/Plural) to {1sg,1pl,2sg,2pl,3sg,3pl}."""
    person = normalize_annotation_str(row.get("person"))
    number = normalize_annotation_str(row.get("number"))
    if person == "1st" and "Sing" in number:
        return "1sg"
    if person == "1st" and "Plur" in number:
        return "1pl"
    if person == "2nd" and "Sing" in number:
        return "2sg"
    if person == "2nd" and "Plur" in number:
        return "2pl"
    if person == "3rd" and "Sing" in number:
        return "3sg"
    if person == "3rd" and "Plur" in number:
        return "3pl"
    return None


def ordered_index_for_crosstab(
    index: pd.Index,
    preferred: list[str] | None = None,
) -> list[str]:
    values = [str(x) for x in index]
    if preferred is None:
        return values
    head = [x for x in preferred if x in values]
    tail = [x for x in values if x not in head]
    return head + tail
