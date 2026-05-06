"""Descriptive statistics on stanza-level pronoun annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import mannwhitneyu, norm

try:
    from scipy.stats import false_discovery_control as _fdr_bh
except ImportError:
    _fdr_bh = None                                  

try:
    from statsmodels.stats.multitest import multipletests as _multipletests_bh
except ImportError:
    _multipletests_bh = None                                  

from utils.adaptive_temporal_binning import (              
    INITIAL_MONTHS,
    MIN_POEMS_PER_INTERVAL,
    TARGET_POEMS_PER_BALANCED_BIN,
    adaptive_binning,
    balanced_temporal_binning,
)
from utils.pronoun_encoding import pronoun_class_sixway, pronoun_class_sixway_column
from utils.stats_common import mode_with_tie_order, normalize_bool_flag, period_pre_post_2022

                                                                             
          
                                                                             
DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "03_reporting_descriptive_statistics"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"

                                                                    
PERIOD_DESCRIPTIVE_ORDER = ("pre_2022", "post_2022")
PROP_FEATURES = ["prop_1st", "prop_2nd", "prop_3rd", "prop_plural", "prop_pro_drop"]
                                                                                   
PROP_FEATURES_FOR_PLOT = [c for c in PROP_FEATURES if c != "prop_3rd"]
VLINE_YEARS = (2014, 2022)

MW_PERIOD_PRE = "pre_2022"
MW_PERIOD_POST = "post_2022"
MIN_PRONOUNS_FOR_INFERENCE = 5
MIN_ROWS_FOR_LANGUAGE_MW = 10
                                                                                       
MIN_STANZAS_PER_PERIOD_FOR_LANGUAGE_INFERENCE = 10
                                                                    
MIN_POEMS_EACH_PERIOD_FOR_AUTHOR_PRE_POST = 6

                                                                                              
WILSON_CONFIDENCE_LEVEL = 0.95

PERSON_ORDER = ["1st", "2nd", "3rd", "Impersonal"]
NUMBER_ORDER = ["Singular", "Plural"]
SIXWAY_ORDER = ["1sg", "1pl", "2sg", "2pl", "3sg", "3pl"]
MAJOR_LANGUAGES = ["Ukrainian", "Russian", "Qirimli"]

                                                                                                   
SIXCODE_TO_DISPLAY = {
    "1sg": "1st Singular",
    "1pl": "1st Plural",
    "2sg": "2nd Singular",
    "2pl": "2nd Plural",
    "3sg": "3rd Singular",
    "3pl": "3rd Plural",
}

                                                                               
PN_KEY_ORDER = [
    "1st Singular",
    "1st Plural",
    "2nd Singular",
    "2nd Plural",
    "3rd Singular",
    "3rd Plural",
    "Impersonal/Other",
    "Other/ambiguous",
]

TOKEN_TREND_ORDER = list(SIXCODE_TO_DISPLAY.values()) + ["Impersonal/Other", "Other/ambiguous"]

                                                                           
DISPLAY_OMIT_PERSPECTIVE_PRIMARY = frozenset({"Impersonal/Other", "Other/ambiguous"})
                                                                                                                  
DISPLAY_OMIT_THIRD_PERSON_PN_LABELS = frozenset({"3rd Singular", "3rd Plural"})
DISPLAY_OMIT_TOKEN_PN = DISPLAY_OMIT_PERSPECTIVE_PRIMARY | DISPLAY_OMIT_THIRD_PERSON_PN_LABELS


def _wilson_lower_bound_pct(
    successes: int | float,
    trials: int | float,
    *,
    confidence: float = WILSON_CONFIDENCE_LEVEL,
) -> float:
    """Lower endpoint of the two-sided Wilson score interval for Binomial(trials, p), as % i..."""
    t = float(trials)
    if t <= 0 or not np.isfinite(t):
        return float("nan")
    x = float(min(max(successes, 0.0), t))
    p = x / t
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / t
    center = p + z2 / (2.0 * t)
    rad = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * t)) / t)
    lower = (center - rad) / denom
    return float(max(0.0, min(100.0, lower * 100.0)))


                                                                             
                                
                                                                             

def _period_pre_post_2022(y) -> str:
    """Calendar year: < 2022 vs >= 2022 (same for ``period`` and ``period_binary_2022``)."""
    return period_pre_post_2022(y)


def load_stanza_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period"] = df["year_int"].map(_period_pre_post_2022)
    df["person"] = df["person"].fillna("").str.strip()
    df["number"] = df["number"].fillna("").str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
    df["language_clean"] = df["language"].fillna("").str.strip()
    df["language_group"] = df["language_clean"].apply(
        lambda x: x if x in MAJOR_LANGUAGES else "Other"
    )
    df["poem_id"] = df["poem_id"].astype(str)
    return df


def attach_adaptive_intervals(
    df: pd.DataFrame,
    *,
    min_poems: int | None = None,
    initial_months: int | None = None,
    temporal_binning: str = "balanced",
    target_poems_per_bin: int | None = None,
    balance_by: str = "poems",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each poem to a temporal interval for B/C/D trend plots."""
    min_poems = MIN_POEMS_PER_INTERVAL if min_poems is None else int(min_poems)
    initial_months = INITIAL_MONTHS if initial_months is None else int(initial_months)
    mode = (temporal_binning or "balanced").strip().lower()
    bal_axis = (balance_by or "poems").strip().lower()
    if bal_axis not in ("poems", "stanzas"):
        bal_axis = "poems"
    tpb = TARGET_POEMS_PER_BALANCED_BIN if target_poems_per_bin is None else int(target_poems_per_bin)

    dfp = df.copy()
    if "date" in dfp.columns:
        dfp["_bin_date"] = pd.to_datetime(dfp["date"], errors="coerce")
    else:
        y = pd.to_numeric(dfp["year"], errors="coerce")
        dfp["_bin_date"] = pd.to_datetime(
            pd.DataFrame({"year": y.astype(float), "month": 7, "day": 1}),
            errors="coerce",
        )

    poem_ref = (
        dfp.loc[dfp["_bin_date"].notna(), ["poem_id", "_bin_date"]]
        .sort_values("_bin_date")
        .drop_duplicates(subset=["poem_id"], keep="first")
    )
    poem_ref = poem_ref.rename(columns={"_bin_date": "date"})

    if (
        mode == "balanced"
        and bal_axis == "stanzas"
        and "stanza_index" in dfp.columns
    ):
        stz = (
            dfp.loc[dfp["stanza_index"].notna(), ["poem_id", "stanza_index"]]
            .drop_duplicates()
            .groupby("poem_id", as_index=False)
            .size()
            .rename(columns={"size": "n_stanzas"})
        )
        poem_ref = poem_ref.merge(stz, on="poem_id", how="left")
        poem_ref["n_stanzas"] = poem_ref["n_stanzas"].fillna(1).astype(int)

    empty_iv = pd.DataFrame(columns=["interval_id", "interval_label", "n_poems", "start_date", "end_date"])
    if poem_ref.empty:
        out = dfp.drop(columns=["_bin_date"], errors="ignore")
        for c in ("interval_id", "interval_label", "interval_n_poems", "interval_start_date", "interval_end_date"):
            out[c] = np.nan
        return out, empty_iv

    if mode in ("min_calendar", "legacy", "calendar_merge"):
        dated, interval_df = adaptive_binning(
            poem_ref,
            date_col="date",
            id_col="poem_id",
            min_poems=min_poems,
            initial_months=initial_months,
        )
    else:
        wcol = "n_stanzas" if bal_axis == "stanzas" and "n_stanzas" in poem_ref.columns else None
        dated, interval_df = balanced_temporal_binning(
            poem_ref,
            date_col="date",
            id_col="poem_id",
            target_poems_per_bin=tpb,
            min_poems=min_poems,
            weight_col=wcol,
            balance_axis="stanzas" if wcol else "poems",
        )
    iv_map = dated[
        ["poem_id", "interval_id", "interval_label", "n_poems", "start_date", "end_date"]
    ].drop_duplicates(subset=["poem_id"])
    iv_map = iv_map.rename(
        columns={
            "n_poems": "interval_n_poems",
            "start_date": "interval_start_date",
            "end_date": "interval_end_date",
        }
    )
    out = dfp.drop(columns=["_bin_date"], errors="ignore").merge(iv_map, on="poem_id", how="left")
    return out, interval_df


def _normalize_bool_flag(series: pd.Series) -> pd.Series:
    return normalize_bool_flag(series)


def attach_repeat_translation_and_filter(
    df: pd.DataFrame, layer0_path: Path | None
) -> tuple[pd.DataFrame, dict]:
    """Drop repeat & translation poems using CSV columns if present, else layer0 flags."""
    n_rows0 = len(df)
    n_poems0 = int(df["poem_id"].nunique())
    diag: dict = {
        "layer0_path": str(layer0_path) if layer0_path else "",
        "layer0_loaded": False,
        "repeat_translation_source": "",
    }
    df = df.copy()

    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = _normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = _normalize_bool_flag(df["is_translation"])
        diag["repeat_translation_source"] = "input_csv_columns"
    elif layer0_path is not None and Path(layer0_path).is_file():
        l0 = pd.read_csv(
            layer0_path,
            usecols=["poem_id", "Is repeat", "I.D. of original (if poem is a translation)"],
            low_memory=False,
        )
        oid = l0["I.D. of original (if poem is a translation)"]
        flags = pd.DataFrame(
            {
                "poem_id": l0["poem_id"].astype(str).str.strip(),
                "is_repeat": l0["Is repeat"].astype(str).str.lower().str.strip().eq("yes"),
                "is_translation": oid.notna() & oid.astype(str).str.strip().ne(""),
            }
        ).drop_duplicates(subset=["poem_id"], keep="first")
        df = df.merge(flags, on="poem_id", how="left")
        df["is_repeat"] = df["is_repeat"].fillna(False).astype(bool)
        df["is_translation"] = df["is_translation"].fillna(False).astype(bool)
        diag["layer0_loaded"] = True
        diag["repeat_translation_source"] = "layer0_merge"
    else:
        df = df.assign(is_repeat=False, is_translation=False)
        diag["repeat_translation_source"] = "none_assumed_false"

    excl = df["is_repeat"] | df["is_translation"]
    diag["n_rows_excluded"] = int(excl.sum())
    diag["n_poems_touching_repeat"] = int(df.loc[df["is_repeat"], "poem_id"].nunique())
    diag["n_poems_touching_translation"] = int(df.loc[df["is_translation"], "poem_id"].nunique())
    df_out = df.loc[~excl].copy()
    diag["n_rows_after_filter"] = len(df_out)
    diag["n_poems_after_filter"] = int(df_out["poem_id"].nunique())
    diag["n_poems_before_filter"] = n_poems0
    diag["n_rows_before_filter"] = n_rows0
    return df_out, diag


def _pro_drop_mask(series: pd.Series) -> pd.Series:
    """True where the row is annotated as pro-drop (string or bool)."""
    s = series
    if getattr(s.dtype, "name", "") == "bool":
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def pronoun_row_pn_label(row: pd.Series) -> str:
    """One label per pronoun row: full person×number, or buckets for impersonal / bad rows."""
    p = str(row.get("person", "")).strip()
    if p == "Impersonal":
        return "Impersonal/Other"
    code = pronoun_class_sixway(row) or ""
    if code:
        return SIXCODE_TO_DISPLAY[code]
    return "Other/ambiguous"


def compute_poem_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Compute poem_perspective_primary/secondary from pronoun annotations."""
    pronouns = df[df["pronoun_word"].notna()].copy()
    if pronouns.empty:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "poem_perspective_primary",
                "poem_perspective_secondary",
                "perspective_confidence",
            ]
        )

    pronouns["pn_key"] = pronouns.apply(pronoun_row_pn_label, axis=1)

    results = []
    for pid, grp in pronouns.groupby("poem_id"):
        counts = grp["pn_key"].value_counts()
        total = counts.sum()
        if total == 0:
            continue

        primary = counts.index[0]
        primary_share = counts.iloc[0] / total

        secondary = None
        if len(counts) > 1:
            sec_share = counts.iloc[1] / total
            if sec_share >= 0.30:
                secondary = counts.index[1]

        results.append(
            {
                "poem_id": pid,
                "poem_perspective_primary": primary,
                "poem_perspective_secondary": secondary,
                "perspective_confidence": round(float(primary_share), 3),
            }
        )
    return pd.DataFrame(results)


def _ordered_primary_categories(series: pd.Series) -> list[str]:
    """Stable column order: canonical PN keys first, then any other observed primaries."""
    seen = set(series.dropna().astype(str).str.strip())
    seen.discard("")
    ordered = [c for c in PN_KEY_ORDER if c in seen]
    extras = sorted(seen - set(PN_KEY_ORDER))
    return ordered + extras


def _perspective_categories_for_display(series: pd.Series) -> list[str]:
    """Primary perspective labels for F tables, omitting impersonal/other buckets."""
    return [c for c in _ordered_primary_categories(series) if c not in DISPLAY_OMIT_PERSPECTIVE_PRIMARY]


def _perspective_categories_for_trend_plots(series: pd.Series) -> list[str]:
    """C/D interval trend plots: omit impersonal/other and third-person primary labels."""
    omit = DISPLAY_OMIT_PERSPECTIVE_PRIMARY | DISPLAY_OMIT_THIRD_PERSON_PN_LABELS
    return [c for c in _ordered_primary_categories(series) if c not in omit]


def build_poem_table_with_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Merge compute_poem_perspective() with poem-level metadata from the full frame."""
    persp = compute_poem_perspective(df)
    if persp.empty:
        return persp

    agg_kw: dict = {
        "author": ("author", "first"),
        "language_clean": ("language_clean", "first"),
        "language_group": ("language_group", "first"),
        "year_int": ("year_int", "first"),
        "period": ("period", "first"),
        "n_stanzas": ("stanza_index", "max"),
    }
    if "interval_id" in df.columns:
        agg_kw["interval_id"] = ("interval_id", "first")
        agg_kw["interval_label"] = ("interval_label", "first")
        agg_kw["interval_n_poems"] = ("interval_n_poems", "first")
        agg_kw["interval_start_date"] = ("interval_start_date", "first")
        agg_kw["interval_end_date"] = ("interval_end_date", "first")
    meta = df.groupby("poem_id", as_index=False).agg(**agg_kw)
    meta["n_stanzas"] = meta["n_stanzas"].fillna(1).astype(int)

    n_pronoun = (
        df[df["pronoun_word"].notna()]
        .groupby("poem_id")
        .size()
        .rename("n_pronoun_rows")
        .reset_index()
    )

    out = persp.merge(meta, on="poem_id", how="left").merge(n_pronoun, on="poem_id", how="left")
    out["period"] = out["year_int"].map(_period_pre_post_2022)
    return out


def compute_poem_pronoun_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per poem with continuous pronoun-distribution features (token proportions)."""
    pronouns = df[df["pronoun_word"].notna()].copy()
    if pronouns.empty:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "n_pronouns",
                "prop_1st",
                "prop_2nd",
                "prop_3rd",
                "prop_plural",
                "prop_pro_drop",
            ]
        )

    is_pd = _pro_drop_mask(pronouns["is_pro_drop"])
    pronouns = pronouns.assign(
        _p1=(pronouns["person"] == "1st").astype(np.int32),
        _p2=(pronouns["person"] == "2nd").astype(np.int32),
        _p3=(pronouns["person"] == "3rd").astype(np.int32),
        _pl=(pronouns["number"] == "Plural").astype(np.int32),
        _pd=is_pd.astype(np.int32),
    )
    agg = pronouns.groupby("poem_id", as_index=False).agg(
        n_pronouns=("poem_id", "count"),
        _s1=("_p1", "sum"),
        _s2=("_p2", "sum"),
        _s3=("_p3", "sum"),
        _spl=("_pl", "sum"),
        _spd=("_pd", "sum"),
    )
    n = agg["n_pronouns"].replace(0, np.nan)
    agg["prop_1st"] = agg["_s1"] / n
    agg["prop_2nd"] = agg["_s2"] / n
    agg["prop_3rd"] = agg["_s3"] / n
    agg["prop_plural"] = agg["_spl"] / n
    agg["prop_pro_drop"] = agg["_spd"] / n
    return agg[
        [
            "poem_id",
            "n_pronouns",
            "prop_1st",
            "prop_2nd",
            "prop_3rd",
            "prop_plural",
            "prop_pro_drop",
        ]
    ]


def build_poems_with_pronoun_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Merge ``compute_poem_pronoun_proportions`` with poem-level period / language / author."""
    props = compute_poem_pronoun_proportions(df)
    if props.empty:
        return props
    meta = df.groupby("poem_id", as_index=False).agg(
        author=("author", "first"),
        language_clean=("language_clean", "first"),
        language_group=("language_group", "first"),
        year_int=("year_int", "first"),
        period=("period", "first"),
    )
    out = props.merge(meta, on="poem_id", how="left")
    out["period"] = out["year_int"].map(_period_pre_post_2022)
    out["period_binary_2022"] = out["period"]
    if "is_repeat" in df.columns:
        ir = df.groupby("poem_id", as_index=False)["is_repeat"].max()
        ir["is_repeat"] = _normalize_bool_flag(ir["is_repeat"])
        out = out.merge(ir, on="poem_id", how="left")
    return out


def write_author_concentration(poem_props: pd.DataFrame, out: Path) -> None:
    """Top authors by distinct poem count (descriptive concentration)."""
    if poem_props.empty or "author" not in poem_props.columns:
        return
    vc = poem_props.groupby("author")["poem_id"].nunique().sort_values(ascending=False)
    total = int(vc.sum())
    if total == 0:
        return
    rows: list[dict] = []
    for i, (auth, n) in enumerate(vc.head(5).items(), start=1):
        nn = int(n)
        p_share = round(100 * nn / total, 2)
        w_lb = round(_wilson_lower_bound_pct(nn, total), 2)
        rows.append(
            {
                "rank": i,
                "author": auth,
                "n_poems": nn,
                "pct_of_poems": p_share,
                "pct_of_poems_wilson_lb": w_lb,
            }
        )
    top5 = int(vc.head(5).sum())
    rows.append(
        {
            "rank": 0,
            "author": "(top_5_combined)",
            "n_poems": top5,
            "pct_of_poems": round(100 * top5 / total, 2),
            "pct_of_poems_wilson_lb": round(_wilson_lower_bound_pct(top5, total), 2),
        }
    )
    rows.append(
        {
            "rank": -1,
            "author": "(all_other_authors)",
            "n_poems": total - top5,
            "pct_of_poems": round(100 * (total - top5) / total, 2),
            "pct_of_poems_wilson_lb": round(_wilson_lower_bound_pct(total - top5, total), 2),
        }
    )
    pd.DataFrame(rows).to_csv(out / "E_top_authors_by_poem_share.csv", index=False)


PROP_FEATURE_LABELS = {
    "prop_1st": "1st (token %)",
    "prop_2nd": "2nd (token %)",
    "prop_3rd": "3rd (token %)",
    "prop_plural": "Plural (token %)",
    "prop_pro_drop": "Pro-drop (token %)",
}


def _slug_author_filename(author: str) -> str:
    raw = str(author).strip()[:100]
    slug = re.sub(r"[^\w\-.]+", "_", raw, flags=re.UNICODE)
    return (slug.strip("_") or "author")[:120]


def _stanza_pn_percentages_by_period(stanza_df: pd.DataFrame, cats: list[str]) -> tuple[dict, dict]:
    """Map period -> {category: pct} for pre/post."""
    out: dict[str, dict[str, float]] = {MW_PERIOD_PRE: {}, MW_PERIOD_POST: {}}
    for per in (MW_PERIOD_PRE, MW_PERIOD_POST):
        sub = stanza_df[stanza_df["period"] == per]
        n = len(sub)
        if n == 0:
            for c in cats:
                out[per][c] = 0.0
            continue
        vc = sub["stanza_pn_mode"].value_counts()
        for c in cats:
            out[per][c] = round(100.0 * float(vc.get(c, 0)) / n, 2)
    return out[MW_PERIOD_PRE], out[MW_PERIOD_POST]


def write_author_pre_post_bridge_plots(
    df: pd.DataFrame,
    poem_props: pd.DataFrame,
    out: Path,
    *,
    min_poems_each_period: int = MIN_POEMS_EACH_PERIOD_FOR_AUTHOR_PRE_POST,
) -> int:
    """One figure per author with pre/post 2022 poems in both eras (≥ ``min_poems_each_perio..."""
    if poem_props.empty or "author" not in poem_props.columns:
        return 0
    if "stanza_index" not in df.columns:
        return 0

    pp = poem_props.copy()
    pp = pp[pp["period_binary_2022"].isin([MW_PERIOD_PRE, MW_PERIOD_POST])]
    if "is_repeat" in pp.columns:
        pp = pp.loc[~pp["is_repeat"].fillna(False).astype(bool)]
    pp = pp.loc[pp["author"].notna() & pp["author"].astype(str).str.strip().ne("")]

    cnt = pp.groupby(["author", "period_binary_2022"])["poem_id"].nunique()
    authors_sel: list[str] = []
    for auth in pp["author"].dropna().astype(str).unique():
        n_pre = int(cnt.get((auth, MW_PERIOD_PRE), 0))
        n_post = int(cnt.get((auth, MW_PERIOD_POST), 0))
        if n_pre >= min_poems_each_period and n_post >= min_poems_each_period:
            authors_sel.append(auth)

    if not authors_sel:
        return 0

    subdir = out / "author_pre_post_2022"
    subdir.mkdir(parents=True, exist_ok=True)
    cats_display = [c for c in TOKEN_TREND_ORDER if c not in DISPLAY_OMIT_TOKEN_PN]

    manifest: list[dict] = []
    for author in sorted(authors_sel, key=lambda a: a.lower()):
        adf = df[(df["author"].astype(str) == author) & (df["period"].isin(PERIOD_DESCRIPTIVE_ORDER))].copy()
        if adf.empty:
            continue
        stanza_df = build_stanza_unit_pronoun_frame_simple(adf)
        if stanza_df.empty:
            stanza_pre, stanza_post = {}, {}
            for c in cats_display:
                stanza_pre[c] = stanza_post[c] = 0.0
        else:
            stanza_pre, stanza_post = _stanza_pn_percentages_by_period(stanza_df, cats_display)

        pauth = pp[pp["author"].astype(str) == author]
        pre_m = pauth[pauth["period_binary_2022"] == MW_PERIOD_PRE][PROP_FEATURES_FOR_PLOT].mean(numeric_only=True)
        post_m = pauth[pauth["period_binary_2022"] == MW_PERIOD_POST][PROP_FEATURES_FOR_PLOT].mean(numeric_only=True)
        n_pre = int(pauth[pauth["period_binary_2022"] == MW_PERIOD_PRE]["poem_id"].nunique())
        n_post = int(pauth[pauth["period_binary_2022"] == MW_PERIOD_POST]["poem_id"].nunique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(PROP_FEATURES_FOR_PLOT))
        w = 0.36
        pre_vals = [float(pre_m.get(c, np.nan)) * 100.0 if np.isfinite(pre_m.get(c, np.nan)) else 0.0 for c in PROP_FEATURES_FOR_PLOT]
        post_vals = [float(post_m.get(c, np.nan)) * 100.0 if np.isfinite(post_m.get(c, np.nan)) else 0.0 for c in PROP_FEATURES_FOR_PLOT]
        axes[0].bar(x - w / 2, pre_vals, width=w, label="pre_2022", alpha=0.85)
        axes[0].bar(x + w / 2, post_vals, width=w, label="post_2022", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([PROP_FEATURE_LABELS[c] for c in PROP_FEATURES_FOR_PLOT], rotation=25, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean % of pronoun tokens per poem")
        axes[0].set_title(f"Poem-level (n={n_pre} / {n_post} poems)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.25)

        xs = np.arange(len(cats_display))
        spre = [stanza_pre.get(c, 0.0) for c in cats_display]
        spost = [stanza_post.get(c, 0.0) for c in cats_display]
        axes[1].bar(xs - w / 2, spre, width=w, label="pre_2022", alpha=0.85)
        axes[1].bar(xs + w / 2, spost, width=w, label="post_2022", alpha=0.85)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(cats_display, rotation=35, ha="right", fontsize=7)
        axes[1].set_ylabel("% of stanzas (modal person×number)")
        n_st_pre = int(stanza_df[stanza_df["period"] == MW_PERIOD_PRE].shape[0]) if not stanza_df.empty else 0
        n_st_post = int(stanza_df[stanza_df["period"] == MW_PERIOD_POST].shape[0]) if not stanza_df.empty else 0
        axes[1].set_title(f"Stanza-level (n={n_st_pre} / {n_st_post} stanzas w/ pronoun)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"{author}: pronoun use pre vs post 2022", fontsize=12, y=1.02)
        plt.tight_layout()
        fname = _slug_author_filename(author) + ".png"
        fig.savefig(subdir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "author": author,
                "n_poems_pre_2022": n_pre,
                "n_poems_post_2022": n_post,
                "plot_file": str((subdir / fname).relative_to(out)),
            }
        )

    pd.DataFrame(manifest).sort_values("author").to_csv(subdir / "manifest.csv", index=False)
    return len(manifest)


def write_exclusion_diagnostics(diag: dict, out: Path) -> None:
    pd.DataFrame([diag]).to_csv(out / "corpus_exclusions_repeat_translation.csv", index=False)


                                                                             
                                                                  
                                                                             


def _cohens_d_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples: (mean(b) − mean(a)) / pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0 if b.mean() == a.mean() else float("nan")
    return float((b.mean() - a.mean()) / pooled)


def _assign_bh_fdr(rows: list[dict], *, p_key: str = "p_value", out_key: str = "p_adjusted_bh") -> None:
    """Benjamini–Hochberg across rows with finite ``p_key`` (one family = all rows passed in)."""
    idx = [i for i, r in enumerate(rows) if np.isfinite(r.get(p_key, np.nan))]
    for r in rows:
        r[out_key] = np.nan
    if not idx:
        return
    pvals = np.array([rows[i][p_key] for i in idx], dtype=float)
    if _fdr_bh is not None:
        adj = _fdr_bh(pvals, method="bh")
    elif _multipletests_bh is not None:
        _, adj, _, _ = _multipletests_bh(pvals, method="fdr_bh")
    else:
        adj = pvals
    for j, i in enumerate(idx):
        rows[i][out_key] = float(adj[j])


def _mann_whitney_continuous_block(
    sub: pd.DataFrame,
    feature_cols: list[str],
    *,
    scope: str,
    language: str,
    unit: str,
    period_col: str = "period_binary_2022",
    pre: str = MW_PERIOD_PRE,
    post: str = MW_PERIOD_POST,
) -> list[dict]:
    """Two-sided Mann–Whitney: post vs pre on each numeric column (one row per poem or binar..."""
    if period_col not in sub.columns:
        return []
    pre_full = sub[sub[period_col] == pre]
    post_full = sub[sub[period_col] == post]
    rows: list[dict] = []
    for col in feature_cols:
        xa = pre_full[col].dropna().to_numpy(dtype=float)
        xb = post_full[col].dropna().to_numpy(dtype=float)
        base = {
            "scope": scope,
            "language": language or "",
            "unit": unit,
            "feature": col,
            "comparison": f"{post}_vs_{pre}",
            "n_pre": int(xa.size),
            "n_post": int(xb.size),
        }
        if xa.size < 3 or xb.size < 3:
            rows.append(
                {
                    **base,
                    "median_pre": np.nan,
                    "median_post": np.nan,
                    "mean_pre": np.nan,
                    "mean_post": np.nan,
                    "pct_positive_pre": np.nan,
                    "pct_positive_post": np.nan,
                    "u_statistic": np.nan,
                    "p_value": np.nan,
                    "cohens_d_post_minus_pre": np.nan,
                }
            )
            continue
        res = mannwhitneyu(xb, xa, alternative="two-sided")
        rows.append(
            {
                **base,
                "median_pre": round(float(np.median(xa)), 6),
                "median_post": round(float(np.median(xb)), 6),
                "mean_pre": round(float(np.mean(xa)), 6),
                "mean_post": round(float(np.mean(xb)), 6),
                "pct_positive_pre": round(100.0 * float(np.mean(xa > 0)), 4) if xa.size else np.nan,
                "pct_positive_post": round(100.0 * float(np.mean(xb > 0)), 4) if xb.size else np.nan,
                "u_statistic": float(res.statistic),
                "p_value": float(res.pvalue),
                "cohens_d_post_minus_pre": round(_cohens_d_two_sample(xa, xb), 4),
            }
        )
    return rows


def _poem_language_subset_ok(sub: pd.DataFrame, *, period_col: str = "period_binary_2022") -> bool:
    if len(sub) < MIN_ROWS_FOR_LANGUAGE_MW or period_col not in sub.columns:
        return False
    n_pre = int((sub[period_col] == MW_PERIOD_PRE).sum())
    n_post = int((sub[period_col] == MW_PERIOD_POST).sum())
    return n_pre >= 3 and n_post >= 3


def _stanza_language_subset_ok(sub: pd.DataFrame, *, period_col: str = "period") -> bool:
    if period_col not in sub.columns or sub.empty:
        return False
    n_pre = int((sub[period_col] == MW_PERIOD_PRE).sum())
    n_post = int((sub[period_col] == MW_PERIOD_POST).sum())
    return (
        n_pre >= MIN_STANZAS_PER_PERIOD_FOR_LANGUAGE_INFERENCE
        and n_post >= MIN_STANZAS_PER_PERIOD_FOR_LANGUAGE_INFERENCE
    )


def write_pre_post_2022_pronoun_inference(
    df: pd.DataFrame,
    poem_props: pd.DataFrame,
    out: Path,
    *,
    min_n_pronouns: int = MIN_PRONOUNS_FOR_INFERENCE,
) -> None:
    """Mann–Whitney + Cohen's d + BH-FDR for pronoun shifts at the 2022 year cut-off."""
    subdir = out / "pre_post_2022_inference"
    subdir.mkdir(parents=True, exist_ok=True)

    poem_rows: list[dict] = []
    if not poem_props.empty:
        mask_period = poem_props["period_binary_2022"].isin([MW_PERIOD_PRE, MW_PERIOD_POST])
        mask_n = poem_props["n_pronouns"] >= int(min_n_pronouns)
        core = poem_props.loc[mask_period & mask_n].copy()
        if "is_repeat" in core.columns:
            core = core.loc[~core["is_repeat"].fillna(False).astype(bool)]

        blk = _mann_whitney_continuous_block(
            core,
            PROP_FEATURES,
            scope="overall",
            language="",
            unit="poem_token_fraction",
        )
        for r in blk:
            r["min_n_pronouns_in_poem"] = int(min_n_pronouns)
        poem_rows.extend(blk)

        for lang in MAJOR_LANGUAGES:
            sub = core[core["language_clean"] == lang]
            if not _poem_language_subset_ok(sub):
                continue
            blk_l = _mann_whitney_continuous_block(
                sub,
                PROP_FEATURES,
                scope="by_language",
                language=lang,
                unit="poem_token_fraction",
            )
            for r in blk_l:
                r["min_n_pronouns_in_poem"] = int(min_n_pronouns)
            poem_rows.extend(blk_l)

        overall_poem = [r for r in poem_rows if r["scope"] == "overall" and r["unit"] == "poem_token_fraction"]
        _assign_bh_fdr(overall_poem)
        o_adj = {id(r): r.get("p_adjusted_bh") for r in overall_poem}
        for r in poem_rows:
            r["p_adjusted_bh_within_scope"] = o_adj.get(id(r), np.nan)
            r["p_adjusted_bh_within_language_block"] = np.nan

        for lang in MAJOR_LANGUAGES:
            lang_rows = [r for r in poem_rows if r["scope"] == "by_language" and r["language"] == lang]
            if not lang_rows:
                continue
            _assign_bh_fdr(lang_rows)
            m = {id(r): r.get("p_adjusted_bh") for r in lang_rows}
            for r in poem_rows:
                if id(r) in m:
                    r["p_adjusted_bh_within_language_block"] = m[id(r)]

        pd.DataFrame(poem_rows).to_csv(subdir / "poem_token_fraction_mann_whitney.csv", index=False)

    if "stanza_index" in df.columns:
        d0 = df[df["period"].isin(PERIOD_DESCRIPTIVE_ORDER)].copy()
        if "is_repeat" in d0.columns:
            d0 = d0.loc[~d0["is_repeat"].fillna(False).astype(bool)]
        stanza_df = build_stanza_unit_pronoun_frame_simple(d0)
        if not stanza_df.empty:
            plang = d0.groupby("poem_id", as_index=False)["language_clean"].first()
            stanza_df = stanza_df.merge(plang, on="poem_id", how="left")

            modal_pn_rows: list[dict] = []
            for cat in TOKEN_TREND_ORDER:
                tdf = stanza_df.copy()
                tdf["y"] = (tdf["stanza_pn_mode"] == cat).astype(float)
                blk_s = _mann_whitney_continuous_block(
                    tdf,
                    ["y"],
                    scope="overall",
                    language="",
                    unit="stanza_modal_pn_indicator",
                    period_col="period",
                )
                for r in blk_s:
                    r["modal_pn_category"] = cat
                    r["feature"] = cat
                modal_pn_rows.extend(blk_s)
            _assign_bh_fdr(modal_pn_rows)
            for r in modal_pn_rows:
                r["p_adjusted_bh_across_modal_pn_categories"] = r.get("p_adjusted_bh", np.nan)
                r["p_adjusted_bh_within_language_modal_pn"] = np.nan

            for lang in MAJOR_LANGUAGES:
                s_lang = stanza_df[stanza_df["language_clean"] == lang]
                if not _stanza_language_subset_ok(s_lang):
                    continue
                lang_pn: list[dict] = []
                for cat in TOKEN_TREND_ORDER:
                    tdf = s_lang.copy()
                    tdf["y"] = (tdf["stanza_pn_mode"] == cat).astype(float)
                    blk_s = _mann_whitney_continuous_block(
                        tdf,
                        ["y"],
                        scope="by_language",
                        language=lang,
                        unit="stanza_modal_pn_indicator",
                        period_col="period",
                    )
                    for r in blk_s:
                        r["modal_pn_category"] = cat
                        r["feature"] = cat
                    lang_pn.extend(blk_s)
                _assign_bh_fdr(lang_pn)
                for r in lang_pn:
                    r["p_adjusted_bh_across_modal_pn_categories"] = np.nan
                    r["p_adjusted_bh_within_language_modal_pn"] = r.get("p_adjusted_bh", np.nan)
                    r.pop("p_adjusted_bh", None)
                modal_pn_rows.extend(lang_pn)

            pd.DataFrame(modal_pn_rows).to_csv(subdir / "stanza_modal_pn_mann_whitney.csv", index=False)

            sdf = stanza_df[stanza_df["stanza_number_mode"].isin(NUMBER_ORDER)].copy()
            if not sdf.empty:
                num_rows: list[dict] = []
                for lab in NUMBER_ORDER:
                    tdf = sdf.copy()
                    tdf["y"] = (tdf["stanza_number_mode"] == lab).astype(float)
                    blk_n = _mann_whitney_continuous_block(
                        tdf,
                        ["y"],
                        scope="overall",
                        language="",
                        unit="stanza_modal_number_indicator",
                        period_col="period",
                    )
                    for r in blk_n:
                        r["modal_number_category"] = lab
                        r["feature"] = lab
                    num_rows.extend(blk_n)
                _assign_bh_fdr(num_rows)
                for r in num_rows:
                    r["p_adjusted_bh_across_number_modes"] = r.get("p_adjusted_bh", np.nan)
                    r["p_adjusted_bh_within_language_number_modes"] = np.nan

                for lang in MAJOR_LANGUAGES:
                    s_lang = sdf[sdf["language_clean"] == lang]
                    if not _stanza_language_subset_ok(s_lang):
                        continue
                    lang_num: list[dict] = []
                    for lab in NUMBER_ORDER:
                        tdf = s_lang.copy()
                        tdf["y"] = (tdf["stanza_number_mode"] == lab).astype(float)
                        blk_n = _mann_whitney_continuous_block(
                            tdf,
                            ["y"],
                            scope="by_language",
                            language=lang,
                            unit="stanza_modal_number_indicator",
                            period_col="period",
                        )
                        for r in blk_n:
                            r["modal_number_category"] = lab
                            r["feature"] = lab
                        lang_num.extend(blk_n)
                    _assign_bh_fdr(lang_num)
                    for r in lang_num:
                        r["p_adjusted_bh_across_number_modes"] = np.nan
                        r["p_adjusted_bh_within_language_number_modes"] = r.get("p_adjusted_bh", np.nan)
                        r.pop("p_adjusted_bh", None)
                    num_rows.extend(lang_num)

                pd.DataFrame(num_rows).to_csv(subdir / "stanza_modal_number_mann_whitney.csv", index=False)


                                                                             
                      
                                                                             

def write_corpus_overview(df: pd.DataFrame, poems: pd.DataFrame, out: Path):
    known = df[df["year_int"].notna()]
    tok = df["pronoun_word"].notna()
    overview = pd.DataFrame(
        [
            {"metric": "total_poems", "value": int(df["poem_id"].nunique())},
            {"metric": "total_stanzas", "value": int(df.groupby(["poem_id", "stanza_index"]).ngroups)},
            {"metric": "total_csv_rows", "value": len(df)},
            {"metric": "rows_with_pronoun_token", "value": int(tok.sum())},
            {"metric": "poems_with_known_year", "value": int(known["poem_id"].nunique())},
            {"metric": "year_range", "value": f"{int(known['year_int'].min())}–{int(known['year_int'].max())}"},
            *[
                {
                    "metric": f"poems_{p}",
                    "value": int(poems[poems["period"] == p].shape[0]),
                }
                for p in PERIOD_DESCRIPTIVE_ORDER
            ],
            {"metric": "poems_unknown_period", "value": int(poems[poems["period"] == "unknown"].shape[0])},
        ]
    )
    if "interval_id" in df.columns and df["interval_id"].notna().any():
        overview = pd.concat(
            [
                overview,
                pd.DataFrame(
                    [
                        {
                            "metric": "n_adaptive_intervals",
                            "value": int(df["interval_id"].dropna().nunique()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    overview.to_csv(out / "A_corpus_overview.csv", index=False)

    lang_poems = (
        df.groupby("language_clean")["poem_id"].nunique()
        .rename("n_poems").reset_index()
        .sort_values("n_poems", ascending=False)
    )
    lang_stanzas = (
        df.groupby("language_clean")
        .apply(lambda g: g.groupby(["poem_id", "stanza_index"]).ngroups, include_groups=False)
        .rename("n_stanzas")
        .reset_index()
    )
    lang_tok = df.loc[tok].groupby("language_clean").size().rename("n_rows_with_pronoun_token").reset_index()
    lang = lang_poems.merge(lang_stanzas, on="language_clean").merge(lang_tok, on="language_clean", how="left")
    lang["n_rows_with_pronoun_token"] = lang["n_rows_with_pronoun_token"].fillna(0).astype(int)
    lang.to_csv(out / "A_corpus_by_language.csv", index=False)

    if "interval_id" in df.columns and df["interval_id"].notna().any():
        iv_rows = []
        for iid in sorted(df["interval_id"].dropna().unique()):
            g = df[df["interval_id"] == iid]
            iv_rows.append(
                {
                    "interval_id": int(iid),
                    "interval_label": str(g["interval_label"].iloc[0]),
                    "interval_start_date": g["interval_start_date"].iloc[0],
                    "interval_end_date": g["interval_end_date"].iloc[0],
                    "n_poems_bin_definition": int(g["interval_n_poems"].iloc[0])
                    if pd.notna(g["interval_n_poems"].iloc[0])
                    else "",
                    "n_poems_touching_corpus": int(g["poem_id"].nunique()),
                    "n_stanzas": int(g.groupby(["poem_id", "stanza_index"]).ngroups),
                    "n_rows_with_pronoun_token": int(g["pronoun_word"].notna().sum()),
                }
            )
        pd.DataFrame(iv_rows).to_csv(out / "A_corpus_by_adaptive_interval.csv", index=False)


                                                                             
                                                
                                                                             

def _ordered_interval_ids(interval_df: pd.DataFrame) -> list[int]:
    if interval_df is None or interval_df.empty:
        return []
    return interval_df.sort_values("start_date")["interval_id"].astype(int).tolist()


def _pct_by_interval(
    df: pd.DataFrame,
    col: str,
    categories: list[str],
    *,
    interval_order: list[int] | None = None,
) -> pd.DataFrame:
    sub = df[df["interval_id"].notna() & df[col].isin(categories)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["interval_id"] = sub["interval_id"].astype(int)
    ct = pd.crosstab(sub["interval_id"], sub[col])
    for c in categories:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[categories]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    pct = pct.sort_index()
    if interval_order:
        pct = pct.reindex([i for i in interval_order if i in pct.index])
    return pct


def _mode_with_tie_order(series: pd.Series, preference: list[str]) -> str:
    """Modal category; ties broken by first match in ``preference``."""
    return mode_with_tie_order(series, preference)


def build_stanza_unit_pronoun_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (poem_id, stanza_index): mode label within stanza (equal weight per stanza)."""
    if "stanza_index" not in df.columns:
        raise ValueError("B trends require column stanza_index on the input CSV.")
    if "interval_id" not in df.columns:
        raise ValueError("B trends require adaptive intervals: run attach_adaptive_intervals() first.")
    tok = df[df["pronoun_word"].notna() & df["stanza_index"].notna() & df["interval_id"].notna()].copy()
    if tok.empty:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "stanza_index",
                "interval_id",
                "n_pronoun_tokens_in_stanza",
                "stanza_pn_mode",
                "stanza_number_mode",
            ]
        )
    tok["token_pn_label"] = tok.apply(pronoun_row_pn_label, axis=1)
    tok["stanza_index"] = pd.to_numeric(tok["stanza_index"], errors="coerce")
    tok = tok[tok["stanza_index"].notna()].copy()
    tok["interval_id"] = pd.to_numeric(tok["interval_id"], errors="coerce")

    rows: list[dict] = []
    for (pid, sid), g in tok.groupby(["poem_id", "stanza_index"], sort=False):
        iv = g["interval_id"].dropna()
        if iv.empty:
            continue
        pn_mode = _mode_with_tie_order(g["token_pn_label"], TOKEN_TREND_ORDER)
        gnum = g[g["number"].isin(NUMBER_ORDER)]
        num_mode = _mode_with_tie_order(gnum["number"], NUMBER_ORDER) if not gnum.empty else ""
        rows.append(
            {
                "poem_id": pid,
                "stanza_index": int(sid),
                "interval_id": int(iv.iloc[0]),
                "n_pronoun_tokens_in_stanza": int(len(g)),
                "stanza_pn_mode": pn_mode,
                "stanza_number_mode": num_mode,
            }
        )
    return pd.DataFrame(rows)


def build_stanza_unit_pronoun_frame_simple(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (poem_id, stanza_index): modal person×number; uses ``period`` on input ro..."""
    if "stanza_index" not in df.columns:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "stanza_index",
                "stanza_pn_mode",
                "stanza_number_mode",
                "period",
                "n_pronoun_tokens_in_stanza",
            ]
        )
    tok = df[df["pronoun_word"].notna() & df["stanza_index"].notna()].copy()
    if tok.empty:
        return pd.DataFrame(
            columns=[
                "poem_id",
                "stanza_index",
                "stanza_pn_mode",
                "stanza_number_mode",
                "period",
                "n_pronoun_tokens_in_stanza",
            ]
        )
    tok["token_pn_label"] = tok.apply(pronoun_row_pn_label, axis=1)
    tok["stanza_index"] = pd.to_numeric(tok["stanza_index"], errors="coerce")
    tok = tok[tok["stanza_index"].notna()].copy()
    rows: list[dict] = []
    for (pid, sid), g in tok.groupby(["poem_id", "stanza_index"], sort=False):
        per = str(g["period"].iloc[0]) if "period" in g.columns else "unknown"
        pn_mode = _mode_with_tie_order(g["token_pn_label"], TOKEN_TREND_ORDER)
        gnum = g[g["number"].isin(NUMBER_ORDER)]
        num_mode = _mode_with_tie_order(gnum["number"], NUMBER_ORDER) if not gnum.empty else ""
        rows.append(
            {
                "poem_id": pid,
                "stanza_index": int(sid),
                "stanza_pn_mode": pn_mode,
                "stanza_number_mode": num_mode,
                "period": per,
                "n_pronoun_tokens_in_stanza": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def write_pronoun_trend(
    df: pd.DataFrame,
    out: Path,
    interval_df: pd.DataFrame,
    *,
    interval_trend_x_axis: str = "ordinal",
) -> None:
    """B charts: each stanza counts once; x-axis = interval bins (ordinal or density layout)."""
    if interval_df is None or interval_df.empty:
        return
    io = _ordered_interval_ids(interval_df)
    stanza_df = build_stanza_unit_pronoun_frame(df)
    if stanza_df.empty or not io:
        return

    n_iv = (
        stanza_df.groupby("interval_id", sort=True)
        .size()
        .rename("n_stanzas_with_pronoun")
        .reset_index()
        .astype({"interval_id": int})
    )
    n_iv = n_iv.merge(
        interval_df[["interval_id", "interval_label", "start_date", "end_date", "n_poems"]],
        on="interval_id",
        how="left",
    )
    n_iv.to_csv(out / "B_stanza_unit_n_per_interval.csv", index=False)

    cats_display = [c for c in TOKEN_TREND_ORDER if c not in DISPLAY_OMIT_TOKEN_PN]
    stanza_df["interval_id"] = stanza_df["interval_id"].astype(int)
    ct_pn = pd.crosstab(stanza_df["interval_id"], stanza_df["stanza_pn_mode"])
    for c in TOKEN_TREND_ORDER:
        if c not in ct_pn.columns:
            ct_pn[c] = 0
    ct_pn = ct_pn.sort_index()
    row_tot_pn = ct_pn.sum(axis=1).replace(0, np.nan)
    pct_six = ct_pn[cats_display].div(row_tot_pn, axis=0) * 100
    pct_six = pct_six.reindex([i for i in io if i in pct_six.index])
    _attach_interval_metadata(pct_six, interval_df).to_csv(
        out / "B_pronoun_person_number_trend_by_interval.csv", index=False
    )

    num_df = stanza_df[stanza_df["stanza_number_mode"].isin(NUMBER_ORDER)].copy()
    pct_number = _pct_by_interval(num_df, "stanza_number_mode", NUMBER_ORDER, interval_order=io)
    _attach_interval_metadata(pct_number, interval_df).to_csv(
        out / "B_pronoun_number_trend_by_interval.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    _plot_adaptive_interval_trend(
        axes[0],
        pct_six,
        interval_df,
        "By stanza (mode): % of stanzas per person×number, by adaptive interval",
        interval_x_axis=interval_trend_x_axis,
    )
    _plot_adaptive_interval_trend(
        axes[1],
        pct_number,
        interval_df,
        "By stanza (mode): Singular vs Plural among stanzas with a number mode, by adaptive interval",
        interval_x_axis=interval_trend_x_axis,
    )

    plt.tight_layout()
    plt.savefig(out / "B_pronoun_trend_by_interval.png", dpi=150, bbox_inches="tight")
    plt.close()


def _chronological_interval_ids_for_plot(
    pct: pd.DataFrame,
    interval_df: pd.DataFrame,
) -> list[int]:
    """``interval_id``s present in ``pct``, ordered by ``start_date`` then ``interval_id``."""
    iv = interval_df.drop_duplicates(subset=["interval_id"]).copy()
    iv["start_date"] = pd.to_datetime(iv["start_date"])
    iv["end_date"] = pd.to_datetime(iv["end_date"])
    iv = iv.sort_values(["start_date", "end_date", "interval_id"])
    want = set(int(i) for i in pct.index.astype(int))
    return [int(r) for r in iv["interval_id"] if int(r) in want]


def _ordinal_equal_x(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit width per interval; centers at 0.5, 1.5, … (no calendar spacing)."""
    if n <= 0:
        return np.array([]), np.array([0.0])
    edges = np.arange(n + 1, dtype=float)
    xc = edges[:-1] + 0.5
    return xc, edges


def _density_spaced_x(
    order_ids: list[int],
    interval_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative x where each bin's width ∝ ``n_poems / calendar_days`` (dense eras widen)."""
    iv = interval_df.set_index("interval_id")
    widths: list[float] = []
    for iid in order_ids:
        row = iv.loc[int(iid)]
        s = pd.Timestamp(row["start_date"])
        e = pd.Timestamp(row["end_date"])
        n = max(1, int(row["n_poems"]))
        days = max(1, (e - s).days + 1)
        widths.append(float(n) / float(days))
    w = np.asarray(widths, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(w)])
    xc = (edges[:-1] + edges[1:]) / 2.0
    return xc, edges


def _year_vline_x_density(
    year: int,
    order_ids: list[int],
    interval_df: pd.DataFrame,
    edges: np.ndarray,
) -> float | None:
    """Map Jan 1 ``year`` to x in density layout (linear within the covering interval)."""
    ts = pd.Timestamp(int(year), 1, 1)
    iv = interval_df.set_index("interval_id")
    for i, iid in enumerate(order_ids):
        row = iv.loc[int(iid)]
        s = pd.Timestamp(row["start_date"])
        e = pd.Timestamp(row["end_date"])
        if e < ts:
            continue
        if s > ts:
            return float(edges[i])
        span = e - s
        if span <= pd.Timedelta(0):
            frac = 0.5
        else:
            frac = float(np.clip((ts - s) / span, 0.0, 1.0))
        return float(edges[i] + frac * (edges[i + 1] - edges[i]))
    return None


def _year_vline_x_ordinal(
    year: int,
    order_ids: list[int],
    interval_df: pd.DataFrame,
) -> float | None:
    """First chronological bin whose ``start_date`` is on/after Jan 1 ``year`` → boundary at..."""
    ts = pd.Timestamp(int(year), 1, 1)
    iv = interval_df.set_index("interval_id")
    for k, iid in enumerate(order_ids):
        s = pd.Timestamp(iv.loc[int(iid)]["start_date"])
        if s >= ts:
            return float(k) - 0.5
    return None


def _plot_adaptive_interval_trend(
    ax,
    pct: pd.DataFrame,
    interval_df: pd.DataFrame,
    title: str,
    *,
    vline_years: tuple[int, ...] = VLINE_YEARS,
    interval_x_axis: str = "ordinal",
) -> None:
    """Line chart over time bins: default **ordinal** (equal width per bin); optional **dens..."""
    if pct.empty or interval_df is None or interval_df.empty:
        ax.set_title(f"{title} (no interval data)")
        return
    order_ids = _chronological_interval_ids_for_plot(pct, interval_df)
    if not order_ids:
        ax.set_title(f"{title} (no interval data)")
        return
    pct_ord = pct.loc[order_ids]
    n = len(order_ids)
    mode = (interval_x_axis or "ordinal").strip().lower()
    if mode == "density":
        xn, edges = _density_spaced_x(order_ids, interval_df)

        def year_x(y: int) -> float | None:
            return _year_vline_x_density(y, order_ids, interval_df, edges)

        xlabel = "Position (width ∝ poems / day in bin; ticks ≈ bin start)"
    else:
        xn, _edges = _ordinal_equal_x(n)

        def year_x(y: int) -> float | None:
            return _year_vline_x_ordinal(y, order_ids, interval_df)

        xlabel = "Interval order (equal width; ticks ≈ bin start, not calendar scale)"

    for col in pct_ord.columns:
        ax.plot(xn, pct_ord[col].values, marker="o", markersize=4, label=col)
    for yv in vline_years:
        xv = year_x(yv)
        if xv is not None:
            ax.axvline(xv, color="grey", linestyle="--", linewidth=1, alpha=0.65)
    iv = interval_df.set_index("interval_id")
    step = max(1, n // 10)
    tick_idx = list(range(0, n, step))
    ax.set_xticks(xn[tick_idx])
    ax.set_xticklabels(
        [iv.loc[int(order_ids[j])]["start_date"].strftime("%Y-%m") for j in tick_idx],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def _attach_interval_metadata(pct: pd.DataFrame, interval_df: pd.DataFrame) -> pd.DataFrame:
    """Join ``interval_label`` / dates / ``n_poems`` onto a crosstab-derived frame indexed b..."""
    if pct.empty:
        return pct.reset_index()
    out = pct.reset_index()
    if out.columns[0] != "interval_id":
        out = out.rename(columns={out.columns[0]: "interval_id"})
    meta = interval_df[
        ["interval_id", "interval_label", "start_date", "end_date", "n_poems"]
    ].drop_duplicates(subset=["interval_id"])
    return out.merge(meta, on="interval_id", how="left")


                                                                             
                             
                                                                             

def write_perspective_trend(
    poems: pd.DataFrame,
    out: Path,
    interval_df: pd.DataFrame,
    *,
    interval_trend_x_axis: str = "ordinal",
) -> None:
    if interval_df is None or interval_df.empty or "interval_id" not in poems.columns:
        return
    io = _ordered_interval_ids(interval_df)
    sub = poems[poems["interval_id"].notna()].copy()
    if sub.empty or not io:
        return
    sub["interval_id"] = sub["interval_id"].astype(int)
    cats_all = _ordered_primary_categories(sub["poem_perspective_primary"])
    cats = _perspective_categories_for_trend_plots(sub["poem_perspective_primary"])
    if not cats_all or not cats:
        return
    ct = pd.crosstab(sub["interval_id"], sub["poem_perspective_primary"])
    for c in cats_all:
        if c not in ct.columns:
            ct[c] = 0
    row_tot = ct.sum(axis=1).replace(0, np.nan)
    for c in cats:
        if c not in ct.columns:
            ct[c] = 0
    pct = ct[cats].div(row_tot, axis=0) * 100
    pct = pct.reindex([i for i in io if i in pct.index])
    _attach_interval_metadata(pct, interval_df).to_csv(out / "C_poem_perspective_trend_by_interval.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    _plot_adaptive_interval_trend(
        ax,
        pct,
        interval_df,
        "Poem perspective (post-hoc from pronouns) by adaptive interval",
        interval_x_axis=interval_trend_x_axis,
    )
    plt.tight_layout()
    plt.savefig(out / "C_poem_perspective_trend_by_interval.png", dpi=150, bbox_inches="tight")
    plt.close()


                                                                             
                                   
                                                                             

def write_perspective_by_language(
    poems: pd.DataFrame,
    out: Path,
    interval_df: pd.DataFrame,
    *,
    interval_trend_x_axis: str = "ordinal",
) -> None:
    if interval_df is None or interval_df.empty or "interval_id" not in poems.columns:
        return
    all_ids = _ordered_interval_ids(interval_df)
    sub = poems[poems["interval_id"].notna() & poems["language_group"].isin(MAJOR_LANGUAGES)].copy()
    if sub.empty or not all_ids:
        return
    sub["interval_id"] = sub["interval_id"].astype(int)

    langs = [l for l in MAJOR_LANGUAGES if l in sub["language_group"].values]
    n_langs = len(langs)
    if n_langs == 0:
        return

    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, lang in enumerate(langs):
        ldf = sub[sub["language_group"] == lang]
        if ldf.empty:
            continue
        cats_all = _ordered_primary_categories(ldf["poem_perspective_primary"])
        cats = _perspective_categories_for_trend_plots(ldf["poem_perspective_primary"])
        if not cats_all or not cats:
            continue
        ct = pd.crosstab(ldf["interval_id"], ldf["poem_perspective_primary"])
        for c in cats_all:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct.reindex(all_ids, fill_value=0)
        total = ct.sum(axis=1).replace(0, np.nan)
        for c in cats:
            if c not in ct.columns:
                ct[c] = 0
        pct = ct[cats].div(total, axis=0) * 100
        pct = pct.reindex([i for i in all_ids if i in pct.index])
        _attach_interval_metadata(pct, interval_df).to_csv(
            out / f"D_perspective_{lang.lower()}_trend_by_interval.csv", index=False
        )
        _plot_adaptive_interval_trend(
            axes[idx], pct, interval_df, f"{lang}", interval_x_axis=interval_trend_x_axis
        )

    fig.suptitle(
        "Poem perspective (post-hoc from pronouns) by language and adaptive interval",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out / "D_perspective_by_language_trend_by_interval.png", dpi=150, bbox_inches="tight")
    plt.close()


                                                                             
                                       
                                                                             

def write_extra_tables(df: pd.DataFrame, poems: pd.DataFrame, out: Path):
    core = df[df["period"].isin(PERIOD_DESCRIPTIVE_ORDER)].copy()

                                       
    sixway = core[core["person_number"].ne("")].copy()
    ct = pd.crosstab(sixway["person_number"], sixway["period"])
    for p in PERIOD_DESCRIPTIVE_ORDER:
        if p not in ct.columns:
            ct[p] = 0
    ct = ct[list(PERIOD_DESCRIPTIVE_ORDER)].reindex(SIXWAY_ORDER, fill_value=0)
    col_totals = ct.sum(axis=0).replace(0, np.nan)
    pct = ct.div(col_totals, axis=1) * 100
    combined = ct.copy()
    combined.columns = [f"{c}_n" for c in ct.columns]
    for c in pct.columns:
        combined[f"{c}_pct"] = pct[c]
        tot = float(col_totals[c]) if np.isfinite(col_totals[c]) else 0.0
        combined[f"{c}_pct_wilson_lb"] = pd.Series(
            [round(_wilson_lower_bound_pct(ct.loc[idx, c], tot), 2) for idx in combined.index],
            index=combined.index,
        )
    combined.to_csv(out / "F_person_number_crosstab_by_period.csv")

                                       
    rows = []
    for lang in MAJOR_LANGUAGES:
        ldf = core[core["language_clean"] == lang]
        for person in ["1st", "2nd", "3rd"]:
            for period in PERIOD_DESCRIPTIVE_ORDER:
                sub = ldf[(ldf["person"] == person) & (ldf["period"] == period)]
                period_total = len(ldf[ldf["period"] == period])
                n_cell = len(sub)
                p_raw = round(n_cell / period_total * 100, 2) if period_total else 0.0
                w_lb = round(_wilson_lower_bound_pct(n_cell, period_total), 2) if period_total else float("nan")
                rows.append({
                    "language": lang,
                    "person": person,
                    "period": period,
                    "n": n_cell,
                    "n_period": period_total,
                    "pct": p_raw,
                    "pct_wilson_lb": w_lb,
                })
    f2 = pd.DataFrame(rows)
    f2 = f2.sort_values(
        ["language", "period", "pct_wilson_lb", "person"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    f2.to_csv(out / "F_person_by_language_and_period.csv", index=False)

                                              
    pro_rows = []
    for lang in MAJOR_LANGUAGES:
        for period in PERIOD_DESCRIPTIVE_ORDER:
            sub = core[(core["language_clean"] == lang) & (core["period"] == period)]
            if sub.empty:
                continue
            is_pd = _pro_drop_mask(sub["is_pro_drop"])
            n_tot = len(sub)
            n_pd = int(is_pd.sum())
            pro_rows.append({
                "language": lang,
                "period": period,
                "n_total": n_tot,
                "n_pro_drop": n_pd,
                "pro_drop_rate_pct": round(n_pd / n_tot * 100, 2) if n_tot else 0.0,
                "pro_drop_rate_wilson_lb_pct": round(_wilson_lower_bound_pct(n_pd, n_tot), 2) if n_tot else float("nan"),
            })
    f3 = pd.DataFrame(pro_rows)
    f3 = f3.sort_values(
        ["period", "pro_drop_rate_wilson_lb_pct", "language"],
        ascending=[True, False, True],
        na_position="last",
    )
    f3.to_csv(out / "F_pro_drop_rate_by_language_period.csv", index=False)

                                                                    
    core_poems = poems[poems["period"].isin(PERIOD_DESCRIPTIVE_ORDER)].copy()
    persp_cats_f = _perspective_categories_for_display(core_poems["poem_perspective_primary"])
    persp_rows = []
    for lang in MAJOR_LANGUAGES:
        for period in PERIOD_DESCRIPTIVE_ORDER:
            sub = core_poems[(core_poems["language_clean"] == lang) & (core_poems["period"] == period)]
            total = len(sub)
            for persp in persp_cats_f:
                n = int((sub["poem_perspective_primary"] == persp).sum())
                p_raw = round(n / total * 100, 2) if total else 0.0
                w_lb = round(_wilson_lower_bound_pct(n, total), 2) if total else float("nan")
                persp_rows.append({
                    "language": lang,
                    "period": period,
                    "poem_perspective_primary": persp,
                    "n": n,
                    "n_poems_period": total,
                    "pct": p_raw,
                    "pct_wilson_lb": w_lb,
                })
    f4 = pd.DataFrame(persp_rows)
    f4 = f4.sort_values(
        ["language", "period", "pct_wilson_lb", "poem_perspective_primary"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    f4.to_csv(out / "F_poem_perspective_by_language_period.csv", index=False)

                                            
    stanza_rows = []
    for period in PERIOD_DESCRIPTIVE_ORDER:
        sub = poems[poems["period"] == period]
        if sub.empty:
            continue
        stanza_rows.append({
            "period": period,
            "n_poems": len(sub),
            "mean_stanzas": round(sub["n_stanzas"].mean(), 2),
            "median_stanzas": round(sub["n_stanzas"].median(), 2),
            "max_stanzas": int(sub["n_stanzas"].max()),
        })
    pd.DataFrame(stanza_rows).to_csv(out / "F_stanza_length_by_period.csv", index=False)


                                                                             
      
                                                                             

def main():
    parser = argparse.ArgumentParser(
        description="Descriptive statistics on stanza-level pronoun annotation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Stanza-level pronoun_annotation.csv (default: Annotated_GPT_rerun)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory")
    parser.add_argument(
        "--layer0",
        type=Path,
        default=DEFAULT_LAYER0,
        help="layer0_poems_one_per_row.csv — used only if input lacks is_repeat/is_translation",
    )
    parser.add_argument(
        "--no-repeat-translation-filter",
        action="store_true",
        help="Keep repeat and translation poems (not recommended for inference).",
    )
    parser.add_argument(
        "--min-poems-per-interval",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Temporal bins: minimum distinct poems per interval after merges "
            f"(default: {MIN_POEMS_PER_INTERVAL}; used by balanced and min_calendar)."
        ),
    )
    parser.add_argument(
        "--initial-months",
        type=int,
        default=None,
        metavar="M",
        help=(
            "min_calendar mode only: initial multi-month period width "
            f"(default: {INITIAL_MONTHS})."
        ),
    )
    parser.add_argument(
        "--temporal-binning",
        type=str,
        default="balanced",
        choices=("balanced", "min_calendar"),
        help=(
            "balanced = similar poem (or stanza) counts per chronological bin; "
            "min_calendar = legacy sparse calendar buckets merged to min poems."
        ),
    )
    parser.add_argument(
        "--target-poems-per-interval",
        type=int,
        default=None,
        metavar="N",
        help=(
            "balanced mode only: target poems per bin (sets number of bins ≈ "
            f"ceil(n_poems/N); default {TARGET_POEMS_PER_BALANCED_BIN})."
        ),
    )
    parser.add_argument(
        "--balance-by",
        type=str,
        default="poems",
        choices=("poems", "stanzas"),
        help=(
            "balanced mode: equalize bin sizes by distinct poems or by stanza count "
            "(stanzas requires stanza_index on input)."
        ),
    )
    parser.add_argument(
        "--interval-trend-x-axis",
        type=str,
        default="ordinal",
        choices=("ordinal", "density"),
        help=(
            "B/C/D interval line charts: ordinal = equal width per bin (default, not calendar); "
            "density = horizontal width ∝ poems per calendar day in bin."
        ),
    )
    parser.add_argument(
        "--author-bridge-min-poems-per-period",
        type=int,
        default=MIN_POEMS_EACH_PERIOD_FOR_AUTHOR_PRE_POST,
        metavar="N",
        help=(
            "author_pre_post_2022 plots: require at least N distinct poems in pre_2022 "
            f"and in post_2022 (default {MIN_POEMS_EACH_PERIOD_FOR_AUTHOR_PRE_POST}, i.e. >5 per era)."
        ),
    )
    parser.add_argument(
        "--min-pronouns-for-inference",
        type=int,
        default=MIN_PRONOUNS_FOR_INFERENCE,
        metavar="N",
        help=(
            "pre_post_2022_inference: include poems only if they have at least N pronoun tokens "
            f"(default {MIN_PRONOUNS_FOR_INFERENCE})."
        ),
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} …")
    df = load_stanza_data(input_path)
    excl_diag: dict = {}
    if args.no_repeat_translation_filter:
        write_exclusion_diagnostics(
            {"filter_applied": False, "note": "CLI --no-repeat-translation-filter"},
            out,
        )
    else:
        layer0_path = args.layer0.resolve() if args.layer0 else None
        df, excl_diag = attach_repeat_translation_and_filter(df, layer0_path)
        write_exclusion_diagnostics({**excl_diag, "filter_applied": True}, out)
        if excl_diag.get("repeat_translation_source") == "none_assumed_false":
            print(
                "  Warning: no is_repeat/is_translation in input and layer0 missing; "
                f"repeat/translation not filtered. Expected layer0: {DEFAULT_LAYER0}",
                file=sys.stderr,
            )

    df, interval_df = attach_adaptive_intervals(
        df,
        min_poems=args.min_poems_per_interval,
        initial_months=args.initial_months,
        temporal_binning=args.temporal_binning,
        target_poems_per_bin=args.target_poems_per_interval,
        balance_by=args.balance_by,
    )
    if not interval_df.empty:
        _iv_pub = interval_df[
            [c for c in ("interval_id", "interval_label", "n_poems", "start_date", "end_date", "n_stanzas_bin_definition") if c in interval_df.columns]
        ]
        _iv_pub.to_csv(out / "adaptive_intervals.csv", index=False)
        if args.temporal_binning == "balanced":
            print(
                f"  Balanced temporal bins: {len(interval_df)} intervals "
                f"(target_poems≈{args.target_poems_per_interval or TARGET_POEMS_PER_BALANCED_BIN}, "
                f"balance_by={args.balance_by}, min_poems="
                f"{args.min_poems_per_interval or MIN_POEMS_PER_INTERVAL})."
            )
        else:
            print(
                f"  min_calendar temporal bins: {len(interval_df)} intervals "
                f"(min_poems={args.min_poems_per_interval or MIN_POEMS_PER_INTERVAL}, "
                f"initial_months={args.initial_months or INITIAL_MONTHS})."
            )
    else:
        print(
            "  Warning: adaptive binning produced no intervals (no poem dates); "
            "B/C/D trend outputs will be skipped.",
            file=sys.stderr,
        )

    for stale in (
        "A_corpus_by_year.csv",
        "B_stanza_unit_n_per_year.csv",
        "B_pronoun_person_number_trend_by_year.csv",
        "B_pronoun_number_trend_by_year.csv",
        "B_pronoun_person_trend_by_year.csv",
        "B_pronoun_trend_by_year.png",
        "C_poem_perspective_trend_by_year.csv",
        "C_poem_perspective_trend_by_year.png",
        "D_perspective_by_language_trend.png",
        "D_perspective_ukrainian_trend_by_year.csv",
        "D_perspective_russian_trend_by_year.csv",
        "D_perspective_qirimli_trend_by_year.csv",
    ):
        p = out / stale
        if p.is_file():
            p.unlink()

    poems = build_poem_table_with_perspective(df)
    print(f"  {len(df):,} rows, {df['poem_id'].nunique():,} poems (after exclusions).")

    if not poems.empty:
        poems.to_csv(out / "C_poem_perspective_derived_per_poem.csv", index=False, encoding="utf-8-sig")
        print(f"  Wrote per-poem post-hoc perspective: {out / 'C_poem_perspective_derived_per_poem.csv'}")

    print("Write: A – corpus overview …")
    write_corpus_overview(df, poems, out)

    print("Write: B – pronoun trend charts …")
    write_pronoun_trend(df, out, interval_df, interval_trend_x_axis=args.interval_trend_x_axis)

    print("Write: C – poem perspective trend …")
    write_perspective_trend(poems, out, interval_df, interval_trend_x_axis=args.interval_trend_x_axis)

    print("Write: D – perspective by language …")
    write_perspective_by_language(
        poems, out, interval_df, interval_trend_x_axis=args.interval_trend_x_axis
    )

    print("Write: E – author summaries & pre/post bridge plots …")
    for stale in (
        "E_significance_pre_post_2022.csv",
        "E_significance_by_language.csv",
        "E_poem_proportion_mann_whitney_overall.csv",
        "E_poem_proportion_mann_whitney_by_language.csv",
        "E_poem_proportion_kruskal_wallis.csv",
        "E_poem_proportion_pairwise_mw.csv",
        "E_poem_proportion_mann_whitney.csv",
        "E_author_sensitivity_mann_whitney.csv",
        "E_poem_pronoun_proportions_per_poem.csv",
        "E_corpus_exclusions_repeat_translation.csv",
    ):
        sp = out / stale
        if sp.is_file():
            sp.unlink()

    poem_props = build_poems_with_pronoun_proportions(df)
    if not poem_props.empty:
        write_author_concentration(poem_props, out)
        n_bridge = write_author_pre_post_bridge_plots(
            df,
            poem_props,
            out,
            min_poems_each_period=args.author_bridge_min_poems_per_period,
        )
        print(f"  Wrote {out / 'E_top_authors_by_poem_share.csv'} ({n_bridge} author pre/post figure(s) under author_pre_post_2022/).")

    print("Writing pre/post 2022 pronoun inference (Mann–Whitney, Cohen's d, BH-FDR) …")
    write_pre_post_2022_pronoun_inference(
        df,
        poem_props,
        out,
        min_n_pronouns=args.min_pronouns_for_inference,
    )
    print(f"  Wrote tables under {out / 'pre_post_2022_inference'}/.")

    print("Write: F – supplementary tables …")
    write_extra_tables(df, poems, out)

    print(f"Done. All outputs written to: {out}")


if __name__ == "__main__":
    main()
