"""Model-based significance analysis for pronoun change (pre vs post 2022)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.adaptive_temporal_binning import balanced_temporal_binning
from utils.pronoun_encoding import pronoun_class_sixway_column
from utils.stats_common import bh_adjust, mode_with_tie_order, normalize_bool_flag, period_pre_post_2022
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_significance_models"

PERIOD_PRE = "pre_2022"
PERIOD_POST = "post_2022"
MAJOR_LANGUAGES = ["Ukrainian", "Russian"]
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}
PROP_FEATURES = ["prop_1st", "prop_2nd", "prop_3rd", "prop_plural", "prop_pro_drop"]
SIXWAY_ORDER = ["1sg", "1pl", "2sg", "2pl", "3sg", "3pl"]


def _pro_drop_mask(series: pd.Series) -> pd.Series:
    if getattr(series.dtype, "name", "") == "bool":
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _mode_with_tie_order(series: pd.Series, preference: list[str]) -> str:
    return mode_with_tie_order(series, preference)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period"] = df["year_int"].map(period_pre_post_2022)
    df["person"] = df["person"].fillna("").str.strip()
    df["number"] = df["number"].fillna("").str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
    df["language_clean"] = df["language"].fillna("").str.strip()
    return df


def attach_repeat_translation_and_filter(
    df: pd.DataFrame,
    layer0_path: Path | None,
) -> pd.DataFrame:
    df = df.copy()
    date_posted_attached = False
    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = normalize_bool_flag(df["is_translation"])
    elif layer0_path is not None and layer0_path.is_file():
        l0 = pd.read_csv(
            layer0_path,
            usecols=[
                "poem_id",
                "Is repeat",
                "I.D. of original (if poem is a translation)",
                "Date posted",
            ],
            low_memory=False,
        )
        oid = l0["I.D. of original (if poem is a translation)"]
        flags = pd.DataFrame(
            {
                "poem_id": l0["poem_id"].astype(str).str.strip(),
                "is_repeat": l0["Is repeat"].astype(str).str.lower().str.strip().eq("yes"),
                "is_translation": oid.notna() & oid.astype(str).str.strip().ne(""),
                "date_posted": pd.to_datetime(l0["Date posted"], errors="coerce"),
            }
        ).drop_duplicates(subset=["poem_id"], keep="first")
        df = df.merge(flags, on="poem_id", how="left")
        df["is_repeat"] = df["is_repeat"].fillna(False).astype(bool)
        df["is_translation"] = df["is_translation"].fillna(False).astype(bool)
        date_posted_attached = True
    else:
        df = df.assign(is_repeat=False, is_translation=False)
    if not date_posted_attached and layer0_path is not None and layer0_path.is_file():
        l0 = pd.read_csv(
            layer0_path,
            usecols=["poem_id", "Date posted"],
            low_memory=False,
        )
        l0_dates = pd.DataFrame(
            {
                "poem_id": l0["poem_id"].astype(str).str.strip(),
                "date_posted": pd.to_datetime(l0["Date posted"], errors="coerce"),
            }
        ).drop_duplicates(subset=["poem_id"], keep="first")
        df = df.merge(l0_dates, on="poem_id", how="left")
    if "date_posted" not in df.columns:
        df["date_posted"] = pd.NaT
    out = df.loc[~(df["is_repeat"] | df["is_translation"])].copy()
    out = out[~out["language_clean"].isin(QIRIMLI_CODES)].copy()
    return out


def build_poem_props(df: pd.DataFrame) -> pd.DataFrame:
    pronouns = df[df["pronoun_word"].notna()].copy()
    if pronouns.empty:
        return pd.DataFrame(columns=["poem_id", "author", "language_clean", "period", "n_pronouns", *PROP_FEATURES])
    is_pd = _pro_drop_mask(pronouns["is_pro_drop"]).astype(np.int32)
    pronouns = pronouns.assign(
        _p1=(pronouns["person"] == "1st").astype(np.int32),
        _p2=(pronouns["person"] == "2nd").astype(np.int32),
        _p3=(pronouns["person"] == "3rd").astype(np.int32),
        _pl=(pronouns["number"] == "Plural").astype(np.int32),
        _pd=is_pd,
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

    meta_cols = {
        "author": ("author", "first"),
        "language_clean": ("language_clean", "first"),
        "year_int": ("year_int", "first"),
    }
    if "date_posted" in df.columns:
        meta_cols["date_posted"] = ("date_posted", "first")
    meta = df.groupby("poem_id", as_index=False).agg(**meta_cols)
    out = agg.merge(meta, on="poem_id", how="left")
    out["period"] = out["year_int"].map(period_pre_post_2022)
    cols = ["poem_id", "author", "language_clean", "year_int", "period", "n_pronouns", *PROP_FEATURES]
    if "date_posted" in out.columns:
        cols.append("date_posted")
    return out[cols]


def build_stanza_modal_number(df: pd.DataFrame) -> pd.DataFrame:
    if "stanza_index" not in df.columns:
        return pd.DataFrame(columns=["poem_id", "author", "language_clean", "period", "stanza_number_mode"])
    tok = df[df["pronoun_word"].notna() & df["stanza_index"].notna()].copy()
    if tok.empty:
        return pd.DataFrame(columns=["poem_id", "author", "language_clean", "period", "stanza_number_mode"])
    tok["stanza_index"] = pd.to_numeric(tok["stanza_index"], errors="coerce")
    tok = tok[tok["stanza_index"].notna()].copy()
    rows: list[dict] = []
    for (pid, sid), g in tok.groupby(["poem_id", "stanza_index"], sort=False):
        gnum = g[g["number"].isin(["Singular", "Plural"])]
        if gnum.empty:
            continue
        mode = _mode_with_tie_order(gnum["number"], ["Singular", "Plural"])
        rows.append(
            {
                "poem_id": pid,
                "stanza_index": int(sid),
                "stanza_number_mode": mode,
                "period": str(g["period"].iloc[0]),
                "author": g["author"].iloc[0] if "author" in g.columns else "",
                "language_clean": g["language_clean"].iloc[0] if "language_clean" in g.columns else "",
            }
        )
    return pd.DataFrame(rows)


def build_stanza_modal_pn(df: pd.DataFrame) -> pd.DataFrame:
    if "stanza_index" not in df.columns:
        return pd.DataFrame(columns=["poem_id", "author", "language_clean", "period", "stanza_pn_mode"])
    tok = df[df["pronoun_word"].notna() & df["stanza_index"].notna()].copy()
    if tok.empty:
        return pd.DataFrame(columns=["poem_id", "author", "language_clean", "period", "stanza_pn_mode"])
    tok["stanza_index"] = pd.to_numeric(tok["stanza_index"], errors="coerce")
    tok = tok[tok["stanza_index"].notna()].copy()
    rows: list[dict] = []
    for (pid, sid), g in tok.groupby(["poem_id", "stanza_index"], sort=False):
        gpn = g[g["person_number"].isin(SIXWAY_ORDER)]
        if gpn.empty:
            continue
        mode = _mode_with_tie_order(gpn["person_number"], SIXWAY_ORDER)
        rows.append(
            {
                "poem_id": pid,
                "stanza_index": int(sid),
                "stanza_pn_mode": mode,
                "period": str(g["period"].iloc[0]),
                "author": g["author"].iloc[0] if "author" in g.columns else "",
                "language_clean": g["language_clean"].iloc[0] if "language_clean" in g.columns else "",
            }
        )
    return pd.DataFrame(rows)


def _fit_ols_clustered(formula: str, data: pd.DataFrame, group_col: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    model = smf.ols(formula, data=data)
    n_groups = int(pd.Series(data[group_col]).nunique(dropna=True))
    if n_groups >= 2:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[group_col]})
    return model.fit(cov_type="HC3")


def _fit_glm_binomial_clustered(formula: str, data: pd.DataFrame, group_col: str):
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    n_groups = int(pd.Series(data[group_col]).nunique(dropna=True))
    if n_groups >= 2:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[group_col]})
    return model.fit(cov_type="HC3")


def poem_level_models(poem: pd.DataFrame, min_n_pronouns: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = poem[
        poem["period"].isin([PERIOD_PRE, PERIOD_POST]) & (poem["n_pronouns"] >= int(min_n_pronouns))
    ].copy()
    core["post_2022"] = (core["period"] == PERIOD_POST).astype(int)
    eps = 1e-4

    rows_overall: list[dict] = []
    rows_lang: list[dict] = []
    for feat in PROP_FEATURES:
        sub = core[["poem_id", "author", "language_clean", "post_2022", feat]].dropna().copy()
        if len(sub) < 50 or sub["post_2022"].nunique() < 2:
            continue
        sub["y_logit"] = np.log(np.clip(sub[feat].to_numpy(dtype=float), eps, 1.0 - eps) / np.clip(1.0 - sub[feat].to_numpy(dtype=float), eps, 1.0))
        fit = _fit_ols_clustered("y_logit ~ post_2022 + C(language_clean)", sub, "author")
        ci = fit.conf_int().loc["post_2022"]
        rows_overall.append(
            {
                "feature": feat,
                "n_rows": int(len(sub)),
                "coef_post_2022_logit": float(fit.params["post_2022"]),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "p_value": float(fit.pvalues["post_2022"]),
            }
        )

        for lang in MAJOR_LANGUAGES:
            ldf = sub[sub["language_clean"] == lang].copy()
            if len(ldf) < 30 or ldf["post_2022"].nunique() < 2:
                continue
            lfit = _fit_ols_clustered("y_logit ~ post_2022", ldf, "author")
            lci = lfit.conf_int().loc["post_2022"]
            rows_lang.append(
                {
                    "language": lang,
                    "feature": feat,
                    "n_rows": int(len(ldf)),
                    "coef_post_2022_logit": float(lfit.params["post_2022"]),
                    "ci95_low": float(lci.iloc[0]),
                    "ci95_high": float(lci.iloc[1]),
                    "p_value": float(lfit.pvalues["post_2022"]),
                }
            )

    overall_df = pd.DataFrame(rows_overall)
    if not overall_df.empty:
        overall_df["q_value_bh"] = bh_adjust(overall_df["p_value"])

    lang_df = pd.DataFrame(rows_lang)
    if not lang_df.empty:
        lang_df["q_value_bh_within_language"] = (
            lang_df.groupby("language", group_keys=False)["p_value"].apply(bh_adjust)
        )
    return overall_df, lang_df


def stanza_plural_models(stanza: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = stanza[
        stanza["period"].isin([PERIOD_PRE, PERIOD_POST]) & stanza["stanza_number_mode"].isin(["Singular", "Plural"])
    ].copy()
    core["post_2022"] = (core["period"] == PERIOD_POST).astype(int)
    core["y_plural"] = (core["stanza_number_mode"] == "Plural").astype(int)

    rows_overall: list[dict] = []
    rows_lang: list[dict] = []
    if len(core) >= 100 and core["post_2022"].nunique() == 2 and core["y_plural"].nunique() == 2:
        fit = _fit_glm_binomial_clustered(
            "y_plural ~ post_2022 + C(language_clean)",
            core,
            "poem_id",
        )
        ci = fit.conf_int().loc["post_2022"]
        rows_overall.append(
            {
                "outcome": "plural_mode",
                "n_rows": int(len(core)),
                "coef_post_2022_log_odds": float(fit.params["post_2022"]),
                "odds_ratio_post_2022": float(np.exp(fit.params["post_2022"])),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "or_ci95_low": float(np.exp(ci.iloc[0])),
                "or_ci95_high": float(np.exp(ci.iloc[1])),
                "p_value": float(fit.pvalues["post_2022"]),
            }
        )

    for lang in MAJOR_LANGUAGES:
        ldf = core[core["language_clean"] == lang].copy()
        if len(ldf) < 80 or ldf["post_2022"].nunique() < 2 or ldf["y_plural"].nunique() < 2:
            continue
        lfit = _fit_glm_binomial_clustered("y_plural ~ post_2022", ldf, "poem_id")
        lci = lfit.conf_int().loc["post_2022"]
        rows_lang.append(
            {
                "language": lang,
                "outcome": "plural_mode",
                "n_rows": int(len(ldf)),
                "coef_post_2022_log_odds": float(lfit.params["post_2022"]),
                "odds_ratio_post_2022": float(np.exp(lfit.params["post_2022"])),
                "ci95_low": float(lci.iloc[0]),
                "ci95_high": float(lci.iloc[1]),
                "or_ci95_low": float(np.exp(lci.iloc[0])),
                "or_ci95_high": float(np.exp(lci.iloc[1])),
                "p_value": float(lfit.pvalues["post_2022"]),
            }
        )

    overall_df = pd.DataFrame(rows_overall)
    if not overall_df.empty:
        overall_df["q_value_bh"] = bh_adjust(overall_df["p_value"])

    lang_df = pd.DataFrame(rows_lang)
    if not lang_df.empty:
        lang_df["q_value_bh"] = bh_adjust(lang_df["p_value"])
    return overall_df, lang_df


def stanza_pn_one_vs_rest_models(stanza_pn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = stanza_pn[stanza_pn["period"].isin([PERIOD_PRE, PERIOD_POST])].copy()
    core["post_2022"] = (core["period"] == PERIOD_POST).astype(int)
    core = core[core["stanza_pn_mode"].isin(SIXWAY_ORDER)].copy()

    rows_overall: list[dict] = []
    rows_lang: list[dict] = []
    for cat in SIXWAY_ORDER:
        cdf = core.copy()
        cdf["y_cat"] = (cdf["stanza_pn_mode"] == cat).astype(int)
        if len(cdf) < 100 or cdf["y_cat"].nunique() < 2 or cdf["post_2022"].nunique() < 2:
            continue
        fit = _fit_glm_binomial_clustered("y_cat ~ post_2022 + C(language_clean)", cdf, "poem_id")
        ci = fit.conf_int().loc["post_2022"]
        rows_overall.append(
            {
                "category": cat,
                "n_rows": int(len(cdf)),
                "coef_post_2022_log_odds": float(fit.params["post_2022"]),
                "odds_ratio_post_2022": float(np.exp(fit.params["post_2022"])),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "or_ci95_low": float(np.exp(ci.iloc[0])),
                "or_ci95_high": float(np.exp(ci.iloc[1])),
                "p_value": float(fit.pvalues["post_2022"]),
            }
        )
        for lang in MAJOR_LANGUAGES:
            ldf = cdf[cdf["language_clean"] == lang].copy()
            if len(ldf) < 80 or ldf["y_cat"].nunique() < 2 or ldf["post_2022"].nunique() < 2:
                continue
            lfit = _fit_glm_binomial_clustered("y_cat ~ post_2022", ldf, "poem_id")
            lci = lfit.conf_int().loc["post_2022"]
            rows_lang.append(
                {
                    "language": lang,
                    "category": cat,
                    "n_rows": int(len(ldf)),
                    "coef_post_2022_log_odds": float(lfit.params["post_2022"]),
                    "odds_ratio_post_2022": float(np.exp(lfit.params["post_2022"])),
                    "ci95_low": float(lci.iloc[0]),
                    "ci95_high": float(lci.iloc[1]),
                    "or_ci95_low": float(np.exp(lci.iloc[0])),
                    "or_ci95_high": float(np.exp(lci.iloc[1])),
                    "p_value": float(lfit.pvalues["post_2022"]),
                }
            )
    overall_df = pd.DataFrame(rows_overall)
    if not overall_df.empty:
        overall_df["q_value_bh"] = bh_adjust(overall_df["p_value"])
    lang_df = pd.DataFrame(rows_lang)
    if not lang_df.empty:
        lang_df["q_value_bh_within_language"] = (
            lang_df.groupby("language", group_keys=False)["p_value"].apply(bh_adjust)
        )
    return overall_df, lang_df


EUROMAIDAN_BREAK = pd.Timestamp("2014-02-20")
FULL_SCALE_INVASION_BREAK = pd.Timestamp("2022-02-24")
TARGET_POEMS_PER_ITS_BIN = 50


def poem_level_segmented_time_models(poem: pd.DataFrame, min_n_pronouns: int) -> pd.DataFrame:
    """Segmented (interrupted) time-series WLS on ~35 balanced poem-count bins.

    Replaces the prior year-aggregated version (T=13, k=5 → df_resid=7) with
    balanced bins of ≈50 poems each via
    :func:`utils.adaptive_temporal_binning.balanced_temporal_binning`, which
    yields ≈ N_poems/50 bins (≈35 on the current corpus). With 5 ITS regressors
    plus the intercept, residual df ≥ 30, restoring statistical power for the
    Hahn–McKnight–Sloan style breaks at the Euromaidan (2014-02-20) and the
    full-scale invasion (2022-02-24).
    """
    core = poem[
        poem["period"].isin([PERIOD_PRE, PERIOD_POST]) & (poem["n_pronouns"] >= int(min_n_pronouns))
    ].copy()
    if "date_posted" not in core.columns:
        core["date_posted"] = pd.NaT
    core["date_posted"] = pd.to_datetime(core["date_posted"], errors="coerce")
    if core["date_posted"].notna().sum() < 100:
        return pd.DataFrame()
    core = core[core["date_posted"].notna()].copy()
    if core.empty:
        return pd.DataFrame()
    eps = 1e-4
    rows: list[dict] = []
    for feat in PROP_FEATURES:
        sub = core[["poem_id", "date_posted", feat]].dropna().copy()
        if len(sub) < 100:
            continue
        sub = sub.rename(columns={"date_posted": "date"})
        binned, intervals = balanced_temporal_binning(
            sub,
            date_col="date",
            id_col="poem_id",
            target_poems_per_bin=TARGET_POEMS_PER_ITS_BIN,
        )
        if binned.empty or len(intervals) < 8:
            continue
        by_bin = binned.groupby("interval_id", as_index=False).agg(
            y_mean=(feat, "mean"),
            n_obs=(feat, "size"),
            start_date=("_date", "min"),
        )
        by_bin = by_bin.sort_values("start_date").reset_index(drop=True)
        by_bin["y_logit"] = np.log(
            np.clip(by_bin["y_mean"].to_numpy(dtype=float), eps, 1.0 - eps)
            / np.clip(1.0 - by_bin["y_mean"].to_numpy(dtype=float), eps, 1.0)
        )
        t0 = by_bin["start_date"].min()
        by_bin["t"] = (by_bin["start_date"] - t0).dt.days.astype(float) / 365.25
        by_bin["post_2014"] = (by_bin["start_date"] >= EUROMAIDAN_BREAK).astype(int)
        by_bin["post_2022"] = (by_bin["start_date"] >= FULL_SCALE_INVASION_BREAK).astype(int)
        by_bin["t_after_2014"] = np.where(
            by_bin["start_date"] >= EUROMAIDAN_BREAK,
            (by_bin["start_date"] - EUROMAIDAN_BREAK).dt.days.astype(float) / 365.25,
            0.0,
        )
        by_bin["t_after_2022"] = np.where(
            by_bin["start_date"] >= FULL_SCALE_INVASION_BREAK,
            (by_bin["start_date"] - FULL_SCALE_INVASION_BREAK).dt.days.astype(float) / 365.25,
            0.0,
        )
        fit = smf.wls(
            "y_logit ~ t + post_2014 + t_after_2014 + post_2022 + t_after_2022",
            data=by_bin,
            weights=by_bin["n_obs"],
        ).fit(cov_type="HC3")
        for term in ("post_2014", "t_after_2014", "post_2022", "t_after_2022"):
            if term not in fit.params.index:
                continue
            ci = fit.conf_int().loc[term]
            rows.append(
                {
                    "feature": feat,
                    "term": term,
                    "n_rows": int(len(sub)),
                    "n_bins": int(len(by_bin)),
                    "df_resid": float(fit.df_resid),
                    "time_axis": "balanced_bins",
                    "target_poems_per_bin": int(TARGET_POEMS_PER_ITS_BIN),
                    "coef": float(fit.params[term]),
                    "ci95_low": float(ci.iloc[0]),
                    "ci95_high": float(ci.iloc[1]),
                    "p_value": float(fit.pvalues[term]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_bh_within_term"] = out.groupby("term", group_keys=False)["p_value"].apply(bh_adjust)
    return out


def poem_level_loess_descriptive(
    poem: pd.DataFrame,
    min_n_pronouns: int,
    *,
    frac: float = 0.3,
    n_bootstrap: int = 200,
    seed: int = 20260511,
) -> pd.DataFrame:
    """LOESS smooth of each poem-level proportion with author-cluster bootstrap bands.

    Descriptive companion to :func:`poem_level_segmented_time_models`. Produced
    purely for visualization in stage 02e; **not** used for inference. The 95%
    band is a 200-replicate author-cluster bootstrap (Rademacher resample of
    authors), evaluated at the original LOESS grid.
    """
    core = poem[
        poem["period"].isin([PERIOD_PRE, PERIOD_POST]) & (poem["n_pronouns"] >= int(min_n_pronouns))
    ].copy()
    if "date_posted" not in core.columns:
        core["date_posted"] = pd.NaT
    core["date_posted"] = pd.to_datetime(core["date_posted"], errors="coerce")
    core = core[core["date_posted"].notna()].copy()
    if core.empty:
        return pd.DataFrame()
    t0 = core["date_posted"].min()
    core["t_days"] = (core["date_posted"] - t0).dt.days.astype(float)
    rng = np.random.default_rng(int(seed))
    rows: list[dict] = []
    for feat in PROP_FEATURES:
        sub = core[["poem_id", "author", "t_days", feat]].dropna().copy()
        if len(sub) < 100:
            continue
        x = sub["t_days"].to_numpy(dtype=float)
        y = sub[feat].to_numpy(dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        smoothed = lowess(y_sorted, x_sorted, frac=float(frac), it=2, return_sorted=True)
        grid_x = smoothed[:, 0]
        grid_y = smoothed[:, 1]
        if n_bootstrap > 0:
            authors = sub["author"].astype(str).to_numpy()
            uniq_authors = np.unique(authors)
            n_grid = len(grid_x)
            boot_curves = np.full((int(n_bootstrap), n_grid), np.nan, dtype=float)
            for b in range(int(n_bootstrap)):
                sample_authors = rng.choice(uniq_authors, size=len(uniq_authors), replace=True)
                pieces_x: list[np.ndarray] = []
                pieces_y: list[np.ndarray] = []
                for a in sample_authors:
                    mask = authors == a
                    pieces_x.append(x[mask])
                    pieces_y.append(y[mask])
                if not pieces_x:
                    continue
                bx = np.concatenate(pieces_x)
                by = np.concatenate(pieces_y)
                if len(bx) < 30:
                    continue
                order_b = np.argsort(bx)
                try:
                    bs = lowess(
                        by[order_b],
                        bx[order_b],
                        frac=float(frac),
                        it=1,
                        xvals=grid_x,
                    )
                except Exception:
                    continue
                boot_curves[b, :] = bs
            with np.errstate(invalid="ignore"):
                ci_low = np.nanpercentile(boot_curves, 2.5, axis=0)
                ci_high = np.nanpercentile(boot_curves, 97.5, axis=0)
        else:
            ci_low = np.full_like(grid_y, np.nan)
            ci_high = np.full_like(grid_y, np.nan)
        for i in range(len(grid_x)):
            rows.append(
                {
                    "feature": feat,
                    "t_days": float(grid_x[i]),
                    "date_origin": t0.isoformat(),
                    "y_smooth": float(grid_y[i]),
                    "ci95_low": float(ci_low[i]) if np.isfinite(ci_low[i]) else np.nan,
                    "ci95_high": float(ci_high[i]) if np.isfinite(ci_high[i]) else np.nan,
                    "n_obs": int(len(sub)),
                    "frac": float(frac),
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-based significance analysis for pronoun change.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--min-pronouns-for-poem-model", type=int, default=5)
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input.resolve())
    df = attach_repeat_translation_and_filter(df, args.layer0.resolve() if args.layer0 else None)

    poem = build_poem_props(df)
    stanza = build_stanza_modal_number(df)
    stanza_pn = build_stanza_modal_pn(df)

    poem_overall, poem_lang = poem_level_models(poem, args.min_pronouns_for_poem_model)
    st_overall, st_lang = stanza_plural_models(stanza)
    pn_overall, pn_lang = stanza_pn_one_vs_rest_models(stanza_pn)
    seg_poem = poem_level_segmented_time_models(poem, args.min_pronouns_for_poem_model)
    loess_descr = poem_level_loess_descriptive(poem, args.min_pronouns_for_poem_model)

    poem_overall.to_csv(out / "poem_level_logit_ols.csv", index=False)
    poem_lang.to_csv(out / "poem_level_logit_ols_by_language.csv", index=False)
    st_overall.to_csv(out / "stanza_plural_glm.csv", index=False)
    st_lang.to_csv(out / "stanza_plural_glm_by_language.csv", index=False)
    pn_overall.to_csv(out / "stanza_pn_one_vs_rest_glm.csv", index=False)
    pn_lang.to_csv(out / "stanza_pn_one_vs_rest_glm_by_language.csv", index=False)
    seg_poem.to_csv(out / "poem_level_segmented_2014_2022.csv", index=False)
    loess_descr.to_csv(out / "poem_level_loess_descriptive.csv", index=False)

    print(f"Wrote model outputs to: {out}")


if __name__ == "__main__":
    main()
