"""Three-period confirmatory contrasts with author FE and sensitivity battery."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "16_three_period_models"
DEFAULT_ROSTER = ROOT / "outputs" / "15_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_ROSTER_GE8 = ROOT / "outputs" / "15_roster_freeze" / "roster_v1_threshold_ge8.csv"

PERIODS = ["P1_2014_18", "P2_2019_21", "P3_2022plus"]
CELL4 = ["1sg", "1pl", "2sg", "2pl"]
CELL6 = ["1sg", "1pl", "2sg", "2pl", "3sg", "3pl"]
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}
SWITCHERS = {"Iya Kiva", "Andrij Bondar", "Alex Averbuch", "Olena Boryshpolets"}
FORMULA = 'y ~ person * number * C(period3, Treatment("P2_2019_21")) + C(author)'


def _period_three_way(y) -> str:
    if pd.isna(y):
        return "unknown"
    yy = int(y)
    if 2014 <= yy <= 2018:
        return "P1_2014_18"
    if 2019 <= yy <= 2021:
        return "P2_2019_21"
    if yy >= 2022:
        return "P3_2022plus"
    return "unknown"


def _normalize_bool_flag(series: pd.Series) -> pd.Series:
    if getattr(series.dtype, "name", "") == "bool":
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


def _sixway(row: pd.Series) -> str:
    pmap = {"1st": "1", "2nd": "2", "3rd": "3"}
    nmap = {"Singular": "sg", "Plural": "pl"}
    p = str(row.get("person", "")).strip()
    n = str(row.get("number", "")).strip()
    if p in pmap and n in nmap:
        return f"{pmap[p]}{nmap[n]}"
    return ""


def _mode_with_tie_order(series: pd.Series, preference: list[str]) -> str:
    vc = series.dropna().astype(str).str.strip()
    vc = vc[vc.ne("")]
    if vc.empty:
        return ""
    counts = vc.value_counts()
    top = int(counts.iloc[0])
    tied = counts[counts.eq(top)].index.tolist()
    if len(tied) == 1:
        return str(tied[0])
    for p in preference:
        if p in tied:
            return p
    return str(tied[0])


def bh_adjust(pvals: pd.Series) -> pd.Series:
    vals = pvals.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return pd.Series(out, index=pvals.index)
    pv = vals[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q = ranked * m / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[np.where(mask)[0]] = restored
    return pd.Series(out, index=pvals.index)


def load_and_filter(path: Path, layer0_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period3"] = df["year_int"].map(_period_three_way)
    df["person"] = df["person"].fillna("").str.strip()
    df["number"] = df["number"].fillna("").str.strip()
    df["person_number"] = df.apply(_sixway, axis=1)
    df["language_clean"] = df["language"].fillna("").str.strip()
    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = _normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = _normalize_bool_flag(df["is_translation"])
    elif layer0_path is not None and layer0_path.is_file():
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
    else:
        df = df.assign(is_repeat=False, is_translation=False)
    out = df.loc[~(df["is_repeat"] | df["is_translation"])].copy()
    out = out[~out["language_clean"].isin(QIRIMLI_CODES)].copy()
    return out


def build_poem_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    tok = df[df["pronoun_word"].notna()].copy()
    if tok.empty:
        return pd.DataFrame()
    ct = pd.crosstab(tok["poem_id"], tok["person_number"])
    for c in CELL6:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[CELL6].reset_index()
    ct["n_all6"] = ct[CELL6].sum(axis=1)
    ct["n_12"] = ct[CELL4].sum(axis=1)
    meta = df.groupby("poem_id", as_index=False).agg(
        author=("author", "first"),
        language_clean=("language_clean", "first"),
        year_int=("year_int", "first"),
    )
    out = ct.merge(meta, on="poem_id", how="left")
    out["period3"] = out["year_int"].map(_period_three_way)
    return out


def load_roster_authors(roster_path: Path) -> list[str]:
    r = pd.read_csv(roster_path, low_memory=False)
    return sorted(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def build_poem_long_4cells(poem_cell: pd.DataFrame, min_n_12: int, roster_authors: list[str]) -> pd.DataFrame:
    core = poem_cell[
        poem_cell["period3"].isin(PERIODS)
        & (poem_cell["n_12"] >= int(min_n_12))
        & poem_cell["author"].isin(roster_authors)
    ].copy()
    if core.empty:
        return pd.DataFrame()
    core["period3"] = pd.Categorical(core["period3"], categories=PERIODS, ordered=True)
    rows: list[dict] = []
    for _, r in core.iterrows():
        denom = int(r["n_12"])
        if denom <= 0:
            continue
        for cell in CELL4:
            person = 1 if cell.startswith("1") else 0
            number = 1 if cell.endswith("pl") else 0
            k = int(r[cell])
            rows.append(
                {
                    "poem_id": r["poem_id"],
                    "author": r.get("author", ""),
                    "language_clean": r.get("language_clean", ""),
                    "year_int": r.get("year_int", np.nan),
                    "period3": r["period3"],
                    "cell": cell,
                    "person": person,
                    "number": number,
                    "k": k,
                    "n": denom,
                    "y": k / denom,
                }
            )
    return pd.DataFrame(rows)


def _build_contrast_specs(names: list[str]) -> list[tuple[str, np.ndarray]]:
    vec = {n: i for i, n in enumerate(names)}

    def _term(period: str, suffix: str = "") -> str:
        base = f'C(period3, Treatment("P2_2019_21"))[T.{period}]'
        return base if not suffix else f"{suffix}:{base}"

    def _v(weights: dict[str, float]) -> np.ndarray:
        v = np.zeros(len(names), dtype=float)
        for k, w in weights.items():
            if k in vec:
                v[vec[k]] = float(w)
        return v

    tests = [
        ("P3_vs_P2_2sg_cell_shift", _v({_term("P3_2022plus"): 1.0, _term("P3_2022plus", "person"): 1.0})),
        ("P3_vs_P2_1pl_cell_shift", _v({_term("P3_2022plus"): 1.0, _term("P3_2022plus", "number"): 1.0})),
        ("P3_vs_P2_person_x_number", _v({_term("P3_2022plus", "person:number"): 1.0})),
        ("P1_vs_P2_2sg_cell_shift", _v({_term("P1_2014_18"): 1.0, _term("P1_2014_18", "person"): 1.0})),
        ("P1_vs_P2_1pl_cell_shift", _v({_term("P1_2014_18"): 1.0, _term("P1_2014_18", "number"): 1.0})),
        ("P1_vs_P2_person_x_number", _v({_term("P1_2014_18", "person:number"): 1.0})),
    ]
    return tests

def fit_glm(long_df: pd.DataFrame):
    return smf.glm(FORMULA, data=long_df, family=sm.families.Binomial(), freq_weights=long_df["n"]).fit()


def evaluate_contrasts(fit, long_df: pd.DataFrame) -> pd.DataFrame:
    names = fit.params.index.tolist()
    tests = _build_contrast_specs(names)
    cov = fit.cov_params().to_numpy(dtype=float)
    beta = fit.params.to_numpy(dtype=float)
    rows: list[dict] = []
    for label, v in tests:
        est = float(np.dot(v, beta))
        se = float(np.sqrt(max(0.0, np.dot(v, np.dot(cov, v)))))
        z = est / se if se > 0 else np.nan
        p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "contrast": label,
                "estimate_logit": est,
                "se": se,
                "z_value": z,
                "p_value": p,
                "ci95_low": est - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": est + 1.96 * se if np.isfinite(se) else np.nan,
                "odds_ratio": float(np.exp(est)),
                "n_poems": int(long_df["poem_id"].nunique()),
                "n_rows_long": int(len(long_df)),
            }
        )
    out = pd.DataFrame(rows)
    out["q_value_bh_confirmatory_family"] = bh_adjust(out["p_value"])
    return out


def wild_cluster_bootstrap(long_df: pd.DataFrame, fit, b_reps: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    names = fit.params.index.tolist()
    tests = _build_contrast_specs(names)
    obs = evaluate_contrasts(fit, long_df).set_index("contrast")
    resid = long_df["y"].to_numpy(dtype=float) - fit.fittedvalues.to_numpy(dtype=float)
    mu = fit.fittedvalues.to_numpy(dtype=float)
    authors = long_df["author"].astype(str).to_numpy()
    uniq = sorted(np.unique(authors).tolist())

    extreme = {name: 0 for name, _ in tests}
    valid = {name: 0 for name, _ in tests}
    t_obs = {name: float(obs.loc[name, "z_value"]) for name, _ in tests}
    start = time.time()

    for _ in range(int(b_reps)):
        w = {a: rng.choice([-1.0, 1.0]) for a in uniq}
        signs = np.array([w[a] for a in authors], dtype=float)
        y_star = np.clip(mu + signs * resid, 1e-5, 1 - 1e-5)
        bdat = long_df.copy()
        bdat["y"] = y_star
        try:
            bfit = fit_glm(bdat)
        except Exception:
            continue
        bnames = bfit.params.index.tolist()
        if bnames != names:
            continue
        bcov = bfit.cov_params().to_numpy(dtype=float)
        bbeta = bfit.params.to_numpy(dtype=float)
        for cname, v in tests:
            best = float(np.dot(v, bbeta))
            bse = float(np.sqrt(max(0.0, np.dot(v, np.dot(bcov, v)))))
            bt = best / bse if bse > 0 else np.nan
            if np.isfinite(bt) and np.isfinite(t_obs[cname]):
                valid[cname] += 1
                if abs(bt) >= abs(t_obs[cname]):
                    extreme[cname] += 1

    pvals = {
        name: (extreme[name] + 1.0) / (valid[name] + 1.0) if valid[name] > 0 else np.nan
        for name, _ in tests
    }
    elapsed = time.time() - start
    log = {"b_reps": int(b_reps), "seed": int(seed), "elapsed_sec": round(elapsed, 2), "valid_reps": valid}
    wdf = pd.DataFrame({"contrast": list(pvals.keys()), "p_value_wild_bootstrap": list(pvals.values())})
    return wdf, log


def run_model_variant(long_df: pd.DataFrame, variant: str, drop_author: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat = long_df.copy()
    if variant == "drop_2014":
        dat = dat[dat["year_int"] > 2014].copy()
    elif variant == "drop_switchers":
        dat = dat[~dat["author"].isin(SWITCHERS)].copy()
    elif variant == "leave_one_author_out" and drop_author:
        dat = dat[dat["author"] != drop_author].copy()
    if dat.empty:
        return pd.DataFrame(), pd.DataFrame()
    fit = fit_glm(dat)
    cdf = evaluate_contrasts(fit, dat)
    cdf["variant"] = variant
    cdf["dropped_author"] = drop_author or ""
    coef_df = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values})
    coef_df["variant"] = variant
    coef_df["dropped_author"] = drop_author or ""
    return cdf, coef_df


def build_per_author_contrasts(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for author, ad in long_df.groupby("author", sort=True):
        if ad["period3"].nunique() < 2:
            continue
        try:
            fit = smf.glm(
                'y ~ person * number * C(period3, Treatment("P2_2019_21"))',
                data=ad,
                family=sm.families.Binomial(),
                freq_weights=ad["n"],
            ).fit()
        except Exception:
            continue
        cdf = evaluate_contrasts(fit, ad)
        for cname in ("P3_vs_P2_1pl_cell_shift", "P3_vs_P2_2sg_cell_shift"):
            rr = cdf[cdf["contrast"].eq(cname)]
            if rr.empty:
                continue
            rows.append(
                {
                    "author": author,
                    "contrast": cname,
                    "estimate_logit": float(rr["estimate_logit"].iloc[0]),
                    "ci95_low": float(rr["ci95_low"].iloc[0]),
                    "ci95_high": float(rr["ci95_high"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_forest(main_df: pd.DataFrame, sens_df: pd.DataFrame, out_path: Path) -> None:
    order = main_df["contrast"].tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6))
    m = main_df.set_index("contrast").loc[order]
    ax.errorbar(m["estimate_logit"], y, xerr=[m["estimate_logit"] - m["ci95_low"], m["ci95_high"] - m["estimate_logit"]], fmt="o", color="black", label="main")
    for variant, g in sens_df[sens_df["variant"] != "leave_one_author_out"].groupby("variant"):
        gg = g.set_index("contrast").reindex(order)
        ax.scatter(gg["estimate_logit"], y, s=28, alpha=0.8, label=variant)
    lo = sens_df[sens_df["variant"] == "leave_one_author_out"]
    if not lo.empty:
        lo_agg = lo.groupby("contrast")["estimate_logit"].agg(["min", "max"]).reindex(order)
        ax.hlines(y, lo_agg["min"], lo_agg["max"], color="gray", linewidth=2, alpha=0.7, label="LOAO range")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Log-odds estimate")
    ax.set_title("Confirmatory contrasts with sensitivity overlays")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_caterpillar(per_author: pd.DataFrame, out_path: Path) -> None:
    if per_author.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    for i, cname in enumerate(["P3_vs_P2_2sg_cell_shift", "P3_vs_P2_1pl_cell_shift"]):
        ax = axes[i]
        d = per_author[per_author["contrast"] == cname].sort_values("estimate_logit")
        y = np.arange(len(d))
        ax.hlines(y, d["ci95_low"], d["ci95_high"], color="#4c78a8", alpha=0.8)
        ax.plot(d["estimate_logit"], y, "o", color="#4c78a8")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(d["author"])
        ax.set_title(cname)
        ax.set_xlabel("Log-odds")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-period confirmatory contrast models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--roster-ge8", type=Path, default=DEFAULT_ROSTER_GE8)
    parser.add_argument("--min-n12-per-poem", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=1999)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    parser.add_argument("--prereg-commit", type=str, default="")
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.input.resolve(), args.layer0.resolve() if args.layer0 else None)
    poem_cell = build_poem_cell_table(df)
    roster_main = load_roster_authors(args.roster.resolve())
    roster_ge8 = load_roster_authors(args.roster_ge8.resolve())

    long_main = build_poem_long_4cells(poem_cell, args.min_n12_per_poem, roster_main)
    long_ge8 = build_poem_long_4cells(poem_cell, args.min_n12_per_poem, roster_ge8)

    fit_main = fit_glm(long_main)
    confirm_main = evaluate_contrasts(fit_main, long_main)
    wb_df, wb_log = wild_cluster_bootstrap(long_main, fit_main, args.bootstrap_reps, args.bootstrap_seed)
    confirm_main = confirm_main.merge(wb_df, on="contrast", how="left")
    confirm_main["q_value_bh_wild_bootstrap_family"] = bh_adjust(confirm_main["p_value_wild_bootstrap"])
    confirm_main.to_csv(out / "confirmatory_contrasts_main.csv", index=False)
    pd.DataFrame({"term": fit_main.params.index, "coef": fit_main.params.values, "p_value": fit_main.pvalues.values}).to_csv(
        out / "unified_model_coefficients.csv", index=False
    )

    sens_frames: list[pd.DataFrame] = []
    for v in ("drop_2014", "drop_switchers"):
        cdf, _ = run_model_variant(long_main, variant=v)
        if not cdf.empty:
            sens_frames.append(cdf)
    cdf_ge8, _ = run_model_variant(long_ge8, variant="roster_ge8")
    if not cdf_ge8.empty:
        sens_frames.append(cdf_ge8)
    for a in sorted(set(long_main["author"].astype(str))):
        cdf_loo, _ = run_model_variant(long_main, variant="leave_one_author_out", drop_author=a)
        if not cdf_loo.empty:
            sens_frames.append(cdf_loo)
    sensitivity = pd.concat(sens_frames, ignore_index=True) if sens_frames else pd.DataFrame()
    sensitivity.to_csv(out / "confirmatory_contrasts_sensitivity.csv", index=False)

    per_author = build_per_author_contrasts(long_main)
    per_author.to_csv(out / "per_author_contrast_estimates.csv", index=False)

    plot_forest(confirm_main, sensitivity, fig_dir / "fig1_confirmatory_forest_with_sensitivity.pdf")
    plot_caterpillar(per_author, fig_dir / "fig2_per_author_caterpillar.pdf")

    with (out / "wild_bootstrap_log.txt").open("w", encoding="utf-8") as f:
        f.write(f"B={wb_log['b_reps']}\n")
        f.write(f"seed={wb_log['seed']}\n")
        f.write(f"elapsed_sec={wb_log['elapsed_sec']}\n")
        f.write(f"valid_reps={wb_log['valid_reps']}\n")

    with (out / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Three-period confirmatory model outputs\n\n")
        if args.prereg_commit:
            f.write(f"- Preregistration commit hash: `{args.prereg_commit}`\n")
        f.write("- Main family includes 6 confirmatory contrasts with BH correction.\n")
        f.write("- Wild cluster bootstrap uses Rademacher weights by author cluster.\n")

    print(f"Wrote three-period outputs to: {out}")


if __name__ == "__main__":
    main()

