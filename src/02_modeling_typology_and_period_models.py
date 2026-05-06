"""Run cohort model, typology classification, and period visuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils.pronoun_encoding import pronoun_class_sixway_column
from utils.stats_common import normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUT = ROOT / "outputs" / "02_modeling_typology_and_period_models"

PERIODS = ["P1_2014_2021", "P2_2022_plus"]
CELL4 = ["1sg", "1pl", "2sg", "2pl"]
SWITCHERS = {"Iya Kiva", "Andrij Bondar", "Alex Averbuch", "Olena Boryshpolets"}


def load_and_filter(path: Path, layer0_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df["period3"] = df["year_int"].map(period_three_way)
    df["person"] = df["person"].fillna("").str.strip()
    df["number"] = df["number"].fillna("").str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = normalize_bool_flag(df["is_translation"])
    else:
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
    return df.loc[~(df["is_repeat"] | df["is_translation"])].copy()


def load_roster(roster_path: Path) -> list[str]:
    r = pd.read_csv(roster_path, low_memory=False)
    return sorted(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def build_long(df: pd.DataFrame, roster_authors: list[str], min_n12: int = 5) -> pd.DataFrame:
    tok = df[df["pronoun_word"].notna() & df["person_number"].isin(CELL4)].copy()
    tok = tok[tok["period3"].isin(PERIODS)].copy()
    if tok.empty:
        return pd.DataFrame()
    ct = pd.crosstab(tok["poem_id"], tok["person_number"])
    for c in CELL4:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[CELL4].reset_index()
    ct["n_12"] = ct[CELL4].sum(axis=1)
    meta = tok.groupby("poem_id", as_index=False).agg(author=("author", "first"), year_int=("year_int", "first"), period3=("period3", "first"))
    poem = ct.merge(meta, on="poem_id", how="left")
    poem = poem[(poem["author"].isin(roster_authors)) & (poem["n_12"] >= int(min_n12))].copy()
    rows: list[dict] = []
    for _, r in poem.iterrows():
        denom = float(r["n_12"])
        for cell in CELL4:
            person = 1 if cell.startswith("1") else 0
            number = 1 if cell.endswith("pl") else 0
            k = float(r[cell])
            rows.append(
                {
                    "poem_id": r["poem_id"],
                    "author": r["author"],
                    "year_int": float(r["year_int"]),
                    "period3": r["period3"],
                    "cell": cell,
                    "person": person,
                    "number": number,
                    "k": k,
                    "n": denom,
                    "y": k / denom if denom > 0 else np.nan,
                    "log_n_tokens": float(np.log(max(1.0, denom))),
                }
            )
    out = pd.DataFrame(rows)
    out["period3"] = pd.Categorical(out["period3"], categories=PERIODS, ordered=True)
    return out


def fit_main(long_df: pd.DataFrame):
    formula = 'y ~ person * number * C(period3, Treatment("P1_2014_2021")) + C(author) * person * number + log_n_tokens'
    fit = smf.glm(formula, data=long_df, family=sm.families.Binomial(), freq_weights=long_df["n"]).fit()
    return fit


def per_author_contrasts(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for author, g in long_df.groupby("author", sort=True):
        if g["period3"].nunique() < 2:
            continue
        model = smf.glm(
            'y ~ person * number * C(period3, Treatment("P1_2014_2021")) + log_n_tokens',
            data=g,
            family=sm.families.Binomial(),
            freq_weights=g["n"],
        )
        fit = model.fit_regularized(alpha=1e-4, L1_wt=0.0)
        p = fit.params.to_dict()
        t_p2 = 'C(period3, Treatment("P1_2014_2021"))[T.P2_2022_plus]'
        e_2sg = p.get(t_p2, 0.0) + p.get(f"person:{t_p2}", 0.0)
        e_1pl = p.get(t_p2, 0.0) + p.get(f"number:{t_p2}", 0.0)
        rows.append({"author": author, "contrast": "P2_vs_P1_2sg_cell_shift", "estimate_logit": e_2sg})
        rows.append({"author": author, "contrast": "P2_vs_P1_1pl_cell_shift", "estimate_logit": e_1pl})
    return pd.DataFrame(rows)


def classify_typology(per_author: pd.DataFrame) -> pd.DataFrame:
    piv = per_author.pivot(index="author", columns="contrast", values="estimate_logit").reset_index()
    for c in ["P2_vs_P1_2sg_cell_shift", "P2_vs_P1_1pl_cell_shift"]:
        med = float(piv[c].median())
        sd = float(piv[c].std(ddof=0))
        piv[f"{c}_median"] = med
        piv[f"{c}_sd"] = sd
        sign_a = np.sign(piv[c].astype(float))
        sign_m = np.sign(med)
        piv[f"{c}_direction_discordant"] = (sign_a != sign_m) & (sign_a != 0) & (sign_m != 0)
        piv[f"{c}_extreme_same_direction"] = (~piv[f"{c}_direction_discordant"]) & ((piv[c] - med).abs() >= (1.5 * sd))
    piv["is_atypical_core"] = piv["P2_vs_P1_2sg_cell_shift_direction_discordant"] & piv["P2_vs_P1_1pl_cell_shift_direction_discordant"]
    piv["is_atypical_broad"] = piv["P2_vs_P1_2sg_cell_shift_direction_discordant"] | piv["P2_vs_P1_1pl_cell_shift_direction_discordant"]
    piv["is_extreme_same_direction"] = piv["P2_vs_P1_2sg_cell_shift_extreme_same_direction"] | piv["P2_vs_P1_1pl_cell_shift_extreme_same_direction"]

    def _type(row):
        e2 = float(row["P2_vs_P1_2sg_cell_shift"])
        e1 = float(row["P2_vs_P1_1pl_cell_shift"])
        if e2 > 0 and e1 <= 0:
            return "Type C: accusatory 2sg turn"
        if e1 > 0 and e2 <= 0:
            return "Type A/B: collective-we intensification"
        if e1 > 0 and e2 > 0:
            return "Type B: exclusive-we boundarying"
        return "Type D: introspective 1sg turn"

    piv["typology_class"] = piv.apply(_type, axis=1)
    piv["exploratory_mixed_profile"] = (piv["P2_vs_P1_1pl_cell_shift"] > 0) & (piv["P2_vs_P1_2sg_cell_shift"] > 0)
    piv["notes"] = np.where(piv["author"].isin(list(SWITCHERS)), "bilingual_switcher", "")
    return piv.sort_values(["is_atypical_core", "is_atypical_broad", "author"], ascending=[False, False, True]).reset_index(drop=True)


def sample_close_reading(df: pd.DataFrame, typ: pd.DataFrame, seed: int = 20260505) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    core = typ.loc[typ["is_atypical_core"], "author"].astype(str).tolist()
    broad = typ.loc[typ["is_atypical_broad"] & ~typ["author"].isin(core), ["author", "P2_vs_P1_1pl_cell_shift", "P2_vs_P1_2sg_cell_shift"]].copy()
    broad["gap"] = broad["P2_vs_P1_1pl_cell_shift"].abs() + broad["P2_vs_P1_2sg_cell_shift"].abs()
    secondary = broad.sort_values("gap", ascending=False)["author"].astype(str).tolist()[:1]
    extreme = (
        typ.loc[
            typ["is_extreme_same_direction"] & ~typ["author"].isin(core + secondary),
            ["author", "P2_vs_P1_1pl_cell_shift", "P2_vs_P1_2sg_cell_shift"],
        ]
        .assign(gap=lambda x: x["P2_vs_P1_1pl_cell_shift"].abs() + x["P2_vs_P1_2sg_cell_shift"].abs())
        .sort_values("gap", ascending=False)["author"]
        .astype(str)
        .tolist()[:1]
    )
    target = core + secondary + extreme
    poem_meta = (
        df[df["period3"].eq("P2_2022_plus") & df["author"].isin(target)]
        .groupby("poem_id", as_index=False)
        .agg(author=("author", "first"), year_int=("year_int", "first"))
    )
    rows: list[dict] = []
    for author, g in poem_meta.groupby("author", sort=True):
        ids = g["poem_id"].astype(str).tolist()
        take = min(5, len(ids))
        picked = rng.choice(np.array(ids), size=take, replace=False).tolist() if take > 0 else []
        for pid in picked:
            y = g.loc[g["poem_id"].astype(str).eq(pid), "year_int"].iloc[0]
            rows.append({"author": author, "poem_id": pid, "year_int": int(y) if pd.notna(y) else "", "sample_seed": seed})
    return pd.DataFrame(rows).sort_values(["author", "poem_id"]).reset_index(drop=True)


def plot_period_panels(long_df: pd.DataFrame, typ: pd.DataFrame, out_pdf: Path) -> None:
    d = long_df.groupby(["author", "period3", "cell"], as_index=False)["y"].mean()
    atyp_core = set(typ.loc[typ["is_atypical_core"], "author"].astype(str))
    atyp_broad = set(typ.loc[typ["is_atypical_broad"], "author"].astype(str))
    authors = sorted(d["author"].unique().tolist())
    n = len(authors)
    ncols = 3
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.0 * nrows), sharex=True, sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    colors = {"1sg": "#1f77b4", "1pl": "#ff7f0e", "2sg": "#2ca02c", "2pl": "#d62728"}
    xmap = {p: i for i, p in enumerate(PERIODS)}
    for i, author in enumerate(authors):
        ax = axes[i]
        ad = d[d["author"].eq(author)]
        for c in CELL4:
            cd = ad[ad["cell"].eq(c)]
            xs = [xmap[p] for p in cd["period3"]]
            ax.plot(xs, cd["y"], marker="o", linewidth=1.8, label=c, color=colors[c])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["P1", "P2"])
        ax.set_ylim(0, 1)
        mark = "**" if author in atyp_core else ("*" if author in atyp_broad else "")
        color = "darkred" if author in atyp_core else ("darkorange" if author in atyp_broad else "black")
        title = f"{author} {mark}"
        ax.set_title(title, fontsize=9, color=color)
        ax.grid(alpha=0.2)
    for j in range(len(authors), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Period-based per-author cell proportions (P1/P2)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_cohort_period_profile(long_df: pd.DataFrame, out_pdf: Path) -> None:
    d = long_df.groupby(["period3", "cell"], as_index=False)["y"].mean()
    xmap = {p: i for i, p in enumerate(PERIODS)}
    colors = {"1sg": "#1f77b4", "1pl": "#ff7f0e", "2sg": "#2ca02c", "2pl": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CELL4:
        cd = d[d["cell"].eq(c)]
        xs = [xmap[p] for p in cd["period3"]]
        ax.plot(xs, cd["y"], marker="o", linewidth=2.0, markersize=5, label=c, color=colors[c])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["P1", "P2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean per-poem proportion")
    ax.set_title("Cohort-level pronoun profile by period (9-author roster)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_per_author_deltas(per_author: pd.DataFrame, out_pdf: Path) -> None:
    piv = per_author.pivot(index="author", columns="contrast", values="estimate_logit").reset_index()
    piv = piv.rename(
        columns={
            "P2_vs_P1_1pl_cell_shift": "delta_1pl",
            "P2_vs_P1_2sg_cell_shift": "delta_2sg",
        }
    )
    piv = piv.sort_values("delta_1pl")
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(piv["delta_1pl"], y, color="#ff7f0e", label="1pl (P2-P1)", s=50)
    ax.scatter(piv["delta_2sg"], y, color="#2ca02c", label="2sg (P2-P1)", s=50)
    for i in range(len(piv)):
        ax.plot([piv["delta_1pl"].iloc[i], piv["delta_2sg"].iloc[i]], [y[i], y[i]], color="#cccccc", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(piv["author"])
    ax.set_xlabel("Per-author logit shift (P2 vs P1)")
    ax.set_title("Per-author pronoun shift deltas")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prereg v2 model and typology outputs.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    ap.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20260505)
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.input.resolve(), args.layer0.resolve())
    roster = load_roster(args.roster.resolve())
    long_df = build_long(df, roster_authors=roster, min_n12=5)
    fit = fit_main(long_df)

    coef = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values})
    ci = fit.conf_int()
    coef["ci95_low"] = ci[0].values
    coef["ci95_high"] = ci[1].values
    coef.to_csv(out / "within_author_model_coefficients.csv", index=False)

    per_author = per_author_contrasts(long_df)
    per_author.to_csv(out / "per_author_contrast_estimates_v2.csv", index=False)
    typ = classify_typology(per_author)
    typ.to_csv(out / "author_typology_table.csv", index=False)

    sampled = sample_close_reading(df, typ, seed=args.seed)
    sampled.to_csv(out / "qualitative_close_reading_sample.csv", index=False)

    plot_period_panels(long_df, typ, fig_dir / "fig_period_based_author_panels.pdf")
    plot_cohort_period_profile(long_df, fig_dir / "fig_cohort_period_profile.pdf")
    plot_per_author_deltas(per_author, fig_dir / "fig_per_author_delta_dotplot.pdf")

    counts = long_df.groupby(["author", "period3"])["poem_id"].nunique().unstack(fill_value=0).reset_index()
    counts.to_csv(out / "author_period_poem_counts.csv", index=False)

    with (out / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Prereg v2 outputs\n\n")
        f.write("- Cohort-level within-author model with `log_n_tokens` covariate.\n")
        f.write("- Typology separates direction-discordant atypicality from extreme same-direction intensity.\n")
        f.write("- Qualitative sample prioritizes atypical-core, with fixed seed random selection in P2.\n")

    print(f"Wrote typology and period-model outputs to: {out}")


if __name__ == "__main__":
    main()
