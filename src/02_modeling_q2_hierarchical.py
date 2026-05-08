"""Q2: author-level alignment/divergence via hierarchical count models.

Five primary cells with poem-level counts and negative-binomial likelihood; offset via
log(exposure_n_stanzas). Random slope on ``period_post`` when sampling succeeds.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.language_strata import (
    LANGUAGE_STRATA,
    filter_annotation_for_inference_language,
    filter_poems_by_language_stratum,
)
from utils.poem_cell_counts import build_poem_cell_table_with_exposure
from utils.pronoun_encoding import PRIMARY_GLM_CELLS_BAYESIAN, pronoun_class_sixway_column
from utils.stats_common import normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q2_hierarchical"

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = [PERIOD_P1, PERIOD_P2]


def load_roster_authors(roster_path: Path | None) -> set[str] | None:
    if roster_path is None or not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if "included" not in r.columns or "author" not in r.columns:
        return None
    return set(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def load_and_filter(
    path: Path,
    layer0_path: Path | None,
    *,
    language_audit_dir: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["poem_id"] = df["poem_id"].astype(str).str.strip()
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["period3"] = df["year_int"].map(period_three_way)
    df["person"] = df["person"].fillna("").astype(str).str.strip()
    df["number"] = df["number"].fillna("").astype(str).str.strip()
    df["person_number"] = pronoun_class_sixway_column(df)
    df["language_clean"] = df["language"].fillna("").astype(str).str.strip()

    if "is_repeat" in df.columns and "is_translation" in df.columns:
        df["is_repeat"] = normalize_bool_flag(df["is_repeat"])
        df["is_translation"] = normalize_bool_flag(df["is_translation"])
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
    out, _ = filter_annotation_for_inference_language(out, audit_dir=language_audit_dir)
    return out


def _detect_random_slope_var(posterior, term: str, group: str) -> str:
    candidates = [name for name in posterior.data_vars if (term in name and "|" in name and group in name)]
    if not candidates:
        raise KeyError(f"Could not find random slope var for term={term}, group={group}.")
    exact = [name for name in candidates if name == f"{term}|{group}"]
    return exact[0] if exact else candidates[0]


def _author_dim_name(da) -> str:
    for dim in da.dims:
        if dim not in ("chain", "draw", "__obs__"):
            return dim
    raise KeyError("Could not detect author coordinate dimension.")


def _run_bambi_nb(
    cdf: pd.DataFrame,
    *,
    random_slope: bool,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
):
    import bambi as bmb

    if random_slope:
        formula = "k ~ period_post + offset(log_exposure) + (1 + period_post | author)"
    else:
        formula = "k ~ period_post + offset(log_exposure) + (1 | author)"

    model = bmb.Model(formula, data=cdf, family="negativebinomial")
    if random_slope:
        prior_slope = bmb.Prior("Normal", mu=0.0, sigma=bmb.Prior("HalfNormal", sigma=1.0))
        model.set_priors(priors={"period_post|author": prior_slope})
    return model.fit(
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        cores=int(cores),
        target_accept=float(target_accept),
        random_seed=int(random_seed),
    )


def fit_hierarchical_per_cell(
    poem_cell: pd.DataFrame,
    roster_authors: set[str] | None,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
    *,
    language_stratum: str,
    cells: list[str],
    exposure_type: str = "n_stanzas",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import bambi as bmb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Bambi is required for Q2 hierarchical modeling. Install with `pip install bambi`."
        ) from exc

    if exposure_type == "n_stanzas":
        ex_col = "exposure_n_stanzas"
    elif exposure_type == "n_tokens":
        ex_col = "exposure_n_tokens"
    elif exposure_type == "n_finite_verbs":
        ex_col = "exposure_n_finite_verbs"
    else:
        raise ValueError(f"Unknown exposure_type: {exposure_type!r}")

    dat = poem_cell.copy()
    if exposure_type == "n_finite_verbs" and "include_in_fv_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_fv_offset_models"].astype(bool)].copy()
    elif "include_in_offset_models" in dat.columns:
        dat = dat.loc[dat["include_in_offset_models"].astype(bool)].copy()
    dat = dat[dat["period3"].isin(PERIODS)]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    dat["period_post"] = (dat["period3"] == PERIOD_P2).astype(int)
    dat["author"] = dat["author"].astype(str)
    dat = dat[dat["author"].str.strip().ne("")]

    fixed_rows: list[dict[str, object]] = []
    author_rows: list[dict[str, object]] = []

    for cell in cells:
        cdf = dat.copy()
        cdf["k"] = cdf[cell].astype(int)
        cdf["_ex"] = cdf[ex_col].astype(float)
        cdf = cdf[cdf["_ex"].gt(0)].copy()
        if cdf.empty:
            continue
        cdf["log_exposure"] = np.log(cdf["_ex"])
        assert cdf["log_exposure"].notna().all(), f"NaN log_exposure for cell={cell}"
        assert np.isfinite(cdf["log_exposure"].to_numpy()).all(), f"non-finite log_exposure cell={cell}"

        if cdf["period_post"].nunique() < 2 or cdf["author"].nunique() < 2 or int(cdf["k"].sum()) == 0:
            continue

        idata = None
        used_slope = False
        try:
            idata = _run_bambi_nb(
                cdf,
                random_slope=True,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
            )
            used_slope = True
        except Exception as exc:
            warnings.warn(
                f"Q2 NB random-slope failed for cell={cell!r} stratum={language_stratum!r} "
                f"({type(exc).__name__}: {exc}); falling back to random intercept.",
                stacklevel=1,
            )
            try:
                idata = _run_bambi_nb(
                    cdf,
                    random_slope=False,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    random_seed=random_seed,
                )
                used_slope = False
            except Exception as exc2:
                warnings.warn(
                    f"Q2 NB random-intercept fallback failed for cell={cell!r} "
                    f"stratum={language_stratum!r} ({type(exc2).__name__}: {exc2}). Skipping.",
                    stacklevel=1,
                )
                continue

        post = idata.posterior
        if "period_post" not in post.data_vars:
            continue
        fixed = post["period_post"].stack(sample=("chain", "draw")).values
        fixed_mean = float(np.mean(fixed))
        fixed_low = float(np.quantile(fixed, 0.025))
        fixed_high = float(np.quantile(fixed, 0.975))
        fixed_rows.append(
            {
                "language_stratum": language_stratum,
                "cell": cell,
                "model_spec": "random_slope_nb" if used_slope else "random_intercept_nb_fallback",
                "n_poems": int(len(cdf)),
                "n_authors": int(cdf["author"].nunique()),
                "population_shift_mean_log_mu": fixed_mean,
                "population_shift_hdi95_low": fixed_low,
                "population_shift_hdi95_high": fixed_high,
                "population_shift_rate_ratio_mean": float(np.exp(fixed_mean)),
                "population_shift_rate_ratio_hdi95_low": float(np.exp(fixed_low)),
                "population_shift_rate_ratio_hdi95_high": float(np.exp(fixed_high)),
                "exposure_type": exposure_type,
            }
        )

        if not used_slope:
            continue

        rs_var = _detect_random_slope_var(post, term="period_post", group="author")
        rs = post[rs_var]
        author_dim = _author_dim_name(rs)
        rs_samples = rs.stack(sample=("chain", "draw"))
        author_labels = rs_samples.coords[author_dim].values
        for author_label in author_labels:
            dev_samples = rs_samples.sel({author_dim: author_label}).values
            total_samples = fixed + dev_samples
            dev_mean = float(np.mean(dev_samples))
            dev_low = float(np.quantile(dev_samples, 0.025))
            dev_high = float(np.quantile(dev_samples, 0.975))
            total_mean = float(np.mean(total_samples))
            total_low = float(np.quantile(total_samples, 0.025))
            total_high = float(np.quantile(total_samples, 0.975))
            author_rows.append(
                {
                    "language_stratum": language_stratum,
                    "cell": cell,
                    "author": str(author_label),
                    "author_period_shift_deviation_mean_log_mu": dev_mean,
                    "author_period_shift_deviation_hdi95_low": dev_low,
                    "author_period_shift_deviation_hdi95_high": dev_high,
                    "author_total_period_shift_mean_log_mu": total_mean,
                    "author_total_period_shift_hdi95_low": total_low,
                    "author_total_period_shift_hdi95_high": total_high,
                    "author_total_period_shift_rate_ratio_mean": float(np.exp(total_mean)),
                    "author_total_period_shift_rate_ratio_hdi95_low": float(np.exp(total_low)),
                    "author_total_period_shift_rate_ratio_hdi95_high": float(np.exp(total_high)),
                    "exposure_type": exposure_type,
                }
            )

    return pd.DataFrame(fixed_rows), pd.DataFrame(author_rows)


def plot_author_random_slope_caterpillar(
    author_df: pd.DataFrame,
    out_path: Path,
    *,
    cell_order: list[str],
) -> None:
    """Facet by pronoun cell: authors sorted by random-slope deviation; dashed vertical line at zero."""
    if author_df.empty:
        return
    need = (
        "cell",
        "author",
        "author_period_shift_deviation_mean_log_mu",
        "author_period_shift_deviation_hdi95_low",
        "author_period_shift_deviation_hdi95_high",
    )
    if any(c not in author_df.columns for c in need):
        return

    cells_present = [c for c in cell_order if c in author_df["cell"].unique()]
    if not cells_present:
        return

    n_panels = len(cells_present)
    ncols = 2 if n_panels <= 4 else min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    max_authors_any_cell = int(author_df.groupby("cell")["author"].nunique().max())
    row_h = max(4.2, min(34.0, 0.32 * max_authors_any_cell + 2.5))
    fig_h = row_h * nrows + 1.95
    fig_w = max(12.5, 4.35 * ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=False,
    )
    axes_flat = np.asarray(axes).ravel()

    xlabel_long = (
        "Posterior mean deviation on log(scale): random slope on period_post\n"
        "relative to population mean in this cell (0 ≈ typical author trajectory)"
    )

    for i, cell in enumerate(cells_present):
        ax = axes_flat[i]
        d_sub = author_df.loc[author_df["cell"].eq(cell)].copy()
        d_sub = d_sub.sort_values("author_period_shift_deviation_mean_log_mu", ascending=True)
        y = np.arange(len(d_sub))
        xm = d_sub["author_period_shift_deviation_mean_log_mu"].to_numpy(dtype=float)
        lo = d_sub["author_period_shift_deviation_hdi95_low"].to_numpy(dtype=float)
        hi = d_sub["author_period_shift_deviation_hdi95_high"].to_numpy(dtype=float)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=0)
        ax.hlines(y, lo, hi, color="#4c78a8", alpha=0.88, linewidth=1.6, zorder=1)
        ax.plot(xm, y, "o", color="#1a1a1a", markersize=3.6, zorder=2)

        ax.set_title(cell, fontsize=11, fontweight="semibold")
        ax.set_yticks(y)
        n_y = len(y)
        ylab_fs = max(6.6, min(9.3, 9.9 - 0.055 * max(0.0, n_y - 10.0)))
        ax.set_yticklabels(d_sub["author"].astype(str), fontsize=ylab_fs)

        merged = np.concatenate([xm, lo, hi])
        merged = merged[np.isfinite(merged)]
        if merged.size:
            span_m = float(merged.max() - merged.min())
            pad_m = max(0.085 * span_m, 0.12)
            ax.set_xlim(float(merged.min()) - pad_m, float(merged.max()) + pad_m)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    title = "Author deviations from population wartime shift (1st/2nd person)"
    if "language_stratum" in author_df.columns and author_df["language_stratum"].nunique() == 1:
        title = f"{title} — {author_df['language_stratum'].iloc[0]}"
    fig.suptitle(title, fontsize=12.25, y=1.015)
    fig.tight_layout(h_pad=1.95, w_pad=1.45, rect=[0.0, 0.06, 1.0, 0.985])
    fig.text(0.5, 0.012, xlabel_long, ha="center", fontsize=9.35)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Q2 hierarchical random-slope model per pronoun cell.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20260506)
    parser.add_argument(
        "--strata",
        type=str,
        default=",".join(LANGUAGE_STRATA),
        help=f"Comma-separated subset of: {','.join(LANGUAGE_STRATA)}",
    )
    parser.add_argument(
        "--exposure-type",
        type=str,
        default="n_stanzas",
        choices=("n_stanzas", "n_tokens", "n_finite_verbs"),
    )
    parser.add_argument(
        "--skip-caterpillar",
        action="store_true",
        help="Do not write caterpillar PDFs under figures/",
    )
    args = parser.parse_args()

    want_strata = tuple(s.strip() for s in args.strata.split(",") if s.strip())
    for s in want_strata:
        if s not in LANGUAGE_STRATA:
            raise SystemExit(f"Unknown stratum {s!r}. Choose from {LANGUAGE_STRATA}")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "language_stratum_audit"

    filtered = load_and_filter(
        args.input.resolve(),
        args.layer0.resolve() if args.layer0 else None,
        language_audit_dir=audit_dir,
    )
    roster_authors = load_roster_authors(args.roster.resolve() if args.roster else None)
    poem_cell = build_poem_cell_table_with_exposure(filtered)

    fixed_parts: list[pd.DataFrame] = []
    author_parts: list[pd.DataFrame] = []
    cell_list = list(PRIMARY_GLM_CELLS_BAYESIAN)
    for i, stratum in enumerate(want_strata):
        sub = filter_poems_by_language_stratum(poem_cell, stratum)
        seed_i = int(args.random_seed) + i * 10_007
        fdf, adf = fit_hierarchical_per_cell(
            poem_cell=sub,
            roster_authors=roster_authors,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=seed_i,
            language_stratum=stratum,
            cells=cell_list,
            exposure_type=args.exposure_type,
        )
        if not fdf.empty:
            fixed_parts.append(fdf)
        if not adf.empty:
            author_parts.append(adf)
        if not args.skip_caterpillar and not adf.empty:
            plot_author_random_slope_caterpillar(
                adf,
                fig_dir / f"fig_q2_author_random_slope_caterpillar_{stratum}.pdf",
                cell_order=cell_list,
            )

    poem_cell.to_csv(out_dir / "q2_poem_cell_counts_12.csv", index=False)
    fixed_df = pd.concat(fixed_parts, ignore_index=True) if fixed_parts else pd.DataFrame()
    author_df = pd.concat(author_parts, ignore_index=True) if author_parts else pd.DataFrame()
    fixed_df.to_csv(out_dir / "q2_population_shifts_by_cell.csv", index=False)
    author_df.to_csv(out_dir / "q2_author_random_slope_summaries.csv", index=False)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Q2 Hierarchical negative-binomial models (1st/2nd person, poem level)\n\n")
        f.write(
            "- Primary cells (5-cell, Bayesian): "
            "`{1sg, 1pl, 2sg, 2pl_vy_polite_singular, 2pl_vy_true_plural}`. The polite-singular "
            "ви cell is **kept on the Bayesian path** (`PRIMARY_GLM_CELLS_BAYESIAN`) because "
            "negative-binomial random-slope shrinkage produces meaningful — if wide — HDIs "
            "even when the cell is sparse. Frequentist Q1/Q1b/Q1c/robustness drop this cell "
            "(`PRIMARY_GLM_CELLS_FREQUENTIST`) because separation kills MLE.\n"
        )
        f.write("- Strata: `pooled_Ukrainian_Russian`, `Ukrainian`, `Russian`.\n")
        f.write("- Default model: `k ~ period_post + offset(log_exposure) + (1 + period_post | author)` (NB).\n")
        f.write(f"- Exposure used in this run: `{args.exposure_type}`.\n")

    print(f"Wrote Q2 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
