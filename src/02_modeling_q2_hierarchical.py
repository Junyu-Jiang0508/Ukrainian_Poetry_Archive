"""Q2: author-level alignment/divergence via hierarchical binomial models.

First/second-person cells only ({1sg,1pl,2sg,2pl}); poem-level counts; denominator
per poem is the sum of those four cells (third person excluded).

Per cell and language stratum (pooled Ukrainian∪Russian, Ukrainian-only,
Russian-only), fit:
    p(k, n_total) ~ period_post + (1 + period_post | author)

Uses Bambi / formulae binomial syntax ``p(successes, trials)`` (not ``k | trials(n)``,
which formulae 0.6+ parses incorrectly).

The author-specific random slope for `period_post` is interpreted as each
author's deviation from the population shift (positive = above-trend adaptation,
negative = counter-trend).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.language_strata import LANGUAGE_STRATA, filter_poems_by_language_stratum
from utils.pronoun_encoding import pronoun_class_sixway_column
from utils.stats_common import normalize_bool_flag, period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_ROSTER = ROOT / "outputs" / "03_reporting_roster_freeze" / "roster_v1_frozen.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_q2_hierarchical"

PERIOD_P1 = "P1_2014_2021"
PERIOD_P2 = "P2_2022_plus"
PERIODS = [PERIOD_P1, PERIOD_P2]
CELL12 = ["1sg", "1pl", "2sg", "2pl"]
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}


def load_roster_authors(roster_path: Path | None) -> set[str] | None:
    if roster_path is None or not roster_path.is_file():
        return None
    r = pd.read_csv(roster_path, low_memory=False)
    if "included" not in r.columns or "author" not in r.columns:
        return None
    return set(r.loc[r["included"].astype(bool), "author"].astype(str).tolist())


def load_and_filter(path: Path, layer0_path: Path | None) -> pd.DataFrame:
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
    out = out[~out["language_clean"].isin(QIRIMLI_CODES)].copy()
    out = out[out["person_number"].isin(CELL12)].copy()
    return out


def build_poem_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["poem_id", "person_number"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for cell in CELL12:
        if cell not in counts.columns:
            counts[cell] = 0
    counts = counts[["poem_id"] + CELL12]
    counts["n_total"] = counts[CELL12].sum(axis=1).astype(int)

    meta = (
        df.groupby("poem_id", as_index=False)
        .agg(author=("author", "first"), language_clean=("language_clean", "first"), year_int=("year_int", "first"))
        .copy()
    )
    meta["period3"] = meta["year_int"].map(period_three_way)
    return counts.merge(meta, on="poem_id", how="left")


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import bambi as bmb
    except ImportError as exc:
        raise ImportError(
            "Bambi is required for Q2 hierarchical modeling. Install with `pip install bambi`."
        ) from exc

    dat = poem_cell.copy()
    dat = dat[dat["period3"].isin(PERIODS)]
    dat = dat[dat["n_total"] > 0]
    if roster_authors is not None:
        dat = dat[dat["author"].astype(str).isin(roster_authors)]
    dat["period_post"] = (dat["period3"] == PERIOD_P2).astype(int)
    dat["author"] = dat["author"].astype(str)
    dat = dat[dat["author"].str.strip().ne("")]

    fixed_rows: list[dict] = []
    author_rows: list[dict] = []

    for cell in cells:
        cdf = dat.copy()
        cdf["k"] = cdf[cell].astype(int)
        if cdf["period_post"].nunique() < 2 or cdf["author"].nunique() < 2 or cdf["k"].nunique() < 2:
            continue
        model = bmb.Model(
            "p(k, n_total) ~ period_post + (1 + period_post | author)",
            data=cdf,
            family="binomial",
        )
        idata = model.fit(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores),
            target_accept=float(target_accept),
            random_seed=int(random_seed),
        )
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
                "n_poems": int(len(cdf)),
                "n_authors": int(cdf["author"].nunique()),
                "population_shift_mean_log_odds": fixed_mean,
                "population_shift_hdi95_low": fixed_low,
                "population_shift_hdi95_high": fixed_high,
                "population_shift_or_mean": float(np.exp(fixed_mean)),
                "population_shift_or_hdi95_low": float(np.exp(fixed_low)),
                "population_shift_or_hdi95_high": float(np.exp(fixed_high)),
            }
        )

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
                    "author_period_shift_deviation_mean_log_odds": dev_mean,
                    "author_period_shift_deviation_hdi95_low": dev_low,
                    "author_period_shift_deviation_hdi95_high": dev_high,
                    "author_total_period_shift_mean_log_odds": total_mean,
                    "author_total_period_shift_hdi95_low": total_low,
                    "author_total_period_shift_hdi95_high": total_high,
                    "author_total_period_shift_or_mean": float(np.exp(total_mean)),
                    "author_total_period_shift_or_hdi95_low": float(np.exp(total_low)),
                    "author_total_period_shift_or_hdi95_high": float(np.exp(total_high)),
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
        "author_period_shift_deviation_mean_log_odds",
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
        "Posterior mean deviation (log-odds): author random slope on period_post\n"
        "relative to population mean in this cell (0 = average author)"
    )

    for i, cell in enumerate(cells_present):
        ax = axes_flat[i]
        d = author_df.loc[author_df["cell"].eq(cell)].copy()
        d = d.sort_values("author_period_shift_deviation_mean_log_odds", ascending=True)
        y = np.arange(len(d))
        xm = d["author_period_shift_deviation_mean_log_odds"].to_numpy(dtype=float)
        lo = d["author_period_shift_deviation_hdi95_low"].to_numpy(dtype=float)
        hi = d["author_period_shift_deviation_hdi95_high"].to_numpy(dtype=float)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=0)
        ax.hlines(y, lo, hi, color="#4c78a8", alpha=0.88, linewidth=1.6, zorder=1)
        ax.plot(xm, y, "o", color="#1a1a1a", markersize=3.6, zorder=2)

        ax.set_title(cell, fontsize=11, fontweight="semibold")
        ax.set_yticks(y)
        n_y = len(y)
        ylab_fs = max(6.6, min(9.3, 9.9 - 0.055 * max(0.0, n_y - 10.0)))
        ax.set_yticklabels(d["author"].astype(str), fontsize=ylab_fs)

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
    parser = argparse.ArgumentParser(description="Q2 hierarchical random-slope model per pronoun cell.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=20260506)
    parser.add_argument(
        "--strata",
        type=str,
        default=",".join(LANGUAGE_STRATA),
        help=f"Comma-separated subset of: {','.join(LANGUAGE_STRATA)}",
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

    filtered = load_and_filter(args.input.resolve(), args.layer0.resolve() if args.layer0 else None)
    roster_authors = load_roster_authors(args.roster.resolve() if args.roster else None)
    poem_cell = build_poem_cell_table(filtered)

    fixed_parts: list[pd.DataFrame] = []
    author_parts: list[pd.DataFrame] = []
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
            cells=list(CELL12),
        )
        if not fdf.empty:
            fixed_parts.append(fdf)
        if not adf.empty:
            author_parts.append(adf)
        if not args.skip_caterpillar and not adf.empty:
            plot_author_random_slope_caterpillar(
                adf,
                fig_dir / f"fig_q2_author_random_slope_caterpillar_{stratum}.pdf",
                cell_order=list(CELL12),
            )

    poem_cell.to_csv(out_dir / "q2_poem_cell_counts_12.csv", index=False)
    fixed_df = pd.concat(fixed_parts, ignore_index=True) if fixed_parts else pd.DataFrame()
    author_df = pd.concat(author_parts, ignore_index=True) if author_parts else pd.DataFrame()
    fixed_df.to_csv(out_dir / "q2_population_shifts_by_cell.csv", index=False)
    author_df.to_csv(out_dir / "q2_author_random_slope_summaries.csv", index=False)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Q2 Hierarchical random-slope outputs (1st/2nd person, poem level)\n\n")
        f.write("- Cells: `1sg, 1pl, 2sg, 2pl`; `n_total` = sum of those counts in the poem.\n")
        f.write("- Strata: `pooled_Ukrainian_Russian`, `Ukrainian`, `Russian` (exact poem language).\n")
        f.write("- Model per stratum × cell: `p(k, n_total) ~ period_post + (1 + period_post | author)`.\n")
        f.write("- CSVs include column `language_stratum`. Caterpillar: one PDF per stratum under `figures/`.\n")

    print(f"Wrote Q2 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
