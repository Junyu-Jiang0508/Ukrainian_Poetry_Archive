"""Join Q1 per-offset GLM outputs into long/wide comparison tables + forest plots."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.pronoun_encoding import PRIMARY_GLM_CELLS
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_Q1_DIR = ROOT / "outputs" / "02_modeling_q1_per_cell_glm"
DEFAULT_OUT = ROOT / "outputs" / "02_modeling_robustness_offset_comparison"

OFFSET_FILES: tuple[tuple[str, str], ...] = (
    ("n_stanzas", "q1_poem_per_cell_glm_by_language.csv"),
    ("n_tokens", "q1_poem_per_cell_glm_by_language_offset_n_tokens.csv"),
    ("n_finite_verbs", "q1_poem_per_cell_glm_by_language_offset_n_finite_verbs.csv"),
    (
        "n_finite_verbs_excl_imperative",
        "q1_poem_per_cell_glm_by_language_offset_n_finite_verbs_excl_imperative.csv",
    ),
)


def _read_if_exists(q1_dir: Path, name: str) -> pd.DataFrame | None:
    p = q1_dir / name
    if not p.is_file():
        return None
    df = pd.read_csv(p, low_memory=False)
    if df.empty:
        return None
    return df


def build_long_table(q1_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return (long_df, exposure_types_present)."""
    parts: list[pd.DataFrame] = []
    present: list[str] = []
    for exposure_type, fname in OFFSET_FILES:
        df = _read_if_exists(q1_dir, fname)
        if df is None:
            log.warning("Missing or empty: %s", fname)
            continue
        d = df.copy()
        d["exposure_type"] = exposure_type
        parts.append(d)
        present.append(exposure_type)
    if not parts:
        return pd.DataFrame(), []
    long_df = pd.concat(parts, ignore_index=True)
    keep = [
        "language_stratum",
        "cell",
        "exposure_type",
        "n_poems",
        "rate_ratio_post_vs_pre",
        "rate_ratio_ci95_low",
        "rate_ratio_ci95_high",
        "p_value_clustered_author",
        "q_value_bh_within_stratum",
        "is_primary_stratum",
    ]
    long_df = long_df[[c for c in keep if c in long_df.columns]]
    return long_df, present


def build_wide_rr(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (st, cell), g in long_df.groupby(["language_stratum", "cell"], sort=False):
        row: dict[str, object] = {"language_stratum": st, "cell": cell}
        for _, r in g.iterrows():
            et = str(r["exposure_type"])
            row[f"rr__{et}"] = r.get("rate_ratio_post_vs_pre", np.nan)
            row[f"ci_low__{et}"] = r.get("rate_ratio_ci95_low", np.nan)
            row[f"ci_high__{et}"] = r.get("rate_ratio_ci95_high", np.nan)
            row[f"q__{et}"] = r.get("q_value_bh_within_stratum", np.nan)
            row[f"n_poems__{et}"] = r.get("n_poems", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_forests(long_df: pd.DataFrame, out_dir: Path, exposure_types: list[str]) -> None:
    if long_df.empty:
        return
    cells = [c for c in PRIMARY_GLM_CELLS if c in long_df["cell"].unique()]
    if not cells:
        cells = sorted(long_df["cell"].unique().tolist())

    for stratum in long_df["language_stratum"].unique():
        sub = long_df.loc[long_df["language_stratum"].eq(stratum)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axes_flat = axes.ravel()
        for ax, cell in zip(axes_flat, cells):
            csub = sub.loc[sub["cell"].eq(cell)]
            if csub.empty:
                ax.set_visible(False)
                continue
            rr: list[float] = []
            lo: list[float] = []
            hi: list[float] = []
            labels: list[str] = []
            for et in exposure_types:
                r = csub.loc[csub["exposure_type"].eq(et)]
                if r.empty:
                    continue
                r0 = r.iloc[0]
                rv = r0["rate_ratio_post_vs_pre"]
                if pd.isna(rv):
                    continue
                rr.append(float(rv))
                lo.append(float(r0["rate_ratio_ci95_low"]) if pd.notna(r0["rate_ratio_ci95_low"]) else float(rv))
                hi.append(float(r0["rate_ratio_ci95_high"]) if pd.notna(r0["rate_ratio_ci95_high"]) else float(rv))
                labels.append(et)
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
            if rr:
                y_pos = np.arange(len(rr))
                rr_a = np.array(rr, dtype=float)
                lo_a = np.array(lo, dtype=float)
                hi_a = np.array(hi, dtype=float)
                xerr = np.vstack([rr_a - lo_a, hi_a - rr_a])
                ax.errorbar(rr_a, y_pos, xerr=xerr, fmt="o", capsize=4, markersize=5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(cell)
            ax.set_xlabel("Rate ratio (P2 vs P1)")
        fig.suptitle(f"Offset comparison — {stratum}")
        fig.tight_layout()
        safe = str(stratum).replace("/", "_")
        fig.savefig(out_dir / f"offset_comparison_forest__{safe}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Merge Q1 offset GLM CSVs for robustness reporting.")
    ap.add_argument("--q1-dir", type=Path, default=DEFAULT_Q1_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    q1_dir = args.q1_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df, present = build_long_table(q1_dir)
    if long_df.empty:
        raise SystemExit("No Q1 offset CSVs found; run 02b/02b2/02b3/02b4 first.")

    long_df.to_csv(out_dir / "offset_comparison_long.csv", index=False)
    wide = build_wide_rr(long_df)
    wide.to_csv(out_dir / "offset_comparison_wide_rr.csv", index=False)

    three = ("n_stanzas", "n_tokens", "n_finite_verbs")
    long_3 = long_df.loc[long_df["exposure_type"].isin(three)].copy()
    long_3.to_csv(out_dir / "offset_comparison_3way.csv", index=False)

    plot_forests(long_df, out_dir, present)
    log.info("Wrote comparison tables and forests to %s", out_dir)


if __name__ == "__main__":
    main()
