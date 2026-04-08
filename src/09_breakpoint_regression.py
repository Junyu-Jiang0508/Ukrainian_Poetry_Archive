"""WLS breakpoint regression + PELT changepoints. Args: [min_poems] [A|B|C]."""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    from statsmodels.regression.linear_model import WLS, OLS
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAIL = True
except ImportError:
    STATSMODELS_AVAIL = False

try:
    import ruptures as rpt
    RUPTURES_AVAIL = True
except ImportError:
    RUPTURES_AVAIL = False

from importlib.util import spec_from_file_location, module_from_spec
_08a = spec_from_file_location("adaptive", Path(__file__).parent / "08_adaptive_binning.py")
_mod_08a = module_from_spec(_08a)
_08a.loader.exec_module(_mod_08a)

OUTPUT_DIR = Path("outputs/09_breakpoint_regression")
BREAK_2014 = pd.Timestamp("2014-02-01")
BREAK_2022 = pd.Timestamp("2022-02-01")
PRONOUN_COLS = ["1sg", "1pl", "2", "3sg", "3pl"]
WEIGHT_SCHEMES = {"A": "poem_count", "B": "time_norm", "C": "composite"}


def load_and_prepare_data(min_poems: int = 30) -> tuple:
    INPUT_CSV = Path("outputs/01_pronoun_detection/ukrainian_pronouns_projection_final.csv")
    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
        on_bad_lines="skip",
        dtype={"person": "object", "number": "object"},
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["person"] = df["person"].astype(str).str.strip()
    df = df[df["person"].isin(["1", "2", "3"])]
    df["number"] = df["number"].astype(str).str.strip().replace("", np.nan)

    df_with_int, interval_df = _mod_08a.adaptive_binning(df, min_poems=min_poems)

    def _agg(g):
        n = len(g)
        if n == 0:
            return pd.Series({"1sg": 0, "1pl": 0, "2": 0, "3sg": 0, "3pl": 0, "n_tokens": 0})
        p1 = (g["person"] == "1") | (g["person"].astype(str) == "1")
        p2 = (g["person"] == "2") | (g["person"].astype(str) == "2")
        p3 = (g["person"] == "3") | (g["person"].astype(str) == "3")
        sing = (g["number"] == "Sing").fillna(False)
        plur = (g["number"] == "Plur").fillna(False)
        return pd.Series({
            "1sg": ((p1) & (sing)).sum() / n, "1pl": ((p1) & (plur)).sum() / n,
            "2": (p2).sum() / n, "3sg": ((p3) & (sing)).sum() / n, "3pl": ((p3) & (plur)).sum() / n,
            "n_tokens": n,
        })

    ts = (
        df_with_int.groupby("interval_id", group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    ts = ts.merge(
        interval_df[["interval_id", "start_date", "end_date", "n_poems"]],
        on="interval_id",
    )

    ts["duration_days"] = (ts["end_date"] - ts["start_date"]).dt.days + 1
    ts["time"] = np.arange(1, len(ts) + 1)
    ts["post_2014"] = (ts["start_date"] >= BREAK_2014).astype(int)
    ts["post_2022"] = (ts["start_date"] >= BREAK_2022).astype(int)
    ts["time_post2014"] = ts["time"] * ts["post_2014"]
    ts["time_post2022"] = ts["time"] * ts["post_2022"]

    ts["weight_A"] = ts["n_poems"]
    ts["weight_B"] = 1.0 / ts["duration_days"]
    ts["weight_C"] = ts["n_poems"] / np.sqrt(ts["duration_days"])

    ts = ts.sort_values("time").reset_index(drop=True)
    return ts, interval_df


def run_breakpoint_regression(
    ts: pd.DataFrame,
    y_col: str,
    weight_col: str = "weight_A",
) -> dict:
    if not STATSMODELS_AVAIL:
        return {"error": "statsmodels not installed"}

    df = ts[[y_col, "time", "post_2014", "post_2022", "time_post2014", "time_post2022", weight_col]].dropna()
    y = df[y_col].values
    w = df[weight_col].values
    n = len(df)

    result = {"y_col": y_col, "weight_scheme": weight_col}

    X0 = np.column_stack([np.ones(n), df["time"].values])
    m0 = WLS(y, X0, weights=w).fit()
    result["m0"] = {"params": m0.params, "rsquared": m0.rsquared, "aic": m0.aic, "bic": m0.bic}

    X1 = np.column_stack([
        np.ones(n),
        df["time"].values,
        df["post_2014"].values,
        df["time_post2014"].values,
    ])
    m1 = WLS(y, X1, weights=w).fit()
    result["m1"] = {"params": m1.params, "rsquared": m1.rsquared, "aic": m1.aic, "bic": m1.bic}

    X2 = np.column_stack([
        np.ones(n),
        df["time"].values,
        df["post_2014"].values,
        df["post_2022"].values,
        df["time_post2014"].values,
        df["time_post2022"].values,
    ])
    m2 = WLS(y, X2, weights=w).fit()
    result["m2"] = {
        "params": m2.params,
        "pvalues": m2.pvalues,
        "bse": m2.bse,
        "rsquared": m2.rsquared,
        "aic": m2.aic,
        "bic": m2.bic,
    }

    f_stat = (m2.rsquared - m0.rsquared) / (1 - m2.rsquared) * (n - 6) / 4
    result["f_test_m0_vs_m2"] = {"f_stat": f_stat, "p_value": 1 - stats.f.cdf(f_stat, 4, n - 6)}

    return result


def build_regression_summary_table(
    ts: pd.DataFrame,
    weight_scheme: str = "A",
) -> pd.DataFrame:
    weight_col = f"weight_{weight_scheme}"
    rows = []
    for y_col in PRONOUN_COLS:
        res = run_breakpoint_regression(ts, y_col, weight_col)
        if "error" in res:
            continue
        m2 = res["m2"]
        pnames = ["const", "time", "post_2014", "post_2022", "time_post2014", "time_post2022"]
        bse = m2.get("bse", np.zeros(6))
        for i, pn in enumerate(pnames):
            if i < len(m2["params"]):
                rows.append({
                    "pronoun": y_col,
                    "coef": pn,
                    "est": m2["params"][i],
                    "se": bse[i] if i < len(bse) else np.nan,
                    "r2": m2["rsquared"],
                })
    return pd.DataFrame(rows)


def run_changepoint_detection(
    ts: pd.DataFrame,
    y_col: str,
    pen: float = None,
) -> list:
    if not RUPTURES_AVAIL:
        return []
    y = ts[y_col].values.reshape(-1, 1)
    y = (y - y.mean()) / (y.std() + 1e-8)
    n = len(ts)
    pen = pen or np.log(n) * 2
    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(y)
    cps = algo.predict(pen=pen)
    return [int(c) for c in cps[:-1] if 0 < c < n]


def compute_descriptive_by_period(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    ts["period_label"] = "pre_2014"
    ts.loc[ts["post_2014"] == 1, "period_label"] = "2014_2022"
    ts.loc[ts["post_2022"] == 1, "period_label"] = "post_2022"

    rows = []
    for pcol in PRONOUN_COLS:
        for period in ["pre_2014", "2014_2022", "post_2022"]:
            sub = ts[ts["period_label"] == period][pcol]
            if len(sub) > 0:
                rows.append({
                    "pronoun": pcol,
                    "period": period,
                    "mean": sub.mean(),
                    "std": sub.std(),
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


def plot_timeseries_with_breaks(
    ts: pd.DataFrame,
    out_path: Path,
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    t_2014 = ts.loc[ts["post_2014"] == 1, "time"].min() if (ts["post_2014"] == 1).any() else None
    t_2022 = ts.loc[ts["post_2022"] == 1, "time"].min() if (ts["post_2022"] == 1).any() else None

    for i, col in enumerate(PRONOUN_COLS):
        ax = axes[i]
        ax.plot(ts["time"], ts[col], "o-", markersize=4, alpha=0.8)
        if t_2014 is not None:
            ax.axvline(t_2014, color="coral", linestyle="--", alpha=0.7, label="2014-02")
        if t_2022 is not None:
            ax.axvline(t_2022, color="darkred", linestyle="--", alpha=0.7, label="2022-02")
        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("Pronoun proportions by interval (vertical lines: breakpoints)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_forest_plot(
    summary: pd.DataFrame,
    out_path: Path,
):
    eff = summary[(summary["coef"].isin(["post_2014", "post_2022"]))].copy()
    if len(eff) == 0:
        return
    eff["xerr"] = eff.get("se", 0.01) * 1.96  # 95% CI
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = 0
    y_labels = []
    x_vals = []
    x_errs = []
    for pcol in PRONOUN_COLS:
        sub = eff[eff["pronoun"] == pcol]
        for _, r in sub.iterrows():
            y_labels.append(f"{pcol}_{r['coef']}")
            x_vals.append(r["est"])
            x_errs.append(r["xerr"])
            y_pos += 1
    y_pos = np.arange(len(y_labels))
    ax.errorbar(x_vals, y_pos, xerr=x_errs, fmt="o", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.axvline(0, color="gray", linestyle="-")
    ax.set_xlabel("Effect size (95% CI)")
    ax.set_title("Forest plot: 2014 & 2022 level effects")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_bootstrap_ci(
    ts: pd.DataFrame,
    y_col: str,
    weight_col: str = "weight_A",
    n_boot: int = 500,
) -> dict:
    if not STATSMODELS_AVAIL:
        return {}
    n = len(ts)
    coefs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        sub = ts.iloc[idx]
        res = run_breakpoint_regression(sub, y_col, weight_col)
        if "error" not in res and "m2" in res:
            coefs.append(res["m2"]["params"])
    if not coefs:
        return {}
    coefs = np.array(coefs)
    return {
        "post_2014_ci": (np.percentile(coefs[:, 2], 2.5), np.percentile(coefs[:, 2], 97.5)),
        "post_2022_ci": (np.percentile(coefs[:, 3], 2.5), np.percentile(coefs[:, 3], 97.5)),
    }


def main():
    min_poems = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
    weight_scheme = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in "ABC" else "A"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("09 Weighted Breakpoint Regression")
    print("=" * 60)
    ws = WEIGHT_SCHEMES.get(weight_scheme, weight_scheme)
    print(f"min_poems={min_poems}, weight_scheme={ws} (A=poem_count, B=1/days, C=composite)")

    ts, intervals = load_and_prepare_data(min_poems=min_poems)
    ts = ts[ts["n_tokens"] >= 5].reset_index(drop=True)
    print(f"\nIntervals: {len(ts)}, range {ts['start_date'].min().date()} to {ts['end_date'].max().date()}")

    print("\n--- Data checks ---")
    print(f"  Proportions sum ≈ 1: {np.allclose(ts[PRONOUN_COLS].sum(axis=1), 1.0)}")
    print(f"  Intervals >= {min_poems} poems: {(ts['n_poems'] >= min_poems).all()}")

    desc = compute_descriptive_by_period(ts)
    desc.to_csv(OUTPUT_DIR / "table1_descriptive_by_period.csv", index=False)
    print(f"\n--- Table 1 saved: table1_descriptive_by_period.csv ---")

    print("\n--- Breakpoint regression (Model 2: double break) ---")
    full_results = []
    for y_col in PRONOUN_COLS:
        res = run_breakpoint_regression(ts, y_col, f"weight_{weight_scheme}")
        if "error" in res:
            print(f"  {y_col}: {res['error']}")
            continue
        m2 = res["m2"]
        pvals = m2.get("pvalues", [np.nan] * 6)
        full_results.append({
            "pronoun": y_col,
            "beta_2014_level": m2["params"][2],
            "p_2014_level": pvals[2] if len(pvals) > 2 else np.nan,
            "beta_2022_level": m2["params"][3],
            "p_2022_level": pvals[3] if len(pvals) > 3 else np.nan,
            "beta_2014_slope": m2["params"][4],
            "beta_2022_slope": m2["params"][5],
            "R2": m2["rsquared"],
            "AIC": m2["aic"],
            "BIC": m2["bic"],
        })
        print(f"  {y_col}: R2={m2['rsquared']:.3f}, b2014={m2['params'][2]:.4f}, b2022={m2['params'][3]:.4f}")

    reg_df = pd.DataFrame(full_results)
    reg_df.to_csv(OUTPUT_DIR / "table2_breakpoint_regression.csv", index=False)
    print("--- Table 2 saved: table2_breakpoint_regression.csv ---")

    res_3sg = run_breakpoint_regression(ts, "3sg", f"weight_{weight_scheme}")
    if "error" not in res_3sg:
        m0, m1, m2 = res_3sg["m0"], res_3sg["m1"], res_3sg["m2"]
        print("\n--- Model comparison (3sg): ---")
        print(f"  M0 (baseline): AIC={m0['aic']:.1f}, BIC={m0['bic']:.1f}")
        print(f"  M1 (2014 only): AIC={m1['aic']:.1f}, BIC={m1['bic']:.1f}")
        print(f"  M2 (2014+2022): AIC={m2['aic']:.1f}, BIC={m2['bic']:.1f}")
        if "f_test_m0_vs_m2" in res_3sg:
            ft = res_3sg["f_test_m0_vs_m2"]
            print(f"  F-test M0 vs M2: F={ft['f_stat']:.2f}, p={ft['p_value']:.4f}")

    if RUPTURES_AVAIL:
        print("\n--- Change point detection (PELT) ---")
        cp_rows = []
        for y_col in PRONOUN_COLS:
            cps = run_changepoint_detection(ts, y_col)
            for cp in cps:
                date_str = str(ts.iloc[cp]["start_date"].date()) if cp < len(ts) else ""
                cp_rows.append({"pronoun": y_col, "interval_id": cp, "date": date_str})
        cp_df = pd.DataFrame(cp_rows)
        cp_df.to_csv(OUTPUT_DIR / "table3_changepoints_detected.csv", index=False)
        print(f"  Detected change points: {len(cp_df)} total")
        print(f"\n--- Table 3 saved: table3_changepoints_detected.csv ---")

    plot_timeseries_with_breaks(ts, OUTPUT_DIR / "fig1_timeseries_with_breaks.png")
    print("\n--- Fig 1 saved: fig1_timeseries_with_breaks.png ---")

    summary = build_regression_summary_table(ts, weight_scheme)
    if len(summary) > 0:
        plot_forest_plot(summary, OUTPUT_DIR / "fig2_forest_plot.png")
        print("--- Fig 2 saved: fig2_forest_plot.png ---")

    if STATSMODELS_AVAIL:
        boot = run_bootstrap_ci(ts, "3sg", f"weight_{weight_scheme}", n_boot=300)
        if boot:
            a, b = boot["post_2014_ci"]
            c, d = boot["post_2022_ci"]
            print(f"\n--- Bootstrap 95% CI (3sg): post_2014=({a:.3f},{b:.3f}), post_2022=({c:.3f},{d:.3f})")

    ts.to_csv(OUTPUT_DIR / "prepared_timeseries.csv", index=False)
    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
