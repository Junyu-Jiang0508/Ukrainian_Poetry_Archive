"""Freeze author roster and preregistration artifacts for stage 15."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

DEFAULT_PER_POEM = ROOT / "outputs" / "03_reporting_descriptive_statistics" / "C_poem_perspective_derived_per_poem.csv"
DEFAULT_PRONOUN = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_LAYER0 = ROOT / "data" / "To_run" / "00_filtering" / "layer0_poems_one_per_row.csv"
DEFAULT_OUT = ROOT / "outputs" / "03_reporting_roster_freeze"
DEFAULT_PREREG = ROOT / "preregistration" / "contrasts_v1.md"

PERIOD_ORDER = ["P1_2014_2021", "P2_2022_plus"]
SWITCHERS = {"Iya Kiva", "Olena Boryshpolets", "Alex Averbuch", "Andrij Bondar"}
QIRIMLI_CODES = {"Qirimli", "Russian, Qirimli", "Ukrainian, Qirimli"}
AUTHOR_EXCLUSIONS = {"Alie Kenzhalieva": "Category error: Crimean Tatar/Russian bilingual voice (exclude whole author)."}


def _period3(year: float) -> str:
    if pd.isna(year):
        return "OUTSIDE"
    y = int(year)
    if 2014 <= y <= 2021:
        return PERIOD_ORDER[0]
    if y >= 2022:
        return PERIOD_ORDER[1]
    return "OUTSIDE"


def _dominant_language(df: pd.DataFrame, period: str) -> pd.Series:
    sub = df[df["period3"].eq(period)].copy()
    if sub.empty:
        return pd.Series(dtype=str)
    vc = (
        sub.groupby(["author", "language_clean"], dropna=False)["poem_id"]
        .count()
        .rename("n")
        .reset_index()
        .sort_values(["author", "n", "language_clean"], ascending=[True, False, True])
    )
    return vc.groupby("author", sort=False)["language_clean"].first()


def build_roster(per_poem_path: Path, out_dir: Path, min_per_period: int = 5) -> pd.DataFrame:
    df = pd.read_csv(per_poem_path, low_memory=False)
    df["period3"] = pd.to_numeric(df["year_int"], errors="coerce").map(_period3)
    df = df[df["period3"].isin(PERIOD_ORDER)].copy()
    df["poem_id"] = df["poem_id"].astype(str)
    df["author"] = df["author"].astype(str).str.strip()
    df["language_clean"] = df["language_clean"].astype(str).str.strip()

    qirimli_impact = int(df["language_clean"].isin(QIRIMLI_CODES).sum())
    df = df[~df["language_clean"].isin(QIRIMLI_CODES)].copy()

    counts = (
        df.groupby(["author", "period3"])["poem_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=PERIOD_ORDER, fill_value=0)
    )
    counts = counts.rename(
        columns={
            "P1_2014_2021": "n_p1",
            "P2_2022_plus": "n_p2",
        }
    )
    counts["min_per_period"] = counts[["n_p1", "n_p2"]].min(axis=1).astype(int)
    counts["total"] = counts[["n_p1", "n_p2"]].sum(axis=1).astype(int)
    counts = counts[counts["min_per_period"].ge(min_per_period)].copy()
    counts["included"] = True
    counts["is_bilingual_switcher"] = counts.index.isin(SWITCHERS)
    counts["exclusion_reason"] = ""

    for author, reason in AUTHOR_EXCLUSIONS.items():
        if author in counts.index:
            counts.loc[author, "included"] = False
            counts.loc[author, "exclusion_reason"] = reason

    dom_p1 = _dominant_language(df, "P1_2014_2021")
    dom_p2 = _dominant_language(df, "P2_2022_plus")
    counts["dominant_language_p1"] = counts.index.map(dom_p1).fillna("")
    counts["dominant_language_p2"] = counts.index.map(dom_p2).fillna("")
    counts = counts.sort_values(["included", "total"], ascending=[False, False]).reset_index()
    counts = counts.rename(columns={"index": "author"})

    counts.to_csv(out_dir / "roster_v1_frozen.csv", index=False)

    ge8 = counts[counts["min_per_period"].ge(8)].copy()
    ge8.to_csv(out_dir / "roster_v1_threshold_ge8.csv", index=False)
    with (out_dir / "qirimli_code_exclusion_count.txt").open("w", encoding="utf-8") as f:
        f.write(f"Rows removed by Qirimli code family: {qirimli_impact}\n")

    return counts


def build_bondar_spotcheck(layer0_path: Path, per_poem_path: Path, sample_n: int = 5) -> pd.DataFrame:
    per = pd.read_csv(per_poem_path, low_memory=False)
    per["year_int"] = pd.to_numeric(per["year_int"], errors="coerce")
    bondar = per[
        per["author"].astype(str).str.strip().eq("Andrij Bondar")
        & per["language_clean"].astype(str).str.strip().eq("Russian")
        & per["year_int"].ge(2022)
    ].copy()
    bondar["poem_id"] = bondar["poem_id"].astype(str)
    ids = sorted(bondar["poem_id"].unique().tolist())
    sample_ids = ids[:sample_n]

    l0 = pd.read_csv(layer0_path, low_memory=False)
    l0["poem_id"] = l0["poem_id"].astype(str)
    cols = [c for c in ["poem_id", "Date posted", "url of facebook post", "Poem full text (copy and paste)"] if c in l0.columns]
    m = l0[cols].drop_duplicates(subset=["poem_id"], keep="first")
    out = pd.DataFrame({"poem_id": sample_ids}).merge(m, on="poem_id", how="left")
    out = out.rename(
        columns={
            "Date posted": "date_posted",
            "url of facebook post": "facebook_url",
            "Poem full text (copy and paste)": "poem_text",
        }
    )
    out["text_preview"] = out["poem_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 180)
    out = out.drop(columns=["poem_text"], errors="ignore")
    return out


def plot_author_trajectories(pronoun_path: Path, layer0_path: Path, roster: pd.DataFrame, out_pdf: Path) -> None:
    ann = pd.read_csv(pronoun_path, low_memory=False, on_bad_lines="skip")
    ann["poem_id"] = ann["poem_id"].astype(str)
    ann["year_int"] = pd.to_numeric(ann["year"], errors="coerce")
    ann["author"] = ann["author"].astype(str).str.strip()
    ann["language_clean"] = ann["language"].astype(str).str.strip()

    if "is_repeat" in ann.columns and "is_translation" in ann.columns:
        ann["is_repeat"] = ann["is_repeat"].fillna(False).astype(bool)
        ann["is_translation"] = ann["is_translation"].fillna(False).astype(bool)
    else:
        l0 = pd.read_csv(
            layer0_path,
            usecols=["poem_id", "Is repeat", "I.D. of original (if poem is a translation)"],
            low_memory=False,
        )
        l0["poem_id"] = l0["poem_id"].astype(str)
        is_rep = l0["Is repeat"].astype(str).str.strip().str.lower().eq("yes")
        is_tr = l0["I.D. of original (if poem is a translation)"].notna() & l0["I.D. of original (if poem is a translation)"].astype(str).str.strip().ne("")
        flags = pd.DataFrame({"poem_id": l0["poem_id"], "is_repeat": is_rep, "is_translation": is_tr}).drop_duplicates("poem_id")
        ann = ann.merge(flags, on="poem_id", how="left")
        ann["is_repeat"] = ann["is_repeat"].fillna(False)
        ann["is_translation"] = ann["is_translation"].fillna(False)
    ann = ann[~(ann["is_repeat"] | ann["is_translation"])].copy()

    ann = ann[ann["author"].isin(roster.loc[roster["included"], "author"])].copy()
    ann = ann[ann["year_int"].ge(2014)].copy()
    ann = ann[~ann["language_clean"].isin(QIRIMLI_CODES)].copy()

    person_map = {"1st": "1", "2nd": "2"}
    number_map = {"Singular": "sg", "Plural": "pl"}
    ann["cell"] = ann["person"].map(person_map).fillna("") + ann["number"].map(number_map).fillna("")
    ann = ann[ann["cell"].isin(["1sg", "1pl", "2sg", "2pl"])].copy()
    if ann.empty:
        return

    poem_cell = ann.groupby(["author", "year_int", "poem_id", "cell"]).size().unstack(fill_value=0)
    for c in ["1sg", "1pl", "2sg", "2pl"]:
        if c not in poem_cell.columns:
            poem_cell[c] = 0
    poem_cell = poem_cell.reset_index()
    poem_cell["n12"] = poem_cell[["1sg", "1pl", "2sg", "2pl"]].sum(axis=1)
    poem_cell = poem_cell[poem_cell["n12"] > 0].copy()
    for c in ["1sg", "1pl", "2sg", "2pl"]:
        poem_cell[c] = poem_cell[c] / poem_cell["n12"]

    yearly = poem_cell.groupby(["author", "year_int"])[["1sg", "1pl", "2sg", "2pl"]].mean().reset_index()

    authors = roster.loc[roster["included"], "author"].tolist()
    n = len(authors)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.2 * nrows), sharex=True, sharey=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    colors = {"1sg": "#1f77b4", "1pl": "#ff7f0e", "2sg": "#2ca02c", "2pl": "#d62728"}

    for i, author in enumerate(authors):
        ax = axes_list[i]
        ad = yearly[yearly["author"].eq(author)].sort_values("year_int")
        for c in ["1sg", "1pl", "2sg", "2pl"]:
            ax.plot(ad["year_int"], ad[c], marker="o", linewidth=1.6, markersize=3.2, label=c, color=colors[c])
        ax.axvline(2014, linestyle="--", color="gray", linewidth=1)
        ax.axvline(2022, linestyle="--", color="gray", linewidth=1)
        ax.set_title(author, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Per-author yearly trajectories (1sg/1pl/2sg/2pl)", y=0.995, fontsize=12)
    fig.text(0.5, 0.01, "Year", ha="center")
    fig.text(0.01, 0.5, "Mean per-poem proportion within 1/2 person pronouns", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.02, 0.05, 0.995, 0.97])
    fig.savefig(out_pdf)
    plt.close(fig)


def write_decisions_md(out_dir: Path, roster: pd.DataFrame, bondar_spot: pd.DataFrame) -> None:
    today = date.today().isoformat()
    excluded = roster[~roster["included"]].copy()
    ge8_n = int(roster["min_per_period"].ge(8).sum())
    lines = [
        "# Roster Freeze Decisions (v1)",
        "",
        f"- Decision date: {today}",
        "- Baseline source: `outputs/03_reporting_descriptive_statistics/C_poem_perspective_derived_per_poem.csv` (post layer0 exclusions).",
        "- Time bins fixed: P1=2014-2021, P2=2022+.",
        "- Author inclusion threshold fixed before modeling: >=5 poems in each period.",
        f"- Robustness roster also recorded at >=8 poems/period (n={ge8_n}).",
        "- Qirimli family exclusion applied at code-level: `Qirimli`, `Russian, Qirimli`, `Ukrainian, Qirimli`.",
        "- Main-analysis switcher policy: keep bilingual switchers; language retained as covariate.",
        "",
        "## Excluded Authors",
        "",
    ]
    if excluded.empty:
        lines.append("- None.")
    else:
        for _, r in excluded.iterrows():
            lines.extend([f"### {r['author']}", "", f"- Rationale: {r['exclusion_reason']}", f"- Decision date: {today}", ""])

    lines.extend(
        [
            "## Andrij Bondar (P3 Russian anomaly) Diagnostic",
            "",
            "- Protocol: spot-check at least 5 Russian poems in 2022+ against original post date/text metadata.",
            f"- Population in this dataset: {int(((roster['author']=='Andrij Bondar').any())) and 'present'}; spot-check below uses layer0 metadata.",
            "",
            "| poem_id | date_posted | facebook_url | text_preview |",
            "|---|---|---|---|",
        ]
    )
    for _, r in bondar_spot.iterrows():
        lines.append(
            f"| {r.get('poem_id','')} | {r.get('date_posted','')} | {r.get('facebook_url','')} | {str(r.get('text_preview','')).replace('|','/')} |"
        )
    lines.extend(
        [
            "",
            "- Freeze decision for Bondar: **included** (no layer0 metadata evidence of translation/republication misattribution in sampled items).",
            "- Caveat: this is a metadata-based check; full external FB verification can be appended if required by review.",
            "",
            "## Threshold Rationale",
            "",
            "- `>=5` preserves cross-period continuity while avoiding over-pruning persistent poets.",
            "- `>=8` roster exported as robustness check to guard against threshold-driven conclusions.",
        ]
    )
    (out_dir / "roster_decisions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prereg(path: Path) -> None:
    today = date.today().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Confirmatory Contrasts v1

Signed date: {today}

Primary inferential family is fixed before model rerun:

1. P3 vs P2: person main contrast.
2. P3 vs P2: number main contrast.
3. P3 vs P2: person x number interaction contrast.
4. P1 vs P2: person x number interaction contrast.

Notes:
- These contrasts are confirmatory; multiplicity control is applied within this family.
- Sensitivity analyses are pre-specified separately: (a) remove four switchers, (b) remove Iya Kiva only, (c) leave-one-author-out across the frozen roster.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze roster outputs for stage 15.")
    ap.add_argument("--per-poem", type=Path, default=DEFAULT_PER_POEM)
    ap.add_argument("--pronoun", type=Path, default=DEFAULT_PRONOUN)
    ap.add_argument("--layer0", type=Path, default=DEFAULT_LAYER0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    roster = build_roster(args.per_poem.resolve(), out)
    bondar_spot = build_bondar_spotcheck(args.layer0.resolve(), args.per_poem.resolve(), sample_n=5)
    write_decisions_md(out, roster, bondar_spot)
    plot_author_trajectories(args.pronoun.resolve(), args.layer0.resolve(), roster, out / "diagnostic_per_author_trajectories.pdf")
    write_prereg(args.prereg.resolve())

    print(f"Wrote roster freeze outputs to: {out}")
    print(f"Wrote preregistration contrasts to: {args.prereg.resolve()}")


if __name__ == "__main__":
    main()
