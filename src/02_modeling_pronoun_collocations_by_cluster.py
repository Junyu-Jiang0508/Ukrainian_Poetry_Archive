"""Author-cluster-conditional collocates, building on the dep-parsed pipeline.

RQ2 (hierarchical NB random-slope) identifies two author clusters on the 1pl
cell of the pooled stratum:

* ``mobilized`` — the highest-1pl-posterior poets who are also documented as
  mobilized: Mitrov, Chornohuz, Zharikova. (Kiva also has a high 1pl posterior
  but is not mobilized, so she falls to the ``other`` control.)
* ``retreat`` — bottom-shift / relocated poets: Babkina, Andrusiak, Falkovych,
  and Khersonsky (relocated to Italy).

For each cluster (and a control: ``other`` = all roster authors not in
either), we re-aggregate the per-token dependency-parsed pronoun contexts
emitted by ``02_modeling_pronoun_collocations.py`` and compute the same
log-Dice, PMI, and ΔPMI as the corpus-wide pass — but conditioned on the
cluster.

Inputs:
- ``outputs/02_modeling_pronoun_collocations/pronoun_contexts_long.csv``
  (one row per pronoun token, with ``author`` column).

Output:
- ``outputs/02_modeling_pronoun_collocations_by_cluster/cluster_deltapmi_1pl.csv``
- ``outputs/02_modeling_pronoun_collocations_by_cluster/top_movers_1pl_by_cluster.md``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "outputs" / "02_modeling_pronoun_collocations" / "pronoun_contexts_long.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_collocations_by_cluster"

# Clusters for the RQ2-localised 1pl collocation table. These are the RQ2 1pl
# posterior-extreme clusters, with the covariate correction applied:
#   * Kiva has a high 1pl posterior but is NOT mobilized, so she moves to 'other'.
#   * Khersonsky relocated to Italy, so he moves to 'retreat'.
# The top ('mobilized') cluster is therefore the high-1pl-posterior poets who are
# also documented as mobilized: Mitrov, Chornohuz, Zharikova.
MOBILIZED = {"Mitrov", "Chornohuz", "Zharikova"}
RETREAT = {"Babkina", "Andrusiak", "Falkovych", "Khersonsky"}


def _cluster_of(author: str) -> str:
    a = (author or "").lower()
    for name in MOBILIZED:
        if name.lower() in a:
            return "mobilized"
    for name in RETREAT:
        if name.lower() in a:
            return "retreat"
    return "other"


def _delta_pmi(parses: pd.DataFrame, cluster: str, cell: str = "1pl") -> pd.DataFrame:
    sub = parses.loc[parses["cluster"].eq(cluster) & parses["cell"].eq(cell)].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = (
        sub.groupby(["deprel", "head_lemma", "period"])
        .size()
        .reset_index(name="cooccurrence")
        .pivot_table(
            index=["deprel", "head_lemma"],
            columns="period",
            values="cooccurrence",
            fill_value=0,
        )
    )
    for p in ("2014_2021", "post_2022"):
        if p not in pivot.columns:
            pivot[p] = 0
    pivot = pivot[["2014_2021", "post_2022"]].reset_index()
    pivot.columns.name = None
    pivot.columns = ["deprel", "head_lemma", "cooc_p1", "cooc_p2"]

    n_p1 = int(sub.loc[sub["period"].eq("2014_2021")].shape[0])
    n_p2 = int(sub.loc[sub["period"].eq("post_2022")].shape[0])
    if n_p1 == 0 or n_p2 == 0:
        return pd.DataFrame()

    # Per-period PMI ≈ log2(P(head | cell, period) / P(head | corpus, period)).
    # We approximate P(head | corpus, period) by the head's share of all pronoun
    # contexts (any cell) in that period, restricted to the same cluster — this
    # keeps the within-cluster comparison fair.
    cluster_p1_total = int(parses.loc[parses["cluster"].eq(cluster) & parses["period"].eq("2014_2021")].shape[0])
    cluster_p2_total = int(parses.loc[parses["cluster"].eq(cluster) & parses["period"].eq("post_2022")].shape[0])
    head_total_p1 = (
        parses.loc[parses["cluster"].eq(cluster) & parses["period"].eq("2014_2021")]
        .groupby("head_lemma").size().rename("head_p1").reset_index()
    )
    head_total_p2 = (
        parses.loc[parses["cluster"].eq(cluster) & parses["period"].eq("post_2022")]
        .groupby("head_lemma").size().rename("head_p2").reset_index()
    )
    pivot = pivot.merge(head_total_p1, on="head_lemma", how="left").fillna({"head_p1": 0})
    pivot = pivot.merge(head_total_p2, on="head_lemma", how="left").fillna({"head_p2": 0})

    eps = 1e-9
    pivot["pmi_p1"] = np.log2(
        ((pivot["cooc_p1"] / max(n_p1, 1)) + eps)
        / ((pivot["head_p1"] / max(cluster_p1_total, 1)) + eps)
    )
    pivot["pmi_p2"] = np.log2(
        ((pivot["cooc_p2"] / max(n_p2, 1)) + eps)
        / ((pivot["head_p2"] / max(cluster_p2_total, 1)) + eps)
    )
    pivot["delta_pmi"] = pivot["pmi_p2"] - pivot["pmi_p1"]
    pivot["total_coocc"] = pivot["cooc_p1"] + pivot["cooc_p2"]
    pivot["cluster"] = cluster
    pivot["cell"] = cell
    return pivot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-coocc", type=int, default=3)
    parser.add_argument("--cell", type=str, default="1pl")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Run 02_modeling_pronoun_collocations.py first; missing {args.input}"
        )
    parses = pd.read_csv(args.input)
    log.info("Loaded %d pronoun-context rows", len(parses))

    parses["cluster"] = parses["author"].astype(str).map(_cluster_of)
    cluster_counts = parses.groupby(["cluster", "period", "cell"]).size().reset_index(name="n_tokens")
    cluster_counts.to_csv(out_dir / "cluster_token_counts.csv", index=False)
    log.info(
        "Cluster sizes (1pl tokens): %s",
        parses.loc[parses["cell"].eq(args.cell)].groupby(["cluster", "period"]).size().to_dict(),
    )

    frames = []
    for cluster in ("mobilized", "retreat", "other"):
        f = _delta_pmi(parses, cluster, cell=args.cell)
        if not f.empty:
            frames.append(f)
    if not frames:
        log.warning("No cluster results; aborting.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.loc[all_df["total_coocc"] >= args.min_coocc].copy()
    all_df.to_csv(out_dir / f"cluster_deltapmi_{args.cell}.csv", index=False)

    md = [f"# Cluster-conditional ΔPMI on the {args.cell} cell\n"]
    md.append(
        "Author clusters from RQ2 hierarchical NB posterior (1pl random slope):\n"
        f"- **mobilized**: {', '.join(sorted(MOBILIZED))} (post-2022 civic-military realignment)\n"
        f"- **retreat**:   {', '.join(sorted(RETREAT))} (post-2022 evacuation/curtailed posting)\n"
        "- **other**:    all remaining roster authors (control)\n\n"
    )
    for cluster in ("mobilized", "retreat", "other"):
        sub = all_df.loc[all_df["cluster"].eq(cluster)].copy()
        if sub.empty:
            continue
        md.append(f"## cluster = {cluster}\n")
        for direction, sort_asc, label in (
            ("post_2022_aff", False, "post-2022 affinity (top ΔPMI > 0)"),
            ("pre_2022_aff", True, "pre-2022 affinity (top ΔPMI < 0)"),
        ):
            md.append(f"\n**{label}**\n\n")
            ranked = sub.sort_values("delta_pmi", ascending=sort_asc).head(args.top_n)
            show = ranked[
                ["deprel", "head_lemma", "delta_pmi", "cooc_p1", "cooc_p2"]
            ].copy()
            md.append(show.to_markdown(index=False, floatfmt=".3f"))
            md.append("\n")
    (out_dir / f"top_movers_{args.cell}_by_cluster.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    log.info("Wrote cluster-conditional outputs to %s", out_dir)


if __name__ == "__main__":
    main()
