"""Three-period pronoun–concept co-occurrence networks."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
    NX_AVAIL = True
except ImportError:
    NX_AVAIL = False

try:
    import community as community_louvain
    LOUVAIN_AVAIL = True
except ImportError:
    LOUVAIN_AVAIL = False

import os
os.chdir(Path(__file__).resolve().parent.parent)

INPUT_CSV = Path("outputs/02_pronoun_cooccurrence/pronoun_cooccurrence_with_date.csv")
OUTPUT_DIR = Path("outputs/17_temporal_network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BREAK_2014 = pd.Timestamp("2014-02-01")
BREAK_2022 = pd.Timestamp("2022-02-01")
PERIODS = ["pre_2014", "2014_2021", "post_2022"]

CORE_PRONOUNS = {"я", "ми", "ти", "він", "вона", "вони", "ви",
                 "мене", "нас", "тебе", "нам", "мені"}


def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    conditions = [
        df["date"] < BREAK_2014,
        (df["date"] >= BREAK_2014) & (df["date"] < BREAK_2022),
        df["date"] >= BREAK_2022,
    ]
    df["period"] = np.select(conditions, PERIODS, default="unknown")
    return {p: df[df["period"] == p] for p in PERIODS}


def build_network(df_period: pd.DataFrame, top_n=100):
    """Build bipartite pronoun-word network from co-occurrence data."""
    if not NX_AVAIL:
        return None

    agg = df_period.groupby(["pronoun", "word"])["count"].sum().reset_index()
    agg = agg.nlargest(top_n, "count")

    G = nx.Graph()
    for _, row in agg.iterrows():
        pron, word, cnt = row["pronoun"], row["word"], row["count"]
        G.add_node(pron, bipartite=0, node_type="pronoun")
        G.add_node(word, bipartite=1, node_type="word")
        G.add_edge(pron, word, weight=cnt)
    return G


def network_metrics(G, period: str):
    if G is None or len(G) == 0:
        return {}

    deg_cent = nx.degree_centrality(G)
    btw_cent = nx.betweenness_centrality(G, weight="weight")

    pronoun_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "pronoun"]
    word_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "word"]

    metrics = {
        "period": period,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_pronouns": len(pronoun_nodes),
        "n_words": len(word_nodes),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
    }

    if LOUVAIN_AVAIL:
        partition = community_louvain.best_partition(G)
        metrics["n_communities"] = len(set(partition.values()))
        metrics["modularity"] = community_louvain.modularity(partition, G)
    else:
        partition = None

    top_deg = sorted(deg_cent.items(), key=lambda x: -x[1])[:10]
    top_btw = sorted(btw_cent.items(), key=lambda x: -x[1])[:10]

    node_data = []
    for node in G.nodes():
        node_data.append({
            "node": node,
            "type": G.nodes[node].get("node_type", "unknown"),
            "degree": G.degree(node),
            "degree_centrality": deg_cent.get(node, 0),
            "betweenness_centrality": btw_cent.get(node, 0),
            "community": partition.get(node, -1) if partition else -1,
            "weighted_degree": sum(d.get("weight", 1) for _, _, d in G.edges(node, data=True)),
        })
    pd.DataFrame(node_data).to_csv(
        OUTPUT_DIR / f"node_metrics_{period}.csv", index=False
    )

    return metrics


def visualize_networks(networks: dict):
    if not NX_AVAIL:
        print("networkx not available, skipping visualization")
        return

    available = [(p, G) for p, G in networks.items() if G is not None and len(G) > 0]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 8))
    if len(available) == 1:
        axes = [axes]

    for ax, (period, G) in zip(axes, available):
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        pronoun_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "pronoun"]
        word_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "word"]

        nx.draw_networkx_nodes(G, pos, nodelist=pronoun_nodes, node_color="red",
                               node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color="skyblue",
                               node_size=100, alpha=0.6, ax=ax)

        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [w / max_w * 3 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)

        labels = {n: n for n in pronoun_nodes}
        top_words = sorted(word_nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)[:15]
        labels.update({n: n for n in top_words})
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

        ax.set_title(f"{period}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        ax.axis("off")

    fig.suptitle("Pronoun-Concept Co-occurrence Networks by Period", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "network_comparison.png", dpi=150)
    plt.close(fig)


def compare_pronoun_centrality(networks: dict):
    """Compare how pronoun centrality shifts across periods."""
    rows = []
    for period, G in networks.items():
        if G is None:
            continue
        deg_cent = nx.degree_centrality(G)
        btw_cent = nx.betweenness_centrality(G, weight="weight")
        for node, d in G.nodes(data=True):
            if d.get("node_type") == "pronoun":
                rows.append({
                    "period": period,
                    "pronoun": node,
                    "degree_centrality": deg_cent.get(node, 0),
                    "betweenness_centrality": btw_cent.get(node, 0),
                    "weighted_degree": sum(
                        dd.get("weight", 1) for _, _, dd in G.edges(node, data=True)
                    ),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "pronoun_centrality_comparison.csv", index=False)

    if len(df) > 0:
        pivot_deg = df.pivot_table(index="pronoun", columns="period",
                                   values="degree_centrality", fill_value=0)
        pivot_deg = pivot_deg.reindex(columns=[p for p in PERIODS if p in pivot_deg.columns])

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_deg.plot(kind="bar", ax=ax)
        ax.set_title("Pronoun Degree Centrality by Period")
        ax.set_ylabel("Degree Centrality")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "pronoun_centrality_bar.png", dpi=150)
        plt.close(fig)

    return df


def main():
    if not NX_AVAIL:
        print("networkx not installed. Install with: pip install networkx")
        return

    period_data = load_and_split()

    networks = {}
    all_metrics = []
    for period in PERIODS:
        df_p = period_data.get(period, pd.DataFrame())
        G = build_network(df_p, top_n=150)
        networks[period] = G
        m = network_metrics(G, period)
        all_metrics.append(m)
        print(f"{period}: {m.get('n_nodes', 0)} nodes, {m.get('n_edges', 0)} edges, "
              f"modularity={m.get('modularity', 'N/A')}")

    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / "network_metrics_summary.csv", index=False)
    visualize_networks(networks)
    compare_pronoun_centrality(networks)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
