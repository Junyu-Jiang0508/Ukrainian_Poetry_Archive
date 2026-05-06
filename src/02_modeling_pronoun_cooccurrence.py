import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain                              
import plotly.graph_objects as go
from pathlib import Path

from utils.workspace import prepare_analysis_environment
from utils.stage_io import read_csv_artifact, stage_output_dir

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)
DEFAULT_OUTPUT_DIR = stage_output_dir("02_modeling_pronoun_cooccurrence", root=ROOT)
DEFAULT_INPUT_PATH = DEFAULT_OUTPUT_DIR / "pronoun_cooccurrence.csv"


def build_network(edge_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["pronoun"], row["word"], weight=row["count"])
    return graph


def render_static_network(
    graph: nx.Graph,
    partition: dict[str, int],
    output_png: str,
) -> None:
    sizes = [max(50, min(graph.degree(node) * 80, 1000)) for node in graph.nodes()]
    colors = [partition[node] for node in graph.nodes()]
    positions = nx.spring_layout(graph, k=0.5, seed=42, iterations=200)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=sizes,
        cmap=plt.cm.tab10,
        node_color=colors,
        alpha=0.8,
    )
    nx.draw_networkx_edges(graph, positions, alpha=0.3)
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
        font_family="Arial",
        font_color="black",
    )
    plt.title("Pronoun–Word Co-Occurrence Network (Simplified)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=400)
    plt.show()


def render_interactive_network(
    graph: nx.Graph,
    partition: dict[str, int],
    output_html: str,
) -> None:
    positions = nx.spring_layout(graph, k=0.5, seed=42, iterations=200)
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#aaa"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(partition[node])
        node_size.append(graph.degree(node) * 6)
        node_text.append(f"{node} ({graph.degree(node)})")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Turbo",
            color=node_color,
            size=node_size,
            line_width=0.5,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Interactive Pronoun–Word Semantic Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )
    fig.write_html(output_html)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize pronoun-word co-occurrence network.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        default=0.90,
        help="Keep edges with count >= this quantile (default: 0.90).",
    )
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_artifact(input_path)
    print(f"Loaded {len(df)} co-occurrence edges")

    threshold = df["count"].quantile(args.quantile_threshold)
    filtered_df = df[df["count"] >= threshold]
    print(f"Kept {len(filtered_df)} high-frequency edges (>= {threshold:.1f})")

    graph = build_network(filtered_df)
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    partition = community_louvain.best_partition(graph, weight="weight")
    nx.set_node_attributes(graph, partition, "community")
    print(f"Detected {len(set(partition.values()))} communities")

    render_static_network(
        graph,
        partition,
        str(output_dir / "pronoun_network_advanced.png"),
    )
    render_interactive_network(
        graph,
        partition,
        str(output_dir / "pronoun_network_interactive.html"),
    )
    print("Saved network plots to output directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
