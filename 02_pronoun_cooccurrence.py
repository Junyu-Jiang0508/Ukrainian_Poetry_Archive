import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # pip install python-louvain
import plotly.graph_objects as go
import numpy as np

INPUT_PATH = "outputs/02_pronoun_cooccurrence/pronoun_cooccurrence.csv"
OUTPUT_DIR = "outputs/02_pronoun_cooccurrence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} co-occurrence edges")

threshold = df["count"].quantile(0.90)
df = df[df["count"] >= threshold]
print(f"Kept {len(df)} high-frequency edges (≥ {threshold:.1f})")

G = nx.Graph()
for _, r in df.iterrows():
    G.add_edge(r["pronoun"], r["word"], weight=r["count"])

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')
print(f"Detected {len(set(partition.values()))} communities")

sizes = [max(50, min(G.degree(n)*80, 1000)) for n in G.nodes()]
colors = [partition[n] for n in G.nodes()]

pos = nx.spring_layout(G, k=0.5, seed=42, iterations=200)

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=sizes, cmap=plt.cm.tab10, node_color=colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=9, font_family="Arial", font_color="black")
plt.title("Pronoun–Word Co-Occurrence Network (Simplified)")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pronoun_network_advanced.png"), dpi=400)
plt.show()

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color="#aaa"),
    hoverinfo='none', mode='lines'
)

node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(partition[node])
    node_size.append(G.degree(node) * 6)
    node_text.append(f"{node} ({G.degree(node)})")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Turbo',
        color=node_color,
        size=node_size,
        line_width=0.5
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
    layout=go.Layout(
        title='Interactive Pronoun–Word Semantic Network',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
)
fig.write_html(os.path.join(OUTPUT_DIR, "pronoun_network_interactive.html"))
print(f"Saved interactive network to pronoun_network_interactive.html")
