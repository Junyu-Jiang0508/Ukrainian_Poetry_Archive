"""Pronoun co-occurrence ego networks by cell × period (P1-E, rebuilt).

For each focal pronoun (ми/я/ти/ви in Ukrainian; мы/я/ты/вы in Russian) and
each period (P1: 2014--2021, P2: post-2022), build a stanza-window
co-occurrence ego graph in which:

* nodes are the focal pronoun + its top-K log-Dice content-word collocates,
* edges connect the pronoun to each collocate (weighted by log-Dice),
* second-order edges connect collocates that co-occur within the same set of
  stanzas (also weighted by log-Dice, with a minimum frequency floor),

then export the graph as GraphML for Gephi and render a PNG with NetworkX's
spring layout.

This module supersedes the deleted ``02_pronoun_cooccurrence.py`` whose
output was a single corpus-wide table with no period or cell stratification
and whose semantic_space clusters labeled only fragmented word-stems
(``пам``, ``об``, ``запам``, …) that were not analytically usable.

Design choices
--------------
* Stanza is the co-occurrence window (matches the inference unit in 02a/02b).
* Tokenization is regex + a closed-class stoplist, the same as
  ``02_modeling_pronoun_semantic_drift.py``. We intentionally avoid Stanza
  parsing here because (a) it would duplicate the parse cost of P1-A/B for a
  primarily descriptive output, and (b) regex co-occurrence is what the
  collocation literature actually uses for ego-network visualization.
* log-Dice is reported instead of raw count to dampen the high-frequency bias
  that PMI introduces at small counts.

Outputs (``outputs/02_modeling_pronoun_cooccurrence/``)
-------------------------------------------------------
* ``ego_edges.csv``  — long table of (focal, neighbour, period, language,
  cooccurrence, log_dice).
* ``ego_graph_<language>_<focal>_<period>.graphml`` — GraphML files for Gephi.
* ``ego_graph_<language>_<focal>_<period>.png`` — quick-look figures.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = ROOT / "data" / "Annotated_GPT_rerun" / "pronoun_annotation.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_cooccurrence"

_TOKEN_RE = re.compile(r"[А-Яа-яҐґЄєІіЇїA-Za-z’']+", re.UNICODE)

# Identical to P1-C stoplists. Centralizing would be cleaner; the values are
# small and stable enough that duplication is acceptable.
STOPWORDS = {
    "Ukrainian": {
        "і", "та", "а", "але", "що", "як", "не", "на", "з", "у", "в", "до", "по",
        "за", "о", "є", "був", "була", "було", "були",
        "це", "цей", "ця", "ці", "то", "той", "та", "те",
        "ж", "же", "б", "би", "ну", "ось", "от",
    },
    "Russian": {
        "и", "а", "но", "что", "как", "не", "на", "с", "у", "в", "до", "по",
        "за", "о", "об", "от", "из", "к",
        "это", "этот", "эта", "эти", "то", "тот", "та", "те",
        "был", "была", "было", "были",
        "же", "ли", "ну", "вот",
    },
}

# Focal pronouns: surface forms we recognize as ``the pronoun'' for ego centring.
# The pronoun lemma is the dictionary form; matches collapse all paradigm
# forms into the same node.
FOCAL_FORMS: dict[str, dict[str, tuple[str, ...]]] = {
    "Ukrainian": {
        "ми": ("ми", "нас", "нам", "нами"),
        "я": ("я", "мене", "мені", "мною"),
        "ти": ("ти", "тебе", "тобі", "тобою"),
        "ви": ("ви", "вас", "вам", "вами"),
    },
    "Russian": {
        "мы": ("мы", "нас", "нам", "нами"),
        "я": ("я", "меня", "мне", "мной", "мною"),
        "ты": ("ты", "тебя", "тебе", "тобой", "тобою"),
        "вы": ("вы", "вас", "вам", "вами"),
    },
}

PERIODS = ("2014_2021", "post_2022")


def _tokenize(text: str, lang: str) -> list[str]:
    stop = STOPWORDS.get(lang, set())
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in stop and len(t) >= 2
    ]


def _canonicalize_pronouns(tokens: list[str], focal_map: dict[str, tuple[str, ...]]) -> list[str]:
    """Replace every surface form of a focal pronoun with its lemma."""
    form_to_lemma: dict[str, str] = {}
    for lemma, forms in focal_map.items():
        for f in forms:
            form_to_lemma[f] = lemma
    return [form_to_lemma.get(t, t) for t in tokens]


def _accumulate_cooccurrence(
    stanzas: list[list[str]],
) -> tuple[Counter, Counter, int]:
    """Return ``(cooc_counts, word_counts, n_stanzas)``.

    ``cooc_counts`` keys are ``(a, b)`` with ``a < b`` sorted by string order;
    counts reflect the number of stanzas in which both terms appear at least
    once. We use *set* co-occurrence (presence/absence per stanza) so that
    very long stanzas do not dominate.
    """
    cooc: Counter = Counter()
    words: Counter = Counter()
    n_stanzas = 0
    for tokens in stanzas:
        unique = set(tokens)
        if not unique:
            continue
        n_stanzas += 1
        for tok in unique:
            words[tok] += 1
        for a, b in combinations(sorted(unique), 2):
            cooc[(a, b)] += 1
    return cooc, words, n_stanzas


def _ego_edges_for_focal(
    focal: str,
    cooc: Counter,
    words: Counter,
    *,
    min_freq: int,
    top_k: int,
) -> list[dict[str, object]]:
    """Return the top-``top_k`` log-Dice collocates for ``focal``."""
    f_focal = words.get(focal, 0)
    if f_focal == 0:
        return []
    candidates: list[tuple[str, int, float]] = []
    for (a, b), c in cooc.items():
        if a == focal:
            other = b
        elif b == focal:
            other = a
        else:
            continue
        if c < min_freq:
            continue
        f_other = words.get(other, 0)
        if f_other == 0:
            continue
        log_dice = 14.0 + np.log2((2.0 * c) / (f_focal + f_other + 1e-12))
        candidates.append((other, c, float(log_dice)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [
        {"neighbour": other, "cooccurrence": c, "log_dice": d}
        for other, c, d in candidates[:top_k]
    ]


def _secondary_edges(
    ego_nodes: set[str],
    cooc: Counter,
    words: Counter,
    *,
    min_freq: int,
) -> list[dict[str, object]]:
    """Edges among the ego-network neighbours themselves."""
    out: list[dict[str, object]] = []
    for a, b in combinations(sorted(ego_nodes), 2):
        c = cooc.get((a, b), 0)
        if c < min_freq:
            continue
        log_dice = 14.0 + np.log2(
            (2.0 * c) / (words.get(a, 1) + words.get(b, 1) + 1e-12)
        )
        out.append({"source": a, "target": b, "cooccurrence": c, "log_dice": float(log_dice)})
    return out


def _render_graph_png(
    focal: str,
    ego_edges: list[dict[str, object]],
    second_edges: list[dict[str, object]],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    g.add_node(focal, kind="focal")
    for e in ego_edges:
        g.add_node(e["neighbour"], kind="neighbour")
        g.add_edge(focal, e["neighbour"], weight=float(e["log_dice"]))
    for e in second_edges:
        g.add_edge(e["source"], e["target"], weight=float(e["log_dice"]))

    pos = nx.spring_layout(g, seed=42, k=0.9 / max(1.0, np.sqrt(len(g.nodes))))
    fig, ax = plt.subplots(figsize=(8, 8))
    node_colors = ["crimson" if n == focal else "lightsteelblue" for n in g.nodes]
    node_sizes = [800 if n == focal else 350 for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    weights = np.array([d["weight"] for _, _, d in g.edges(data=True)])
    if len(weights) > 0:
        w_norm = (weights - weights.min()) / (weights.ptp() + 1e-9)
        edge_widths = 0.4 + w_norm * 2.5
    else:
        edge_widths = 1.0
    nx.draw_networkx_edges(g, pos, width=edge_widths, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_graphml(
    focal: str,
    ego_edges: list[dict[str, object]],
    second_edges: list[dict[str, object]],
    out_path: Path,
) -> None:
    import networkx as nx

    g = nx.Graph()
    g.add_node(focal, kind="focal")
    for e in ego_edges:
        g.add_node(e["neighbour"], kind="neighbour")
        g.add_edge(focal, e["neighbour"], weight=float(e["log_dice"]), cooccurrence=int(e["cooccurrence"]))
    for e in second_edges:
        g.add_edge(e["source"], e["target"], weight=float(e["log_dice"]), cooccurrence=int(e["cooccurrence"]))
    nx.write_graphml(g, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Pronoun co-occurrence ego networks (P1-E).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=30, help="Neighbours per focal.")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Minimum co-occurrence count for an edge to be drawn.",
    )
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False, on_bad_lines="skip")
    dedup = df.drop_duplicates(["poem_id", "stanza_index"]).copy()
    dedup = dedup.loc[dedup["stanza_ukr"].notna()].copy()

    all_edges: list[dict[str, object]] = []
    for language in ("Ukrainian", "Russian"):
        focal_map = FOCAL_FORMS[language]
        for period in PERIODS:
            sub = dedup.loc[
                (dedup["language"].astype(str).eq(language))
                & (dedup["temporal_period"].astype(str).eq(period))
            ]
            stanzas_tokenized = []
            for _, row in sub.iterrows():
                tokens = _tokenize(str(row["stanza_ukr"]), language)
                stanzas_tokenized.append(_canonicalize_pronouns(tokens, focal_map))
            cooc, words, n_stanzas = _accumulate_cooccurrence(stanzas_tokenized)
            log.info(
                "%s %s: %d stanzas, %d unique words, %d co-occurrence pairs",
                language,
                period,
                n_stanzas,
                len(words),
                len(cooc),
            )
            for focal in focal_map.keys():
                ego_edges = _ego_edges_for_focal(
                    focal, cooc, words, min_freq=args.min_freq, top_k=args.top_k
                )
                if not ego_edges:
                    continue
                ego_nodes = {e["neighbour"] for e in ego_edges} | {focal}
                second_edges = _secondary_edges(
                    ego_nodes, cooc, words, min_freq=args.min_freq
                )

                title = f"{language} · {focal} · {period} (n={n_stanzas} stanzas)"
                base_name = f"ego_graph_{language}_{focal}_{period}"
                _render_graph_png(focal, ego_edges, second_edges, out_dir / f"{base_name}.png", title)
                _write_graphml(focal, ego_edges, second_edges, out_dir / f"{base_name}.graphml")

                for e in ego_edges:
                    all_edges.append(
                        {
                            "language": language,
                            "focal": focal,
                            "period": period,
                            **e,
                        }
                    )

    if all_edges:
        pd.DataFrame(all_edges).to_csv(out_dir / "ego_edges.csv", index=False)
    log.info("Wrote co-occurrence outputs to %s", out_dir)


if __name__ == "__main__":
    main()
