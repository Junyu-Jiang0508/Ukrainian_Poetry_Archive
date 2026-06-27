"""Poem-level BERTopic topic model as a downstream covariate (P1-F).

We fit BERTopic on the corpus's full poem texts to obtain a topic label per
poem. The output is intended for use as a fixed-effect covariate in the
attention-allocation and absolute-salience models, so that period contrasts
can be adjusted for topic mix (war / love / lament / civic / etc.). Reviewer
critique anticipated: ``the post-2022 shift is just topic shift.'' Adjusting
for topic dispels that confound or sharpens it as a substantive finding.

Modeling
--------
We split the corpus by language (Ukrainian and Russian) because BERTopic's
c-TF-IDF stopword filtering and the visual cleanliness of the topic
representations both improve when each model sees one language. For
embeddings we use the multilingual sentence-transformer
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``: it covers
both Ukrainian and Russian and is much cheaper than ``paraphrase-xlm-r-*``
while losing little quality on short verse.

The clustering pipeline is BERTopic's default: UMAP $\to$ HDBSCAN. We pass an
explicit ``CountVectorizer`` with a small Cyrillic stoplist so the topic
representations are interpretable in their respective language.

Outputs (``outputs/02_modeling_topic_bertopic/``)
-------------------------------------------------
* ``poem_topic_assignments.csv``  — covariate-ready: ``poem_id, language, topic_id, topic_label``.
* ``topic_info_<language>.csv``   — BERTopic's per-topic info (count, name, representation).
* ``topic_top_words_<language>.csv`` — per-topic top-N words for transparent labelling.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import canonical_pronoun_annotation_csv, prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = canonical_pronoun_annotation_csv(ROOT)
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_topic_bertopic"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Same stoplists as P1-C / P1-E; we centralize copies rather than introduce a
# new utility because the BERTopic CountVectorizer wants a flat list.
STOPWORDS = {
    "Ukrainian": [
        "і", "та", "а", "але", "що", "як", "не", "на", "з", "у", "в", "до", "по",
        "за", "о", "є", "був", "була", "було", "були",
        "це", "цей", "ця", "ці", "то", "той", "те",
        "ж", "же", "б", "би", "ну", "ось", "от",
        "я", "ми", "ти", "ви", "він", "вона", "воно", "вони",
        "мене", "мені", "мною", "нас", "нам", "нами",
        "тебе", "тобі", "тобою", "вас", "вам", "вами",
    ],
    "Russian": [
        "и", "а", "но", "что", "как", "не", "на", "с", "у", "в", "до", "по",
        "за", "о", "об", "от", "из", "к",
        "это", "этот", "эта", "эти", "то", "тот", "та", "те",
        "был", "была", "было", "были",
        "же", "ли", "ну", "вот",
        "я", "мы", "ты", "вы", "он", "она", "оно", "они",
        "меня", "мне", "мной", "нас", "нам", "нами",
        "тебя", "тебе", "тобой", "вас", "вам", "вами",
    ],
}


def _build_poem_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate stanza texts back into poem-level documents."""
    dedup = df.drop_duplicates(["poem_id", "stanza_index"]).copy()
    dedup = dedup.loc[dedup["stanza_ukr"].notna()].copy()
    grouped = (
        dedup.sort_values(["poem_id", "stanza_index"])
        .groupby("poem_id", sort=False, as_index=False)
        .agg(
            language=("language", "first"),
            author=("author", "first"),
            temporal_period=("temporal_period", "first"),
            year=("year", "first"),
            text=("stanza_ukr", lambda g: "\n".join(map(str, g))),
        )
    )
    return grouped


def fit_bertopic_for_language(
    poem_texts: pd.DataFrame,
    language: str,
    *,
    embedding_model_name: str,
    out_dir: Path,
    seed: int = 20260519,
) -> pd.DataFrame:
    """Fit one BERTopic model per language. Returns a poem-level assignments table."""
    sub = poem_texts.loc[poem_texts["language"].astype(str).eq(language)].copy()
    if sub.empty or len(sub) < 30:
        log.warning("Too few %s poems (%d) to fit BERTopic — skipping.", language, len(sub))
        return pd.DataFrame()

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    docs = sub["text"].astype(str).tolist()
    log.info("Loading embedding model %s (%s, %d docs)", embedding_model_name, language, len(docs))
    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(docs, show_progress_bar=False, batch_size=32)

    # Deterministic UMAP + HDBSCAN. Defaults tuned for ~500–2000 docs.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(8, len(docs) // 60),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words=STOPWORDS.get(language, []),
        ngram_range=(1, 2),
        token_pattern=r"(?u)[А-Яа-яҐґЄєІіЇїA-Za-z’']{2,}",
        min_df=3,
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    sub["topic_id"] = topics
    info = topic_model.get_topic_info()
    info.to_csv(out_dir / f"topic_info_{language}.csv", index=False)

    # Per-topic top words.
    rows: list[dict[str, object]] = []
    for tid in info["Topic"].tolist():
        words = topic_model.get_topic(tid)
        if not words:
            continue
        for rank, (word, weight) in enumerate(words[:10], start=1):
            rows.append(
                {
                    "language": language,
                    "topic_id": int(tid),
                    "rank": rank,
                    "word": word,
                    "weight": float(weight),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / f"topic_top_words_{language}.csv", index=False)

    # Compose a short topic label from the top-3 words for the CSV's convenience.
    top_words_by_topic: dict[int, list[str]] = {}
    for tid in info["Topic"].tolist():
        words = topic_model.get_topic(tid)
        if not words:
            continue
        top_words_by_topic[int(tid)] = [w for w, _ in words[:3]]
    sub["topic_label"] = sub["topic_id"].map(
        lambda tid: " | ".join(top_words_by_topic.get(int(tid), ["?"]))
    )
    return sub[
        [
            "poem_id",
            "language",
            "author",
            "temporal_period",
            "year",
            "topic_id",
            "topic_label",
        ]
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Poem-level BERTopic for covariate use (P1-F).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False, on_bad_lines="skip")
    poem_texts = _build_poem_texts(df)

    frames: list[pd.DataFrame] = []
    for language in ("Ukrainian", "Russian"):
        assignments = fit_bertopic_for_language(
            poem_texts, language, embedding_model_name=args.embedding_model, out_dir=out_dir
        )
        if not assignments.empty:
            frames.append(assignments)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out_dir / "poem_topic_assignments.csv", index=False)
        log.info("Wrote topic assignments for %d poems to %s", len(combined), out_dir)
    else:
        log.warning("No topic assignments produced.")


if __name__ == "__main__":
    main()
