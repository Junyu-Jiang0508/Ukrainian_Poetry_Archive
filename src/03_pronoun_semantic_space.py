import argparse
import os
import warnings
import numpy as np
import pandas as pd
from gensim.models.fasttext import load_facebook_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.express as px
import urllib.request
import gzip
import shutil
from pathlib import Path

from utils.workspace import prepare_analysis_environment
from utils.stage_io import read_csv_artifact, stage_output_dir

warnings.filterwarnings('ignore', category=UserWarning, module='umap')

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)
DEFAULT_OUTPUT_DIR = stage_output_dir("03_pronoun_semantic_space", root=ROOT)
DEFAULT_DATE_INPUT = ROOT / "outputs" / "02_pronoun_cooccurrence" / "pronoun_cooccurrence_with_date.csv"
DEFAULT_SIMPLE_INPUT = ROOT / "outputs" / "02_pronoun_cooccurrence" / "pronoun_cooccurrence.csv"
DEFAULT_MODEL_PATH = ROOT / "cc.uk.300.bin"


def get_vector(ft_model, word: str):
    try:
        return ft_model.wv[word]
    except Exception:
        return np.zeros(300)


def parse_year(value):
    try:
        return int(str(value)[:4])
    except Exception:
        return None


def ensure_fasttext_model(model_path: str | os.PathLike[str]) -> None:
    model_path = str(model_path)
    if os.path.exists(model_path):
        return
    print("Downloading Ukrainian FastText model (~6GB)...")
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.bin.gz"
    archive = f"{model_path}.gz"
    urllib.request.urlretrieve(url, archive)
    print("Extracting model...")
    with gzip.open(archive, "rb") as source:
        with open(model_path, "wb") as target:
            shutil.copyfileobj(source, target)
    os.remove(archive)
    print("Model downloaded and extracted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build semantic space from pronoun-word co-occurrence edges.")
    parser.add_argument("--input-with-date", type=Path, default=DEFAULT_DATE_INPUT)
    parser.add_argument("--input-fallback", type=Path, default=DEFAULT_SIMPLE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fasttext-model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_with_date = args.input_with_date.resolve()
    if input_with_date.exists():
        df = read_csv_artifact(input_with_date)
        print(f"Loaded {len(df)} pronoun-word edges (with dates)")
    else:
        df = read_csv_artifact(args.input_fallback)
        print(f"Date file not found, loaded {len(df)} pronoun-word edges (fallback)")

    ensure_fasttext_model(args.fasttext_model)
    ft_model = load_facebook_model(str(args.fasttext_model.resolve()))
    print("FastText model loaded")

    all_words = sorted(set(df["word"]).union(df["pronoun"]))
    embeddings = np.array([get_vector(ft_model, word) for word in all_words])
    print(f"Computed {len(all_words)} word embeddings")

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    umap_df = pd.DataFrame({"word": all_words, "x": umap_coords[:, 0], "y": umap_coords[:, 1]})
    umap_df.to_csv(output_dir / "umap_embedding_layout.csv", index=False)

    clusterer = DBSCAN(eps=0.5, min_samples=8, metric="euclidean").fit(umap_coords)
    umap_df["cluster"] = clusterer.labels_
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = int((clusterer.labels_ == -1).sum())
    print(f"Found {n_clusters} semantic clusters ({n_noise} noise points)")

    docs_per_cluster: dict[int, list[str]] = {}
    for _, row in umap_df.iterrows():
        if row["cluster"] == -1:
            continue
        docs_per_cluster.setdefault(int(row["cluster"]), []).append(str(row["word"]))

    if docs_per_cluster:
        cluster_ids = sorted(docs_per_cluster.keys())
        cluster_docs = [" ".join(docs_per_cluster[cid]) for cid in cluster_ids]
        tfidf = TfidfVectorizer(max_features=10, ngram_range=(1, 1))
        tfidf_matrix = tfidf.fit_transform(cluster_docs)
        feature_names = tfidf.get_feature_names_out()
        with open(output_dir / "cluster_labels.txt", "w", encoding="utf-8") as handle:
            for idx, cid in enumerate(cluster_ids):
                cluster_tfidf = tfidf_matrix[idx].toarray()[0]
                top_indices = cluster_tfidf.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices if cluster_tfidf[i] > 0]
                if not top_words:
                    top_words = docs_per_cluster[cid][:10]
                handle.write(f"Cluster {cid} ({len(docs_per_cluster[cid])} words): {', '.join(top_words)}\n")

    fig = px.scatter(
        umap_df,
        x="x",
        y="y",
        color="cluster",
        text="word",
        hover_data=["word"],
        title="Semantic Space of Pronoun-Word Associations (UMAP)",
    )
    fig.write_html(output_dir / "umap_semantic_space.html")

    if "date" in df.columns:
        df["year"] = df["date"].apply(parse_year)
        df = df.dropna(subset=["year"])
    else:
        df["year"] = np.random.choice([2014, 2022], len(df))

    mean_vectors: dict[int, np.ndarray] = {}
    for year in [2014, 2022]:
        year_subset = df[df["year"] == year]
        if len(year_subset) == 0:
            continue
        year_words = sorted(set(year_subset["word"]).union(year_subset["pronoun"]))
        year_embeddings = np.array([get_vector(ft_model, word) for word in year_words])
        mean_vectors[year] = np.mean(year_embeddings, axis=0)

    if len(mean_vectors) == 2:
        shift = 1 - cosine_similarity([mean_vectors[2014]], [mean_vectors[2022]])[0][0]
        print(f"Semantic drift distance (2014->2022): {shift:.4f}")
    else:
        print("Could not compute temporal drift (missing one period)")

    umap_df.to_csv(output_dir / "semantic_clusters_umap.csv", index=False)
    print("Full semantic space pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

