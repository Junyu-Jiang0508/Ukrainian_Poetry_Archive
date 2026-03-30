import os, re, numpy as np, pandas as pd
from gensim.models.fasttext import load_facebook_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from datetime import datetime
import urllib.request
import gzip
import shutil
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

INPUT_PATH = "outputs/02_pronoun_cooccurrence/pronoun_cooccurrence_with_date.csv"
INPUT_PATH_SIMPLE = "outputs/02_pronoun_cooccurrence/pronoun_cooccurrence.csv"
OUTPUT_DIR = "outputs/03_pronoun_semantic_space"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(INPUT_PATH):
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} pronoun–word edges (with dates)")
else:
    print(f"Date file not found, using simple version")
    df = pd.read_csv(INPUT_PATH_SIMPLE)
    print(f"Loaded {len(df)} pronoun–word edges (no dates)")

model_path = 'cc.uk.300.bin'
if not os.path.exists(model_path):
    print("Downloading Ukrainian FastText model (~6GB)...")
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.bin.gz"
    try:
        urllib.request.urlretrieve(url, 'cc.uk.300.bin.gz')
        print("Extracting model...")
        with gzip.open('cc.uk.300.bin.gz', 'rb') as f_in:
            with open('cc.uk.300.bin', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove('cc.uk.300.bin.gz')
        print("Model downloaded and extracted")
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("You can manually download from: {url}")
        raise

ft = load_facebook_model(model_path)
print("FastText model loaded")

def get_vec(w):
    try:
        return ft.wv[w]
    except:
        return np.zeros(300)

all_words = sorted(set(df["word"]).union(df["pronoun"]))
embeddings = np.array([get_vec(w) for w in all_words])
print(f"Computed {len(all_words)} word embeddings")

reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='cosine', random_state=42)
umap_coords = reducer.fit_transform(embeddings)
umap_df = pd.DataFrame({"word": all_words, "x": umap_coords[:,0], "y": umap_coords[:,1]})
umap_df.to_csv(os.path.join(OUTPUT_DIR, "umap_embedding_layout.csv"), index=False)
print("UMAP layout computed")

clusterer = DBSCAN(eps=0.5, min_samples=8, metric='euclidean').fit(umap_coords)
umap_df["cluster"] = clusterer.labels_
n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
n_noise = list(clusterer.labels_).count(-1)
print(f"Found {n_clusters} semantic clusters ({n_noise} noise points)")


docs_per_cluster = {}
for i, row in umap_df.iterrows():
    if row["cluster"] == -1:
        continue
    docs_per_cluster.setdefault(row["cluster"], []).append(row["word"])

if len(docs_per_cluster) > 0:
    cluster_ids = sorted(docs_per_cluster.keys())
    cluster_docs = [" ".join(docs_per_cluster[cid]) for cid in cluster_ids]
    

    tfidf = TfidfVectorizer(max_features=10, ngram_range=(1,1))
    tfidf_matrix = tfidf.fit_transform(cluster_docs)
    feature_names = tfidf.get_feature_names_out()
    
    with open(os.path.join(OUTPUT_DIR, "cluster_labels.txt"), "w", encoding="utf-8") as f:
        for idx, cid in enumerate(cluster_ids):
            cluster_tfidf = tfidf_matrix[idx].toarray()[0]
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices if cluster_tfidf[i] > 0]
            
            if len(top_words) == 0:
                top_words = docs_per_cluster[cid][:10]
            
            f.write(f"Cluster {cid} ({len(docs_per_cluster[cid])} words): {', '.join(top_words)}\n")
    
    print("Auto-labeled clusters saved via TF-IDF")
else:
    print("No clusters found, skipping labeling")

fig = px.scatter(
    umap_df, x="x", y="y", color="cluster", text="word",
    hover_data=["word"], title="Semantic Space of Pronoun–Word Associations (UMAP)"
)
fig.write_html(os.path.join(OUTPUT_DIR, "umap_semantic_space.html"))
print(f"Saved interactive UMAP map")

def parse_year(s):
    try:
        return int(str(s)[:4])
    except:
        return None

if "date" in df.columns:
    df["year"] = df["date"].apply(parse_year)
    df = df.dropna(subset=["year"])
    print(f"Extracted years from {len(df)} records")
else:
    df["year"] = np.random.choice([2014, 2022], len(df))
    print("WARNING: No 'date' column found, using random years for demonstration")

periods = [2014, 2022]
mean_vecs = {}

for y in periods:
    subset = df[df["year"] == y]
    if len(subset) > 0:
        words_y = sorted(set(subset["word"]).union(subset["pronoun"]))
        vecs_y = np.array([get_vec(w) for w in words_y])
        mean_vecs[y] = np.mean(vecs_y, axis=0)

if len(mean_vecs) == 2:
    shift = 1 - cosine_similarity([mean_vecs[2014]], [mean_vecs[2022]])[0][0]
    print(f"Semantic drift distance (2014→2022): {shift:.4f}")
else:
    print("Could not compute temporal drift (missing data for one or both periods)")

umap_df.to_csv(os.path.join(OUTPUT_DIR, "semantic_clusters_umap.csv"), index=False)
print("Full semantic space pipeline complete.")

