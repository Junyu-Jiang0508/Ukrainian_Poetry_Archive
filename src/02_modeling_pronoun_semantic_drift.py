"""Static-vector pronoun semantic drift (P1-C) via FastText + orthogonal Procrustes.

The question
------------
Does the *semantic neighbourhood* of the central deictic pronouns (ми/я/ти/ви
in Ukrainian, мы/я/ты/вы in Russian) shift across the 24 February 2022
cutpoint? This is the analytic complement to the rate-based (Q1/Q2a) tests
that asks not ``how often'' but ``what surrounds them''.

Method, following \citet{hamilton2016diachronic}
------------------------------------------------
Per (language, period):

1. Tokenize the corpus by a regex over Cyrillic + apostrophe characters.
2. Lower-case; drop a small closed-class stopword list to reduce trivial
   adjacency.
3. Train FastText (subword n-grams) on stanza-level lines with ``min_count=5``,
   ``vector_size=200``, ``window=8``, ``epochs=5``. Subword n-grams provide
   partial robustness to morphological variation without explicit lemmatization;
   for the focal pronouns we additionally aggregate over paradigm forms.

Across periods, align P2 vectors to P1 by **orthogonal Procrustes**:
$R = \arg\min_{R: R^\top R = I} \| W_{P2} R - W_{P1} \|_F$
on the intersection vocabulary (min count $\ge$ 5 in both periods). Drift for
a target word $w$ is
$\text{drift}(w) = 1 - \cos(W_{P1}[w], W_{P2}[w] R)$.

We report drift for {ми/мы, я, ти/ты, ви/вы} and rank each against the full
intersection-vocabulary drift distribution (percentile). For each focal
pronoun, we also list its top-10 nearest neighbours in P1 and in P2 (post
Procrustes), which provides a transparent qualitative trace.

The module is intentionally agnostic to the inclusive/exclusive 1PL
distinction (deferred per user direction) and to GPT-based referent
clustering (skipped pending human validation). It is a corpus-level
*neighbourhood* drift report, not a *referent* attribution.

Outputs (in ``outputs/02_modeling_pronoun_semantic_drift/``)
------------------------------------------------------------
* ``focal_pronoun_drift.csv``
* ``focal_pronoun_neighbours_P1.csv`` / ``focal_pronoun_neighbours_P2.csv``
* ``vocab_drift_distribution.csv`` (percentile reference)
* ``focal_pronoun_drift.png`` (bar chart by language)
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils.workspace import canonical_pronoun_annotation_csv, prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")
log = logging.getLogger(__name__)

DEFAULT_INPUT = canonical_pronoun_annotation_csv(ROOT)
DEFAULT_OUTPUT = ROOT / "outputs" / "02_modeling_pronoun_semantic_drift"


# Cyrillic + Latin alpha + apostrophe (Ukrainian uses U+2019). We do NOT include
# digits or punctuation; the corpus is short verse so the regex stays simple.
_TOKEN_RE = re.compile(r"[А-Яа-яҐґЄєІіЇїA-Za-z’']+", re.UNICODE)

# Minimal closed-class stoplists. We deliberately keep them short — pronouns
# (the focal items) are NOT in the stoplist, and we tolerate noise from common
# prepositions rather than risk dropping semantically loaded function words.
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

# Focal pronoun lemmas and their paradigm forms (case-marked surface variants
# we want to aggregate over before computing drift). The non-nominative forms
# are pooled into the same focal token for the drift report.
FOCAL_LEMMAS: dict[str, dict[str, tuple[str, ...]]] = {
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


def _tokenize(text: str, lang: str) -> list[str]:
    """Lower-case regex tokenization with a small closed-class stoplist."""
    stop = STOPWORDS.get(lang, set())
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in stop and len(t) >= 2]


def _stanza_corpus_by_period(
    df: pd.DataFrame,
    language: str,
    periods: tuple[str, ...],
) -> dict[str, list[list[str]]]:
    """Return ``{period -> list of tokenized stanza lines}`` for one language."""
    out: dict[str, list[list[str]]] = {p: [] for p in periods}
    sub = df.loc[
        (df["language"].astype(str).eq(language))
        & (df["temporal_period"].astype(str).isin(periods))
    ].copy()
    dedup = sub.drop_duplicates(["poem_id", "stanza_index"])
    for _, row in dedup.iterrows():
        text = str(row.get("stanza_ukr", "") or "")
        if not text:
            continue
        period = str(row["temporal_period"])
        tokens = _tokenize(text, language)
        if tokens:
            out[period].append(tokens)
    return out


def _train_fasttext(
    sentences: list[list[str]],
    *,
    min_count: int = 5,
    vector_size: int = 200,
    window: int = 8,
    epochs: int = 5,
    seed: int = 20260519,
):
    """Train a FastText model. Deferred import keeps gensim optional."""
    from gensim.models import FastText

    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        workers=4,
        seed=seed,
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model


def _intersection_matrix(
    model_p1, model_p2, *, min_count_each: int = 5
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return aligned matrices for the intersection vocabulary."""
    vocab_p1 = {w: c for w, c in zip(model_p1.wv.index_to_key, [model_p1.wv.get_vecattr(w, "count") for w in model_p1.wv.index_to_key])}
    vocab_p2 = {w: c for w, c in zip(model_p2.wv.index_to_key, [model_p2.wv.get_vecattr(w, "count") for w in model_p2.wv.index_to_key])}
    common = sorted(
        w for w in vocab_p1 if w in vocab_p2 and vocab_p1[w] >= min_count_each and vocab_p2[w] >= min_count_each
    )
    if len(common) < 100:
        log.warning("Intersection vocabulary is small: %d words", len(common))
    X_p1 = np.stack([model_p1.wv[w] for w in common])
    X_p2 = np.stack([model_p2.wv[w] for w in common])
    return X_p1, X_p2, common


def _procrustes_align(X_p1: np.ndarray, X_p2: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: rotation R minimizing ``||X_p2 @ R - X_p1||_F``."""
    M = X_p2.T @ X_p1
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def _focal_vector(model, forms: tuple[str, ...]) -> np.ndarray | None:
    """Average the in-vocabulary surface-form vectors for a focal pronoun lemma."""
    vectors = []
    for f in forms:
        if f in model.wv:
            vectors.append(model.wv[f])
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)


def _top_neighbours(model, focal_vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """Top-k cosine neighbours of an externally provided vector."""
    if focal_vec is None:
        return []
    sims = model.wv.cosine_similarities(focal_vec, model.wv.vectors)
    order = np.argsort(-sims)[: k + len(FOCAL_LEMMAS) * 6]
    out: list[tuple[str, float]] = []
    for idx in order:
        word = model.wv.index_to_key[idx]
        # filter out the focal pronouns themselves to keep the neighbour list useful
        flat_focal = {f for forms in FOCAL_LEMMAS.values() for fs in forms.values() for f in fs}
        if word in flat_focal:
            continue
        out.append((word, float(sims[idx])))
        if len(out) >= k:
            break
    return out


def run_for_language(
    df: pd.DataFrame,
    language: str,
    *,
    periods: tuple[str, str],
    out_dir: Path,
    fasttext_kwargs: dict,
) -> dict[str, pd.DataFrame]:
    """Train per-period FastText, align, and report focal-pronoun drift."""
    corpora = _stanza_corpus_by_period(df, language, periods)
    if any(not v for v in corpora.values()):
        log.warning("Language %s missing data in one of the periods; skipping.", language)
        return {}

    log.info(
        "%s: %d stanzas P1, %d stanzas P2 — training FastText",
        language,
        len(corpora[periods[0]]),
        len(corpora[periods[1]]),
    )
    model_p1 = _train_fasttext(corpora[periods[0]], **fasttext_kwargs)
    model_p2 = _train_fasttext(corpora[periods[1]], **fasttext_kwargs)

    X_p1, X_p2, common = _intersection_matrix(model_p1, model_p2)
    R = _procrustes_align(X_p1, X_p2)

    # Compute drift for the full intersection vocab as a reference percentile distribution.
    aligned_p2 = X_p2 @ R
    norms_p1 = np.linalg.norm(X_p1, axis=1).clip(min=1e-12)
    norms_p2 = np.linalg.norm(aligned_p2, axis=1).clip(min=1e-12)
    cos_arr = np.einsum("ij,ij->i", X_p1, aligned_p2) / (norms_p1 * norms_p2)
    drift_arr = 1.0 - cos_arr
    vocab_drift_df = pd.DataFrame({"word": common, "drift": drift_arr, "language": language})

    # Focal-pronoun drift (paradigm-form averaged).
    focal_rows: list[dict[str, object]] = []
    neigh_p1_rows: list[dict[str, object]] = []
    neigh_p2_rows: list[dict[str, object]] = []
    for lemma, forms in FOCAL_LEMMAS[language].items():
        v_p1 = _focal_vector(model_p1, forms)
        v_p2 = _focal_vector(model_p2, forms)
        if v_p1 is None or v_p2 is None:
            log.warning("Focal lemma %s missing in one period for %s", lemma, language)
            continue
        v_p2_aligned = v_p2 @ R
        drift = 1.0 - _cosine(v_p1, v_p2_aligned)
        percentile = float((drift_arr < drift).mean() * 100.0)
        focal_rows.append(
            {
                "language": language,
                "focal_lemma": lemma,
                "forms_pooled": "|".join(forms),
                "drift_1_minus_cos": float(drift),
                "percentile_vs_vocab": percentile,
                "n_forms_in_p1": sum(1 for f in forms if f in model_p1.wv),
                "n_forms_in_p2": sum(1 for f in forms if f in model_p2.wv),
            }
        )
        for word, sim in _top_neighbours(model_p1, v_p1, k=10):
            neigh_p1_rows.append({"language": language, "focal_lemma": lemma, "neighbour": word, "cosine": sim})
        for word, sim in _top_neighbours(model_p2, v_p2, k=10):
            neigh_p2_rows.append({"language": language, "focal_lemma": lemma, "neighbour": word, "cosine": sim})

    return {
        "focal_drift": pd.DataFrame(focal_rows),
        "neigh_p1": pd.DataFrame(neigh_p1_rows),
        "neigh_p2": pd.DataFrame(neigh_p2_rows),
        "vocab_drift": vocab_drift_df,
    }


def _plot_focal_drift(focal_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if focal_df.empty:
        return
    fig, axes = plt.subplots(1, len(focal_df["language"].unique()), figsize=(10, 4), squeeze=False)
    for ax, lang in zip(axes[0], sorted(focal_df["language"].unique())):
        sub = focal_df.loc[focal_df["language"].eq(lang)].sort_values("drift_1_minus_cos")
        ax.barh(sub["focal_lemma"], sub["drift_1_minus_cos"], color="steelblue", alpha=0.7)
        ax.set_title(f"{lang}: pronoun drift (1 − cos, post-Procrustes)")
        ax.set_xlabel("Drift (higher = more contextual change)")
        for y, p in zip(sub["focal_lemma"], sub["percentile_vs_vocab"]):
            ax.annotate(f"P{p:.0f}", (sub.loc[sub["focal_lemma"].eq(y), "drift_1_minus_cos"].iloc[0], y), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Pronoun semantic drift via FastText + Procrustes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False, on_bad_lines="skip")
    periods = ("2014_2021", "post_2022")

    fasttext_kwargs = dict(
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs,
        min_count=args.min_count,
    )

    focal_frames: list[pd.DataFrame] = []
    neigh_p1_frames: list[pd.DataFrame] = []
    neigh_p2_frames: list[pd.DataFrame] = []
    vocab_drift_frames: list[pd.DataFrame] = []
    for language in ("Ukrainian", "Russian"):
        result = run_for_language(df, language, periods=periods, out_dir=out_dir, fasttext_kwargs=fasttext_kwargs)
        if not result:
            continue
        focal_frames.append(result["focal_drift"])
        neigh_p1_frames.append(result["neigh_p1"])
        neigh_p2_frames.append(result["neigh_p2"])
        vocab_drift_frames.append(result["vocab_drift"])

    if focal_frames:
        focal = pd.concat(focal_frames, ignore_index=True)
        focal.to_csv(out_dir / "focal_pronoun_drift.csv", index=False)
        _plot_focal_drift(focal, out_dir / "focal_pronoun_drift.png")
    if neigh_p1_frames:
        pd.concat(neigh_p1_frames, ignore_index=True).to_csv(out_dir / "focal_pronoun_neighbours_P1.csv", index=False)
    if neigh_p2_frames:
        pd.concat(neigh_p2_frames, ignore_index=True).to_csv(out_dir / "focal_pronoun_neighbours_P2.csv", index=False)
    if vocab_drift_frames:
        pd.concat(vocab_drift_frames, ignore_index=True).to_csv(out_dir / "vocab_drift_distribution.csv", index=False)

    log.info("Wrote semantic-drift outputs to %s", out_dir)


if __name__ == "__main__":
    main()
