"""Stratified Stanza validation sample: compare morph-only vs depparse pipelines on finite-verb tagging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.finite_verb_exposure import is_finite_verb, init_stanza_finite_verb_pipeline
from utils.stats_common import period_three_way
from utils.workspace import prepare_analysis_environment

ROOT = prepare_analysis_environment(__file__, matplotlib_backend="Agg")

log = logging.getLogger(__name__)

DEFAULT_LAYER1 = ROOT / "data" / "To_run" / "00_filtering" / "layer1_stanzas_one_per_row.csv"
DEFAULT_POEM_TABLE = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "q1_poem_unit_cell_counts_12.csv"
DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "finite_verb_validation_sample.csv"
DEFAULT_AGREEMENT = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "finite_verb_validation_pipeline_agreement.csv"
DEFAULT_CONV = ROOT / "outputs" / "02_modeling_q1_per_cell_glm" / "finite_verb_validation_converbs_and_participles.csv"


def _parse_feats(feats_str: str | None) -> dict[str, str]:
    if not feats_str:
        return {}
    out: dict[str, str] = {}
    for item in feats_str.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def _stanza_text_col(df: pd.DataFrame) -> str:
    if "stanza_text" in df.columns:
        return "stanza_text"
    if "stanza_ukr" in df.columns:
        return "stanza_ukr"
    raise ValueError("layer1 must have stanza_text or stanza_ukr")


def _build_stanza_depparse():
    import stanza

    return stanza.Pipeline(
        lang="uk",
        processors="tokenize,pos,lemma,depparse",
        download_method=None,
        verbose=False,
    )


def _token_rows_for_doc(
    poem_id: str,
    stanza_index: int,
    doc_morph,
    doc_dep,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (token_rows, conv_part_rows)."""
    rows: list[dict[str, object]] = []
    conv_rows: list[dict[str, object]] = []
    n_sent = min(len(doc_morph.sentences), len(doc_dep.sentences))
    for si in range(n_sent):
        sent_m = doc_morph.sentences[si]
        sent_d = doc_dep.sentences[si]
        n_w = min(len(sent_m.words), len(sent_d.words))
        for wi in range(n_w):
            wm = sent_m.words[wi]
            wd = sent_d.words[wi]
            feats_m = _parse_feats(getattr(wm, "feats", None))
            vf = feats_m.get("VerbForm", "")
            fin_m = is_finite_verb(wm, exclude_imperative=False)
            fin_m_ex = is_finite_verb(wm, exclude_imperative=True)
            fin_d = is_finite_verb(wd, exclude_imperative=False)
            fin_d_ex = is_finite_verb(wd, exclude_imperative=True)
            rows.append(
                {
                    "poem_id": poem_id,
                    "stanza_index": stanza_index,
                    "sentence_index": si,
                    "token_id_in_sentence": int(wm.id),
                    "token_text": wm.text,
                    "lemma": wm.lemma,
                    "upos": wm.upos,
                    "feats": wm.feats or "",
                    "is_finite_verb_default_morph": fin_m,
                    "is_finite_verb_excl_imperative_morph": fin_m_ex,
                    "is_finite_verb_default_depparse": fin_d,
                    "is_finite_verb_excl_imperative_depparse": fin_d_ex,
                    "depparse_agrees_finite_default": fin_m == fin_d,
                    "depparse_agrees_finite_excl_imperative": fin_m_ex == fin_d_ex,
                }
            )
            if vf in ("Conv", "Part"):
                conv_rows.append(
                    {
                        "poem_id": poem_id,
                        "stanza_index": stanza_index,
                        "sentence_index": si,
                        "token_text": wm.text,
                        "lemma": wm.lemma,
                        "VerbForm": vf,
                        "feats": wm.feats or "",
                        "finite_by_morph_default": fin_m,
                    }
                )
    return rows, conv_rows


def sample_poem_ids(poem_tbl: pd.DataFrame, *, rng: np.random.RandomState) -> list[str]:
    """30 poems: 10 Ukrainian P1, 10 Ukrainian P2, 10 Russian (any period in P1/P2)."""
    pt = poem_tbl.copy()
    pt["poem_id"] = pt["poem_id"].astype(str).str.strip()
    if "period3" not in pt.columns and "year_int" in pt.columns:
        pt["period3"] = pt["year_int"].map(period_three_way)
    u_p1 = pt.loc[pt["language_clean"].eq("Ukrainian") & pt["period3"].eq("P1_2014_2021"), "poem_id"].unique()
    u_p2 = pt.loc[pt["language_clean"].eq("Ukrainian") & pt["period3"].eq("P2_2022_plus"), "poem_id"].unique()
    ru = pt.loc[
        pt["language_clean"].eq("Russian") & pt["period3"].isin(("P1_2014_2021", "P2_2022_plus")),
        "poem_id",
    ].unique()

    def take(arr: np.ndarray, n: int) -> list[str]:
        arr = np.array(list(arr), dtype=str)
        if len(arr) <= n:
            return arr.tolist()
        idx = rng.choice(len(arr), size=n, replace=False)
        return arr[idx].tolist()

    out = take(u_p1, 10) + take(u_p2, 10) + take(ru, 10)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Export stratified finite-verb validation tokens (Stanza).")
    ap.add_argument("--poem-table", type=Path, default=DEFAULT_POEM_TABLE)
    ap.add_argument("--layer1", type=Path, default=DEFAULT_LAYER1)
    ap.add_argument("--output-tokens", type=Path, default=DEFAULT_OUTPUT_CSV)
    ap.add_argument("--output-agreement", type=Path, default=DEFAULT_AGREEMENT)
    ap.add_argument("--output-converbs", type=Path, default=DEFAULT_CONV)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    poem_tbl = pd.read_csv(args.poem_table.resolve(), low_memory=False)
    layer1 = pd.read_csv(args.layer1.resolve(), low_memory=False)
    txt_col = _stanza_text_col(layer1)
    layer1["poem_id"] = layer1["poem_id"].astype(str).str.strip()

    rng = np.random.RandomState(int(args.seed))
    sample_ids = sample_poem_ids(poem_tbl, rng=rng)
    log.info("Sampled %s poems for validation", len(sample_ids))

    nlp_m = init_stanza_finite_verb_pipeline()
    nlp_d = _build_stanza_depparse()

    all_tokens: list[dict[str, object]] = []
    all_conv: list[dict[str, object]] = []
    agree_stats: list[dict[str, object]] = []

    for pid in sample_ids:
        sub = layer1.loc[layer1["poem_id"].eq(pid)].copy()
        if sub.empty:
            log.warning("poem_id=%s not in layer1; skip", pid)
            continue
        for _, row in sub.iterrows():
            si_raw = pd.to_numeric(row["stanza_index"], errors="coerce")
            if pd.isna(si_raw):
                continue
            si = int(si_raw)
            text = row[txt_col]
            if text is None or (isinstance(text, float) and pd.isna(text)):
                continue
            s = str(text).strip()
            if not s:
                continue
            try:
                doc_m = nlp_m(s)
                doc_d = nlp_d(s)
            except Exception as exc:
                log.warning("parse failed poem=%s stanza=%s: %s", pid, si, exc)
                continue
            toks, convp = _token_rows_for_doc(pid, si, doc_m, doc_d)
            all_tokens.extend(toks)
            all_conv.extend(convp)
            df_t = pd.DataFrame(toks)
            if not df_t.empty:
                vmask = df_t["upos"].eq("VERB")
                vsub = df_t.loc[vmask]
                n_v = int(len(vsub))
                if n_v:
                    agree_stats.append(
                        {
                            "poem_id": pid,
                            "stanza_index": si,
                            "n_verb_tokens": n_v,
                            "n_agree_finite_default": int(vsub["depparse_agrees_finite_default"].sum()),
                            "n_agree_finite_excl_imp": int(vsub["depparse_agrees_finite_excl_imperative"].sum()),
                        }
                    )

    tok_df = pd.DataFrame(all_tokens)
    args.output_tokens.parent.mkdir(parents=True, exist_ok=True)
    tok_df.to_csv(args.output_tokens, index=False)
    log.info("Wrote %s (%s rows)", args.output_tokens, len(tok_df))

    pd.DataFrame(all_conv).to_csv(args.output_converbs, index=False)
    log.info("Wrote %s (VerbForm=Conv|Part for spot-check)", args.output_converbs)

    if agree_stats:
        a = pd.DataFrame(agree_stats)
        tot_v = int(a["n_verb_tokens"].sum())
        tot_d = int(a["n_agree_finite_default"].sum())
        tot_e = int(a["n_agree_finite_excl_imp"].sum())
        summary = pd.DataFrame(
            [
                {
                    "n_stanzas_parsed": int(len(a)),
                    "n_verb_tokens": tot_v,
                    "finite_default_agree_rate": float(tot_d / tot_v) if tot_v else np.nan,
                    "finite_excl_imp_agree_rate": float(tot_e / tot_v) if tot_v else np.nan,
                }
            ]
        )
        summary.to_csv(args.output_agreement, index=False)
        log.info("Wrote %s", args.output_agreement)
    else:
        pd.DataFrame(
            columns=[
                "n_stanzas_parsed",
                "n_verb_tokens",
                "finite_default_agree_rate",
                "finite_excl_imp_agree_rate",
            ]
        ).to_csv(args.output_agreement, index=False)


if __name__ == "__main__":
    main()
