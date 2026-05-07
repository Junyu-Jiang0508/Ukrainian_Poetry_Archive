"""
Poem Perspective Annotation App - Cloud Multi-User Version
Whole-poem perspective annotation only (sentence-level pronoun annotation deprecated).

Deployable on Streamlit Community Cloud with Supabase backend.

Supabase setup:
  1. Run supabase_schema.sql in your Supabase SQL Editor (see that file for DDL).
  2. Configure st.secrets: [supabase] url = "..." key = "..."

SQL tables used by this version:
  - poem_perspectives(annotator_id, poem_id, perspective_primary,
                      perspective_secondary, author, poem_date)

The legacy `annotations` (sentence-level) table is no longer written to;
related code paths are kept below as commented-out blocks for future restoration.
"""
import os

import pandas as pd
import streamlit as st
from supabase import create_client, Client

# -----------------------------------------------------------------------------
# Paths
# Poem-level CSV with one row per poem.
# Columns: ID, author, year, language, temporal_period, n_stanzas, text, text_en
# -----------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
POEM_FILE = os.path.join(_BASE, "poems_for_manual_annotation.csv")

PERSPECTIVE_OPTIONS = ["1st person", "2nd person", "3rd person", "Mixed", "Other"]
PERSPECTIVE_SECONDARY_OPTIONS = ["None"] + PERSPECTIVE_OPTIONS


def get_supabase() -> Client:
    """Create Supabase client from st.secrets."""
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )


@st.cache_data
def load_poems() -> pd.DataFrame:
    """Load poem-level CSV (one row per poem)."""
    df = pd.read_csv(POEM_FILE)
    # Backwards-compatible columns (older sentence-level CSVs had `date`).
    if "year" in df.columns and "date" not in df.columns:
        df["date"] = df["year"].astype(str)
    return df.reset_index(drop=True)


def load_poem_perspectives(annotator_id: str) -> dict:
    """Load poem perspectives from Supabase for this annotator."""
    try:
        sb = get_supabase()
        r = (
            sb.table("poem_perspectives")
            .select("*")
            .eq("annotator_id", annotator_id)
            .execute()
        )
        rows = r.data or []
        out = {}
        for row in rows:
            out[str(row["poem_id"])] = {
                "perspective_primary": row.get("perspective_primary", ""),
                "perspective_secondary": row.get("perspective_secondary", ""),
                "author": row.get("author", ""),
                "date": row.get("poem_date", ""),
            }
        return out
    except Exception as e:
        st.error(f"Failed to load poem perspectives: {e}")
        return {}


def save_poem_perspective(annotator_id: str, poem_id: str, data: dict):
    """Upsert poem perspective for this annotator."""
    try:
        sb = get_supabase()
        row = {
            "annotator_id": annotator_id,
            "poem_id": str(poem_id),
            "perspective_primary": data.get("perspective_primary", ""),
            "perspective_secondary": data.get("perspective_secondary", ""),
            "author": data.get("author", ""),
            "poem_date": data.get("date", ""),
        }
        sb.table("poem_perspectives").upsert(
            row, on_conflict="annotator_id,poem_id"
        ).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to save poem perspective: {e}")


def build_perspectives_csv(perspectives: dict, poems_df: pd.DataFrame) -> str:
    if not perspectives:
        return ""
    rows = []
    for poem_id, data in perspectives.items():
        meta = poems_df[poems_df["ID"].astype(str) == str(poem_id)]
        text = meta["text"].iloc[0] if not meta.empty else ""
        rows.append(
            {
                "ID": poem_id,
                "author": data.get("author", ""),
                "date": data.get("date", ""),
                "perspective_primary": data.get("perspective_primary", ""),
                "perspective_secondary": data.get("perspective_secondary", ""),
                "text": text,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")


def main():
    st.set_page_config(page_title="Poem Perspective Annotation", layout="wide")
    st.title("Poem Perspective Annotation (Cloud)")

    if not os.path.exists(POEM_FILE):
        st.error(
            f"Poem file not found: {POEM_FILE}. "
            "Add poems_for_manual_annotation.csv to the app folder."
        )
        return

    st.sidebar.header("Annotator")
    annotator_name = st.sidebar.text_input(
        "Annotator Name",
        value=st.session_state.get("annotator_name", ""),
        key="annotator_name_input",
        placeholder="Enter your name",
    )
    if not annotator_name or not annotator_name.strip():
        st.warning("Please enter your Annotator Name in the sidebar to start.")
        st.stop()

    annotator_id = annotator_name.strip()
    st.session_state["annotator_name"] = annotator_id

    if "force_nav_idx" in st.session_state:
        st.session_state.nav_idx = st.session_state.force_nav_idx
        st.session_state["nav_idx_input"] = st.session_state.force_nav_idx
        del st.session_state.force_nav_idx

    poems_df = load_poems()
    if (
        not st.session_state.get("perspectives_loaded")
        or st.session_state.get("annotator_name") != annotator_id
    ):
        st.session_state.poem_perspectives = load_poem_perspectives(annotator_id)
        st.session_state.perspectives_loaded = True
        st.session_state["annotator_name"] = annotator_id

    perspectives = st.session_state.poem_perspectives

    with st.sidebar:
        st.header("Filter")
        all_authors = sorted(poems_df["author"].dropna().unique().tolist())
        author_filter = st.multiselect("Author", all_authors, default=all_authors)
        display_df = poems_df[poems_df["author"].isin(author_filter)].copy().reset_index(drop=True)

        only_unannotated = st.checkbox("Show only unannotated", value=False)
        if only_unannotated:
            done_ids = set(perspectives.keys())
            display_df = display_df[~display_df["ID"].astype(str).isin(done_ids)].reset_index(drop=True)

        jump_id = st.text_input("Jump to poem by ID", placeholder="e.g. UP1184_1", key="jump_poem_id")
        if jump_id and jump_id.strip():
            match_idx = display_df[display_df["ID"].astype(str) == str(jump_id.strip())].index
            if len(match_idx) > 0:
                if st.button("Go to poem", key="jump_poem_btn"):
                    st.session_state.force_nav_idx = int(match_idx[0])
                    st.rerun()
            else:
                st.caption("Poem ID not found in current filter.")

        total = len(display_df)
        annotated_in_view = sum(
            1 for pid in display_df["ID"].astype(str) if pid in perspectives
        )
        st.metric("Poems in view", total, f"Annotated {annotated_in_view}")
        st.metric("Total annotations", len(perspectives))

        st.divider()
        if "nav_idx" not in st.session_state:
            st.session_state.nav_idx = 0
        max_idx = max(0, total - 1)
        safe_value = min(max(0, st.session_state.nav_idx), max_idx)
        idx = st.number_input(
            "Go to poem",
            min_value=0,
            max_value=max_idx,
            value=safe_value,
            step=1,
            key="nav_idx_input",
        )
        st.session_state.nav_idx = int(idx)

        cprev, cnext = st.columns(2)
        with cprev:
            if st.button("← Prev"):
                st.session_state.force_nav_idx = max(int(idx) - 1, 0)
                st.rerun()
        with cnext:
            if st.button("Next →"):
                st.session_state.force_nav_idx = min(int(idx) + 1, max(total - 1, 0))
                st.rerun()

    if display_df.empty:
        st.warning("No poems matching filters.")
        return

    row = display_df.iloc[idx]
    poem_id = str(row["ID"])

    st.subheader(f"Poem {idx + 1} / {total} · {row['author']} · {poem_id}")
    meta_bits = []
    if pd.notna(row.get("year", None)):
        meta_bits.append(f"Year: {row['year']}")
    if pd.notna(row.get("temporal_period", None)):
        meta_bits.append(f"Period: {row['temporal_period']}")
    if pd.notna(row.get("n_stanzas", None)):
        meta_bits.append(f"Stanzas: {int(row['n_stanzas'])}")
    if meta_bits:
        st.caption(" | ".join(meta_bits))

    has_en = bool(str(row.get("text_en", "")).strip())
    if has_en:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ukrainian (full poem)**")
            st.text_area(
                "ukr_text",
                value=str(row.get("text", "")),
                height=400,
                disabled=True,
                key=f"ukr_{poem_id}",
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("**English (Shakespeare-style translation)**")
            st.text_area(
                "en_text",
                value=str(row.get("text_en", "")),
                height=400,
                disabled=True,
                key=f"en_{poem_id}",
                label_visibility="collapsed",
            )
    else:
        st.markdown("**Ukrainian (full poem)**")
        st.text_area(
            "ukr_text",
            value=str(row.get("text", "")),
            height=400,
            disabled=True,
            key=f"ukr_{poem_id}",
            label_visibility="collapsed",
        )

    st.divider()
    st.subheader("Poem Perspective")
    st.caption(
        "Judge the perspective of the whole poem. Primary required; secondary optional."
    )

    current = perspectives.get(poem_id, {})
    primary = current.get("perspective_primary", "")
    secondary = current.get("perspective_secondary", "") or "None"
    legacy_map = {
        "第一人称": "1st person",
        "第二人称": "2nd person",
        "第三人称": "3rd person",
        "混合": "Mixed",
        "其他": "Other",
        "无": "None",
    }
    primary = legacy_map.get(primary, primary)
    secondary = legacy_map.get(secondary, secondary) if secondary else "None"
    idx_primary = (
        PERSPECTIVE_OPTIONS.index(primary) if primary in PERSPECTIVE_OPTIONS else 0
    )
    idx_secondary = (
        PERSPECTIVE_SECONDARY_OPTIONS.index(secondary)
        if secondary in PERSPECTIVE_SECONDARY_OPTIONS
        else 0
    )
    new_primary = st.selectbox(
        "Primary perspective",
        PERSPECTIVE_OPTIONS,
        index=idx_primary,
        key=f"perspective_primary_{poem_id}",
    )
    new_secondary = st.selectbox(
        "Secondary perspective (optional)",
        PERSPECTIVE_SECONDARY_OPTIONS,
        index=idx_secondary,
        key=f"perspective_secondary_{poem_id}",
    )

    def _do_save(and_next: bool):
        try:
            data = {
                "perspective_primary": new_primary,
                "perspective_secondary": new_secondary if new_secondary != "None" else "",
                "author": str(row.get("author", "")),
                "date": str(row.get("date", row.get("year", ""))),
            }
            save_poem_perspective(annotator_id, poem_id, data)
            st.session_state.poem_perspectives[poem_id] = data
            if and_next:
                next_idx = min(int(idx) + 1, max(total - 1, 0))
                st.session_state.force_nav_idx = next_idx
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")

    csave1, csave2 = st.columns(2)
    with csave1:
        if st.button("💾 Save (stay)", key=f"save_stay_{poem_id}"):
            _do_save(and_next=False)
    with csave2:
        if st.button("Save and next →", key=f"save_next_{poem_id}"):
            _do_save(and_next=True)

    if poem_id in perspectives:
        st.success(
            f"Saved · primary: {perspectives[poem_id].get('perspective_primary','')}"
            + (
                f" · secondary: {perspectives[poem_id].get('perspective_secondary','')}"
                if perspectives[poem_id].get("perspective_secondary")
                else ""
            )
        )

    st.divider()
    csv_persp = build_perspectives_csv(st.session_state.poem_perspectives, poems_df)
    if csv_persp:
        st.download_button(
            "Download poem perspectives (CSV)",
            csv_persp,
            file_name=f"manual_annotation_poem_perspectives_{annotator_id.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No poem perspectives to download yet.")

    with st.expander("Preview poem perspectives"):
        if st.session_state.poem_perspectives:
            persp_data = [
                {
                    "ID": k,
                    "author": v.get("author", ""),
                    "Primary": v.get("perspective_primary", ""),
                    "Secondary": v.get("perspective_secondary", ""),
                }
                for k, v in st.session_state.poem_perspectives.items()
            ]
            st.dataframe(pd.DataFrame(persp_data), width="stretch")
        else:
            st.info("No poem perspectives yet.")


# =============================================================================
# DEPRECATED: sentence-level pronoun annotation
# Kept commented out for future restoration. The Supabase `annotations` table
# is no longer written to from the UI; reactivate by restoring these helpers
# and the per-sentence UI block below.
# =============================================================================
# OUTPUT_COLUMNS = [
#     "ID", "author", "date", "Language", "text", "Theme",
#     "pronoun", "lemma", "uk_match_pos", "position", "context",
#     "person", "number", "gender", "case", "is_dropped", "en_reference",
#     "shakespeare_text", "gpt_annotations",
# ]
# PERSON_OPTIONS = ["1st", "2nd", "3rd", "Impersonal"]
# NUMBER_OPTIONS = ["Singular", "Plural", "None"]
#
# def load_annotations(annotator_id: str) -> list:
#     """Load sentence-level pronoun annotations from Supabase."""
#     try:
#         sb = get_supabase()
#         r = sb.table("annotations").select("*").eq("annotator_id", annotator_id).execute()
#         rows = r.data or []
#         out = []
#         for row in rows:
#             rec = {
#                 "ID": row["poem_id"],
#                 "sentence_id": row["sentence_id"],
#                 "no_pronoun": row.get("no_pronoun", False),
#             }
#             if not row.get("no_pronoun"):
#                 rec.update({
#                     "pronoun": row.get("pronoun", ""),
#                     "lemma": row.get("lemma", ""),
#                     "person": row.get("person", ""),
#                     "number": row.get("number", ""),
#                     "is_dropped": row.get("is_dropped", True),
#                     "position": row.get("position", 0),
#                 })
#             out.append(rec)
#         return out
#     except Exception as e:
#         st.error(f"Failed to load annotations: {e}")
#         return []
#
# def save_annotations_for_sentence(annotator_id, poem_id, sentence_id, records):
#     """Delete existing annotations for this sentence, then insert new ones."""
#     try:
#         sb = get_supabase()
#         sb.table("annotations").delete().eq("annotator_id", annotator_id).eq(
#             "poem_id", str(poem_id)
#         ).eq("sentence_id", int(sentence_id)).execute()
#         for a in records:
#             row = {
#                 "annotator_id": annotator_id,
#                 "poem_id": str(a["ID"]),
#                 "sentence_id": int(a.get("sentence_id", 0)),
#                 "no_pronoun": bool(a.get("no_pronoun", False)),
#             }
#             if not a.get("no_pronoun"):
#                 row.update({
#                     "pronoun": str(a.get("pronoun", "")),
#                     "lemma": str(a.get("lemma", a.get("pronoun", ""))),
#                     "person": str(a.get("person", "")),
#                     "number": str(a.get("number", "")),
#                     "is_dropped": bool(a.get("is_dropped", True)),
#                     "position": int(a.get("position", a.get("sentence_id", 0))),
#                 })
#             sb.table("annotations").insert(row).execute()
#     except Exception as e:
#         raise RuntimeError(f"Failed to save annotations: {e}")
#
# def get_reviewed_sentences(annotations: list) -> set:
#     return {(str(a["ID"]), int(a.get("sentence_id", 0))) for a in annotations}
#
# def is_poem_fully_annotated(poem_id, sentences_df, reviewed) -> bool:
#     poem_sents = sentences_df[sentences_df["ID"].astype(str) == str(poem_id)]
#     if poem_sents.empty:
#         return False
#     for _, r in poem_sents.iterrows():
#         key = (str(r["ID"]), int(r["sentence_id"]) if pd.notna(r["sentence_id"]) else 0)
#         if key not in reviewed:
#             return False
#     return True


if __name__ == "__main__":
    main()
