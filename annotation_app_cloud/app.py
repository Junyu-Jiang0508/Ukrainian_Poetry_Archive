"""
Pronoun Pro-Drop Annotation App - Cloud Multi-User Version
Deployable on Streamlit Community Cloud with Supabase backend.

Supabase setup:
  1. Run supabase_schema.sql in your Supabase SQL Editor (see that file for DDL).
  2. Configure st.secrets: [supabase] url = "..." key = "..."

SQL tables (see supabase_schema.sql for full DDL):
  - annotations(annotator_id, poem_id, sentence_id, no_pronoun, pronoun, lemma, person, number, is_dropped, position)
  - poem_perspectives(annotator_id, poem_id, perspective_primary, perspective_secondary, author, poem_date)
"""
import os

import pandas as pd
import streamlit as st
from supabase import create_client, Client

# -----------------------------------------------------------------------------
# Paths (cross-platform, forward slashes)
# SENTENCE_FILE: local CSV with sentences to annotate.
# For Streamlit Cloud: add poems_for_manual_annotation.csv to this folder.
# For local dev: can use parent outputs path as fallback.
# -----------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
SENTENCE_FILE = os.path.join(_BASE, "poems_for_manual_annotation.csv")
if not os.path.exists(SENTENCE_FILE):
    _FALLBACK = os.path.join(_BASE, "..", "outputs", "01_pronouns_detection", "poems_for_manual_annotation.csv")
    if os.path.exists(_FALLBACK):
        SENTENCE_FILE = _FALLBACK

# Column order for export (matches gpt_annotation_test_result.csv)
OUTPUT_COLUMNS = [
    "ID", "author", "date", "Language", "text", "Theme",
    "pronoun", "lemma", "uk_match_pos", "position", "context",
    "person", "number", "gender", "case", "is_dropped", "en_reference",
    "shakespeare_text", "gpt_annotations",
]

PERSON_OPTIONS = ["1st", "2nd", "3rd", "Impersonal"]
NUMBER_OPTIONS = ["Singular", "Plural", "None"]
PERSPECTIVE_OPTIONS = ["1st person", "2nd person", "3rd person", "Mixed", "Other"]
PERSPECTIVE_SECONDARY_OPTIONS = ["None"] + PERSPECTIVE_OPTIONS


def get_supabase() -> Client:
    """Create Supabase client from st.secrets."""
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )


@st.cache_data
def load_sentences():
    """Load sentences from local CSV."""
    df = pd.read_csv(SENTENCE_FILE)
    return df.reset_index(drop=True)


def load_annotations(annotator_id: str) -> list:
    """Load annotations from Supabase for this annotator."""
    try:
        sb = get_supabase()
        r = sb.table("annotations").select("*").eq("annotator_id", annotator_id).execute()
        rows = r.data or []
        out = []
        for row in rows:
            rec = {
                "ID": row["poem_id"],
                "sentence_id": row["sentence_id"],
                "no_pronoun": row.get("no_pronoun", False),
            }
            if not row.get("no_pronoun"):
                rec.update({
                    "pronoun": row.get("pronoun", ""),
                    "lemma": row.get("lemma", ""),
                    "person": row.get("person", ""),
                    "number": row.get("number", ""),
                    "is_dropped": row.get("is_dropped", True),
                    "position": row.get("position", 0),
                })
            out.append(rec)
        return out
    except Exception as e:
        st.error(f"Failed to load annotations: {e}")
        return []


def save_annotations_for_sentence(annotator_id: str, poem_id: str, sentence_id: int, records: list):
    """Delete existing annotations for this sentence, then insert new ones."""
    try:
        sb = get_supabase()
        sb.table("annotations").delete().eq("annotator_id", annotator_id).eq(
            "poem_id", str(poem_id)
        ).eq("sentence_id", int(sentence_id)).execute()
        for a in records:
            row = {
                "annotator_id": annotator_id,
                "poem_id": str(a["ID"]),
                "sentence_id": int(a.get("sentence_id", 0)),
                "no_pronoun": bool(a.get("no_pronoun", False)),
            }
            if not a.get("no_pronoun"):
                row.update({
                    "pronoun": str(a.get("pronoun", "")),
                    "lemma": str(a.get("lemma", a.get("pronoun", ""))),
                    "person": str(a.get("person", "")),
                    "number": str(a.get("number", "")),
                    "is_dropped": bool(a.get("is_dropped", True)),
                    "position": int(a.get("position", a.get("sentence_id", 0))),
                })
            sb.table("annotations").insert(row).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to save annotations: {e}")


def get_reviewed_sentences(annotations: list) -> set:
    return {(str(a["ID"]), int(a.get("sentence_id", 0))) for a in annotations}


def load_poem_perspectives(annotator_id: str) -> dict:
    """Load poem perspectives from Supabase for this annotator."""
    try:
        sb = get_supabase()
        r = sb.table("poem_perspectives").select("*").eq("annotator_id", annotator_id).execute()
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


def _reviewed_key(poem_id, sentence_id) -> tuple:
    """Normalize key for reviewed set (handles float sentence_id from CSV)."""
    return (str(poem_id), int(sentence_id) if pd.notna(sentence_id) else 0)


def is_poem_fully_annotated(poem_id: str, display_df: pd.DataFrame, reviewed: set) -> bool:
    poem_sents = display_df[display_df["ID"].astype(str) == str(poem_id)]
    if poem_sents.empty:
        return False
    for _, r in poem_sents.iterrows():
        key = _reviewed_key(r["ID"], r["sentence_id"])
        if key not in reviewed:
            return False
    return True


def pronoun_row_to_output(row: dict, sentence_row: pd.Series) -> dict:
    return {
        "ID": sentence_row["ID"],
        "author": sentence_row["author"],
        "date": sentence_row["date"],
        "Language": sentence_row.get("Language", "Ukrainian"),
        "text": sentence_row["context"],
        "Theme": sentence_row.get("Theme", ""),
        "pronoun": row["pronoun"],
        "lemma": row.get("lemma", row["pronoun"]),
        "uk_match_pos": row.get("uk_match_pos", ""),
        "position": row.get("position", row.get("sentence_id", "")),
        "context": sentence_row["sentence"],
        "person": row.get("person", ""),
        "number": row.get("number", ""),
        "gender": row.get("gender", ""),
        "case": row.get("case", ""),
        "is_dropped": row.get("is_dropped", True),
        "en_reference": row.get("en_reference", ""),
        "shakespeare_text": "",
        "gpt_annotations": "",
    }


def build_export_csv(annotations: list, sentences_df: pd.DataFrame) -> str:
    """Build CSV content for download (no local file)."""
    pronoun_annots = [a for a in annotations if not a.get("no_pronoun")]
    if not pronoun_annots:
        return ""
    rows = []
    for a in pronoun_annots:
        sent = sentences_df[(sentences_df["ID"] == a["ID"]) & (sentences_df["sentence_id"] == a["sentence_id"])]
        if sent.empty:
            continue
        rows.append(pronoun_row_to_output(a, sent.iloc[0]))
    df = pd.DataFrame(rows)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[OUTPUT_COLUMNS]
    return df.to_csv(index=False, encoding="utf-8-sig")


def build_perspectives_csv(perspectives: dict, sentences_df: pd.DataFrame) -> str:
    if not perspectives:
        return ""
    rows = []
    for poem_id, data in perspectives.items():
        meta = sentences_df[sentences_df["ID"].astype(str) == str(poem_id)]
        context = meta["context"].iloc[0] if not meta.empty else ""
        rows.append({
            "ID": poem_id,
            "author": data.get("author", ""),
            "date": data.get("date", ""),
            "perspective_primary": data.get("perspective_primary", ""),
            "perspective_secondary": data.get("perspective_secondary", ""),
            "text": context,
        })
    return pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")


def main():
    st.set_page_config(page_title="Pronoun Annotation", layout="wide")
    st.title("Pronoun Pro-Drop Annotation (Cloud)")

    if not os.path.exists(SENTENCE_FILE):
        st.error(f"Sentence file not found: {SENTENCE_FILE}. Add poems_for_manual_annotation.csv to the app folder.")
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

    sentences_df = load_sentences()  # cached
    if not st.session_state.get("annotations_loaded") or st.session_state.get("annotator_name") != annotator_id:
        st.session_state.annotations = load_annotations(annotator_id)
        st.session_state.poem_perspectives = load_poem_perspectives(annotator_id)
        st.session_state.annotations_loaded = True
        st.session_state["annotator_name"] = annotator_id

    reviewed = get_reviewed_sentences(st.session_state.annotations)

    with st.sidebar:
        st.header("Filter")
        all_authors = sorted(sentences_df["author"].dropna().unique().tolist())
        author_filter = st.multiselect("Author", all_authors, default=all_authors)
        display_df = sentences_df[sentences_df["author"].isin(author_filter)].copy()

        poem_ids = display_df["ID"].unique().tolist()
        poem_id_filter = st.selectbox("Poem ID", ["All"] + sorted(poem_ids, key=str))
        if poem_id_filter != "All":
            display_df = display_df[display_df["ID"] == poem_id_filter]

        # Direct jump by poem ID (search)
        jump_id = st.text_input("Jump to poem by ID", placeholder="e.g. UP324", key="jump_poem_id")
        if jump_id and jump_id.strip():
            match_idx = display_df[display_df["ID"].astype(str) == str(jump_id.strip())].index
            if len(match_idx) > 0:
                first_idx = min(max(0, display_df.index.get_loc(match_idx[0])), len(display_df) - 1)
                if st.button("Go to poem", key="jump_poem_btn"):
                    st.session_state.force_nav_idx = first_idx
                    st.rerun()
            else:
                st.caption("Poem ID not found in current filter.")

        total = len(display_df)
        done = sum(1 for _, r in display_df.iterrows() if _reviewed_key(r["ID"], r["sentence_id"]) in reviewed)
        st.metric("Sentences", total, f"Annotated {done}")
        pronoun_count = sum(1 for a in st.session_state.annotations if not a.get("no_pronoun"))
        st.metric("Pronouns annotated", pronoun_count, "")
        poems_in_view = display_df["ID"].nunique()
        poems_done = sum(1 for pid in display_df["ID"].unique() if is_poem_fully_annotated(str(pid), display_df, reviewed))
        st.metric("Poems fully annotated", poems_done, f"of {poems_in_view}")
        st.metric("Poem perspectives", len(st.session_state.poem_perspectives), "")

        st.divider()
        if "nav_idx" not in st.session_state:
            st.session_state.nav_idx = 0
        max_idx = max(0, total - 1)
        # Clamp nav_idx to valid range (prevents StreamlitValueAboveMaxError when filters change)
        safe_value = min(max(0, st.session_state.nav_idx), max_idx)
        idx = st.number_input("Go to sentence", min_value=0, max_value=max_idx, value=safe_value, step=1, key="nav_idx_input")
        st.session_state.nav_idx = int(idx)
        if st.button("Next →"):
            next_idx = min(int(idx) + 1, total - 1)
            st.session_state.force_nav_idx = next_idx
            st.rerun()

    if display_df.empty:
        st.warning("No data matching filters")
        return

    row = display_df.iloc[idx]
    sent_key = (str(row["ID"]), int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0)
    key_suffix = f"{row['ID']}_{row['sentence_id']}"

    st.subheader(f"Sentence {idx + 1} / {total} · {row['author']} · {row['ID']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Full poem (context)**")
        st.text_area("ctx", value=row["context"], height=180, disabled=True, key=f"ctx_{idx}_{row['ID']}_{row['sentence_id']}", label_visibility="collapsed")
    with col2:
        st.markdown("**Current sentence**")
        st.info(row["sentence"])
        st.caption(f"ID: {row['ID']} | sentence_id: {row['sentence_id']} | Theme: {row.get('Theme', '')}")

    st.divider()

    existing = [a for a in st.session_state.annotations if str(a["ID"]) == str(row["ID"]) and int(a.get("sentence_id", 0)) == int(row["sentence_id"])]
    existing_pronouns = [a for a in existing if not a.get("no_pronoun")]

    default_has = 0 if existing_pronouns else 1
    has_pronoun = st.radio("**Does this sentence have a pronoun?**", ["Yes", "No"], index=default_has, horizontal=True, key=f"has_pronoun_{key_suffix}")

    def _save_sentence_and_reload(records: list):
        save_annotations_for_sentence(annotator_id, str(row["ID"]), int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0, records)
        st.session_state.annotations = load_annotations(annotator_id)

    if has_pronoun == "Yes":
        if existing_pronouns:
            st.caption(f"{len(existing_pronouns)} annotations saved; edit and save to update")
        st.markdown("**Add pronoun(s)**")

        if "current_pronouns" not in st.session_state or st.session_state.get("current_sent_key") != sent_key:
            st.session_state.current_pronouns = [dict(p) for p in existing_pronouns] if existing_pronouns else []
            st.session_state.current_sent_key = sent_key

        pronouns = st.session_state.current_pronouns

        for i, p in enumerate(pronouns):
            with st.expander(f"Pronoun {i+1}: {p.get('pronoun', '')}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    p["pronoun"] = st.text_input("Ukrainian pronoun", value=p.get("pronoun", ""), key=f"p{i}_ukr_{key_suffix}")
                with c2:
                    p["person"] = st.selectbox("Person", PERSON_OPTIONS, index=PERSON_OPTIONS.index(p["person"]) if p.get("person") in PERSON_OPTIONS else 0, key=f"p{i}_person_{key_suffix}")
                    p["number"] = st.selectbox("Number", NUMBER_OPTIONS, index=NUMBER_OPTIONS.index(p["number"]) if p.get("number") in NUMBER_OPTIONS else 0, key=f"p{i}_num_{key_suffix}")
                    p["is_dropped"] = st.radio("Pro-drop?", [True, False], index=0 if p.get("is_dropped", True) else 1, horizontal=True, key=f"p{i}_drop_{key_suffix}")
                if st.button("Delete", key=f"del_{i}_{key_suffix}"):
                    pronouns.pop(i)
                    st.session_state.current_sent_key = None
                    st.rerun()

        if st.button("➕ Add pronoun", key=f"add_pronoun_{key_suffix}"):
            pronouns.append({"pronoun": "", "person": "1st", "number": "Singular", "is_dropped": True})
            st.rerun()

        def _do_save(and_next: bool):
            has_valid = any(p.get("pronoun", "").strip() for p in pronouns)
            if not has_valid and pronouns:
                st.warning("Enter at least one pronoun or select 'No'")
                return
            records = []
            for p in pronouns:
                if p.get("pronoun", "").strip():
                    records.append({
                        "ID": str(row["ID"]),
                        "sentence_id": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                        "pronoun": str(p["pronoun"]).strip(),
                        "lemma": str(p.get("lemma", p["pronoun"])).strip(),
                        "person": str(p.get("person", "")),
                        "number": str(p.get("number", "")),
                        "is_dropped": bool(p.get("is_dropped", True)),
                        "position": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                    })
            _save_sentence_and_reload(records)
            st.session_state.current_pronouns = []
            st.session_state.current_sent_key = None
            if and_next:
                next_idx = min(idx + 1, total - 1)
                st.session_state.force_nav_idx = next_idx
            st.rerun()

        if st.button("💾 Save (stay)", key=f"save_stay_{key_suffix}"):
            try:
                _do_save(and_next=False)
            except Exception as e:
                st.error(f"Save failed: {e}")
        if st.button("Save and next", key=f"save_next_{key_suffix}"):
            try:
                _do_save(and_next=True)
            except Exception as e:
                st.error(f"Save failed: {e}")

    else:
        def _do_save_no_pronoun(and_next: bool):
            records = [{
                "ID": str(row["ID"]),
                "sentence_id": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                "no_pronoun": True,
            }]
            _save_sentence_and_reload(records)
            st.session_state.current_pronouns = []
            st.session_state.current_sent_key = None
            if and_next:
                next_idx = min(idx + 1, total - 1)
                st.session_state.force_nav_idx = next_idx
            st.rerun()

        if st.button("💾 Save (stay)", key=f"save_no_stay_{key_suffix}"):
            try:
                _do_save_no_pronoun(and_next=False)
            except Exception as e:
                st.error(f"Save failed: {e}")
        if st.button("No pronoun, next", key=f"save_no_next_{key_suffix}"):
            try:
                _do_save_no_pronoun(and_next=True)
            except Exception as e:
                st.error(f"Save failed: {e}")

    poem_id = str(row["ID"])
    poem_fully_done = is_poem_fully_annotated(poem_id, display_df, reviewed)

    st.divider()
    st.subheader("Poem Perspective")
    if poem_fully_done:
        st.caption("Poem fully annotated. Judge the overall perspective. Primary required; secondary optional.")
    else:
        st.caption("You can add perspective now. (Recommended: complete all sentence annotations first.)")
    st.markdown("**Full poem**")
    st.text_area("full_poem", value=row["context"], height=200, disabled=True, key=f"full_poem_{poem_id}", label_visibility="collapsed")
    current = st.session_state.poem_perspectives.get(poem_id, {})
    primary = current.get("perspective_primary", current.get("perspective", ""))
    secondary = current.get("perspective_secondary", "")
    legacy_map = {"第一人称": "1st person", "第二人称": "2nd person", "第三人称": "3rd person", "混合": "Mixed", "其他": "Other", "无": "None"}
    primary = legacy_map.get(primary, primary)
    secondary = legacy_map.get(secondary, secondary) if secondary else "None"
    idx_primary = PERSPECTIVE_OPTIONS.index(primary) if primary in PERSPECTIVE_OPTIONS else 0
    idx_secondary = PERSPECTIVE_SECONDARY_OPTIONS.index(secondary) if secondary in PERSPECTIVE_SECONDARY_OPTIONS else 0
    new_primary = st.selectbox("Primary perspective", PERSPECTIVE_OPTIONS, index=idx_primary, key=f"perspective_primary_{poem_id}")
    new_secondary = st.selectbox("Secondary perspective (optional)", PERSPECTIVE_SECONDARY_OPTIONS, index=idx_secondary, key=f"perspective_secondary_{poem_id}")
    if st.button("Save poem perspective", key=f"save_perspective_{poem_id}"):
        try:
            data = {
                "perspective_primary": new_primary,
                "perspective_secondary": new_secondary if new_secondary != "None" else "",
                "author": row.get("author", ""),
                "date": row.get("date", ""),
            }
            save_poem_perspective(annotator_id, poem_id, data)
            st.session_state.poem_perspectives[poem_id] = data
            st.success("Saved")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.divider()
    pronoun_annots = [a for a in st.session_state.annotations if not a.get("no_pronoun")]
    csv_pronoun = build_export_csv(st.session_state.annotations, sentences_df)
    if csv_pronoun:
        st.download_button(
            "Download pronoun annotations (CSV)",
            csv_pronoun,
            file_name=f"manual_annotation_result_{annotator_id.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No pronoun annotations to download yet")

    csv_persp = build_perspectives_csv(st.session_state.poem_perspectives, sentences_df)
    if csv_persp:
        st.download_button(
            "Download poem perspectives (CSV)",
            csv_persp,
            file_name=f"manual_annotation_poem_perspectives_{annotator_id.replace(' ', '_')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No poem perspectives to download yet")

    with st.expander("Preview annotations"):
        if pronoun_annots:
            preview = pd.DataFrame(pronoun_annots[-100:])
            cols = ["ID", "sentence_id", "pronoun", "person", "is_dropped"]
            st.dataframe(preview[[c for c in cols if c in preview.columns]], width="stretch")
        else:
            st.info("No annotations yet")
    with st.expander("Preview poem perspectives"):
        if st.session_state.poem_perspectives:
            persp_data = [
                {"ID": k, "author": v.get("author", ""), "Primary": v.get("perspective_primary", ""), "Secondary": v.get("perspective_secondary", "")}
                for k, v in st.session_state.poem_perspectives.items()
            ]
            st.dataframe(pd.DataFrame(persp_data), width="stretch")
        else:
            st.info("No poem perspectives yet")


if __name__ == "__main__":
    main()
