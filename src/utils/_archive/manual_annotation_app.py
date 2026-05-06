"""Manual pronoun annotation app."""
import json
import os

import repo_bootstrap

repo_bootstrap.prepare_repo(__file__)

import numpy as np
import pandas as pd
import streamlit as st

OUTPUT_COLUMNS = [
    "ID", "author", "date", "Language", "text", "Theme",
    "pronoun", "lemma", "uk_match_pos", "position", "context",
    "person", "number", "gender", "case", "is_dropped", "en_reference",
    "shakespeare_text", "gpt_annotations",
]

SENTENCE_FILE = "outputs/01_annotation_pronoun_detection/poems_for_manual_annotation.csv"
OUTPUT_FILE = "outputs/01_annotation_pronoun_detection/manual_annotation_result.csv"
PROGRESS_FILE = "outputs/01_annotation_pronoun_detection/manual_annotation_progress.json"
POEM_PERSPECTIVE_FILE = "outputs/01_annotation_pronoun_detection/manual_annotation_poem_perspectives.json"
POEM_PERSPECTIVE_CSV = "outputs/01_annotation_pronoun_detection/manual_annotation_poem_perspectives.csv"

PERSON_OPTIONS = ["1st", "2nd", "3rd", "Impersonal"]
NUMBER_OPTIONS = ["Singular", "Plural", "None"]
PERSPECTIVE_OPTIONS = ["1st person", "2nd person", "3rd person", "Mixed", "Other"]
PERSPECTIVE_SECONDARY_OPTIONS = ["None"] + PERSPECTIVE_OPTIONS


@st.cache_data
def load_sentences():
    df = pd.read_csv(SENTENCE_FILE)
    return df.reset_index(drop=True)


def load_annotations():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_annotations(annotations):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(annotations), f, ensure_ascii=False, indent=2)


def get_reviewed_sentences(annotations):
    """All (ID, sentence_id) that have been reviewed (with or without pronouns)."""
    return {(str(a["ID"]), int(a.get("sentence_id", 0))) for a in annotations}


def load_poem_perspectives():
    if os.path.exists(POEM_PERSPECTIVE_FILE):
        try:
            with open(POEM_PERSPECTIVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_poem_perspectives(perspectives):
    os.makedirs(os.path.dirname(POEM_PERSPECTIVE_FILE), exist_ok=True)
    with open(POEM_PERSPECTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(perspectives), f, ensure_ascii=False, indent=2)


def is_poem_fully_annotated(poem_id, display_df, reviewed):
    """Check if all sentences of a poem are annotated."""
    poem_sents = display_df[display_df["ID"] == poem_id]
    if poem_sents.empty:
        return False
    for _, r in poem_sents.iterrows():
        key = (str(r["ID"]), int(r["sentence_id"]) if pd.notna(r["sentence_id"]) else 0)
        if key not in reviewed:
            return False
    return True


def pronoun_row_to_output(row, sentence_row):
    """Convert a single pronoun annotation to output format."""
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


def save_final_output(annotations, sentences_df):
    if not annotations:
        return None
    rows = []
    for a in annotations:
        if a.get("no_pronoun"):
            continue                                                        
        sent = sentences_df[(sentences_df["ID"] == a["ID"]) & (sentences_df["sentence_id"] == a["sentence_id"])]
        if sent.empty:
            continue
        sent_row = sent.iloc[0]
        out = pronoun_row_to_output(a, sent_row)
        rows.append(out)
    df = pd.DataFrame(rows)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[OUTPUT_COLUMNS]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    return OUTPUT_FILE


def save_poem_perspectives_csv(perspectives, sentences_df):
    """Export poem perspectives to CSV."""
    if not perspectives:
        return None
    rows = []
    for poem_id, data in perspectives.items():
        meta = sentences_df[sentences_df["ID"].astype(str) == str(poem_id)]
        context = meta["context"].iloc[0] if not meta.empty else ""
        primary = data.get("perspective_primary", data.get("perspective", ""))
        secondary = data.get("perspective_secondary", "")
        rows.append({
            "ID": poem_id,
            "author": data.get("author", ""),
            "date": data.get("date", ""),
            "perspective_primary": primary,
            "perspective_secondary": secondary,
            "text": context,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(POEM_PERSPECTIVE_CSV), exist_ok=True)
    df.to_csv(POEM_PERSPECTIVE_CSV, index=False, encoding="utf-8-sig")
    return POEM_PERSPECTIVE_CSV


def main():
    st.set_page_config(page_title="Pronoun Annotation", layout="wide")
    st.title("Pronoun Pro-Drop Annotation")

    if "force_nav_idx" in st.session_state:
        st.session_state.nav_idx = st.session_state.force_nav_idx
        st.session_state["nav_idx_input"] = st.session_state.force_nav_idx
        del st.session_state.force_nav_idx

    sentences_df = load_sentences()
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    if "poem_perspectives" not in st.session_state:
        st.session_state.poem_perspectives = load_poem_perspectives()

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

        total = len(display_df)
        done = sum(1 for _, r in display_df.iterrows() if (r["ID"], r["sentence_id"]) in reviewed)
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
        idx = st.number_input("Go to sentence", min_value=0, max_value=max(0, total - 1), value=st.session_state.nav_idx, step=1, key="nav_idx_input")
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
            st.session_state.annotations = [a for a in st.session_state.annotations if (str(a["ID"]), int(a.get("sentence_id", 0))) != sent_key]
            for p in pronouns:
                if p.get("pronoun", "").strip():
                    rec = {
                        "ID": str(row["ID"]),
                        "sentence_id": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                        "pronoun": str(p["pronoun"]).strip(),
                        "lemma": str(p.get("lemma", p["pronoun"])).strip(),
                        "person": str(p.get("person", "")),
                        "number": str(p.get("number", "")),
                        "is_dropped": bool(p.get("is_dropped", True)),
                        "position": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                    }
                    st.session_state.annotations.append(rec)
            save_annotations(st.session_state.annotations)
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
                                                              
            st.session_state.annotations = [a for a in st.session_state.annotations if (str(a["ID"]), int(a.get("sentence_id", 0))) != sent_key]
                                                                                               
            st.session_state.annotations.append({
                "ID": str(row["ID"]),
                "sentence_id": int(row["sentence_id"]) if pd.notna(row["sentence_id"]) else 0,
                "no_pronoun": True,
            })
            save_annotations(st.session_state.annotations)
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
    if poem_fully_done:
        st.divider()
        st.subheader("Poem Perspective")
        st.caption("Poem fully annotated. Judge the overall perspective. Primary required; secondary optional.")
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
            st.session_state.poem_perspectives[poem_id] = {
                "perspective_primary": new_primary,
                "perspective_secondary": new_secondary if new_secondary != "None" else "",
                "author": row.get("author", ""),
                "date": row.get("date", ""),
            }
            save_poem_perspectives(st.session_state.poem_perspectives)
            st.success("Saved")

    st.divider()
    if st.button("Export (manual_annotation_result.csv)"):
        path = save_final_output(st.session_state.annotations, sentences_df)
        if path:
            st.success(f"Saved to {path} ({len(st.session_state.annotations)} pronouns)")
        else:
            st.warning("No annotations yet")
    if st.button("Export poem perspectives (manual_annotation_poem_perspectives.csv)"):
        path = save_poem_perspectives_csv(st.session_state.poem_perspectives, sentences_df)
        if path:
            st.success(f"Saved to {path} ({len(st.session_state.poem_perspectives)} poems)")
        else:
            st.warning("No poem perspectives yet")

    with st.expander("Preview annotations"):
        if st.session_state.annotations:
            pronoun_annots = [a for a in st.session_state.annotations if not a.get("no_pronoun")]
            preview = pd.DataFrame(pronoun_annots[-100:])
            cols = ["ID", "sentence_id", "pronoun", "person", "is_dropped"]
            st.dataframe(preview[[c for c in cols if c in preview.columns]], width="stretch")
        else:
            st.info("No annotations yet")
    with st.expander("Preview poem perspectives"):
        if st.session_state.poem_perspectives:
            persp_data = [
                {
                    "ID": k,
                    "author": v.get("author", ""),
                    "Primary": v.get("perspective_primary", v.get("perspective", "")),
                    "Secondary": v.get("perspective_secondary", ""),
                }
                for k, v in st.session_state.poem_perspectives.items()
            ]
            st.dataframe(pd.DataFrame(persp_data), width="stretch")
        else:
            st.info("No poem perspectives yet")


if __name__ == "__main__":
    main()
