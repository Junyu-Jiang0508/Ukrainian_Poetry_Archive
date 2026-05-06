import pandas as pd
import spacy
import stanza
import os
import json
import time
from collections import Counter
import matplotlib.pyplot as plt

from utils.workspace import prepare_analysis_environment
from utils.stage_io import read_csv_artifact, stage_output_dir, write_csv_artifact

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)
DEFAULT_STAGE_DIR = stage_output_dir("01_annotation_pronoun_detection", root=ROOT)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[INFO] requests package not installed")

INPUT_PATH = ROOT / "outputs" / "00_filtering" / "ukrainian_filtered.csv"
OUTPUT_DIR = DEFAULT_STAGE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_uk = read_csv_artifact(INPUT_PATH)
print(f"Loaded {len(df_uk)} pre-filtered Ukrainian poems")

detailed_path = os.path.join(OUTPUT_DIR, "ukrainian_pronouns_detailed.csv")
if os.path.exists(detailed_path):
    try:
        df_detailed = read_csv_artifact(detailed_path)
        if 'Theme' in df_detailed.columns:
            theme_map = df_detailed.groupby('ID')['Theme'].first().to_dict()
            df_uk['Theme'] = df_uk['ID'].map(theme_map)
            print(f"Loaded theme information for {len(theme_map)} poems")
    except Exception as e:
        print(f"Could not load theme from detailed file: {e}")

try:
    nlp = spacy.load("uk_core_news_sm")
except Exception:
    print("Please install model first: python -m spacy download uk_core_news_sm")
    raise

print("[INFO] Initializing Stanza pipeline...")
try:
    nlp_stanza = stanza.Pipeline(
        lang='uk', 
        processors='tokenize,pos,lemma,depparse', 
        download_method=None
    )
    print("[INFO] Stanza pipeline initialized")
except Exception as e:
    print(f"Warning: Could not initialize Stanza pipeline: {e}")
    nlp_stanza = None

LMSTUDIO_API_URL = os.getenv(
    "LMSTUDIO_API_URL", 
    "http://localhost:1234/v1/chat/completions"
)
lmstudio_available = False
if REQUESTS_AVAILABLE:
    try:
        test_response = requests.get(
            "http://localhost:1234/v1/models", 
            timeout=2
        )
        if test_response.status_code == 200:
            lmstudio_available = True
            print("[INFO] LM Studio API available")
        else:
            print("[INFO] LM Studio server not responding")
    except Exception:
        print("[INFO] LM Studio not running")
else:
    print("[INFO] requests package not available")

def extract_pronoun_features(text, window=5):
    if pd.isna(text):
        return []
    doc = nlp(str(text))
    tokens = [t.text for t in doc]
    results = []
    for i, token in enumerate(doc):
        if token.pos_ == "PRON":
            feats = token.morph.to_dict()
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            context = " ".join(tokens[start:end])
            results.append({
                "pronoun": token.text.lower(),
                "lemma": token.lemma_.lower(),
                "position": i,
                "context": context,
                "person": feats.get("Person", None),
                "number": feats.get("Number", None),
                "gender": feats.get("Gender", None),
                "case": feats.get("Case", None),
                "is_dropped": False,
                "method": "spacy"
            })
    return results

def analyze_dropped_subjects(text, theme=None):
    if pd.isna(text) or nlp_stanza is None:
        return []
    
    try:
        doc = nlp_stanza(str(text))
    except Exception as e:
        return []
    
    results = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == "VERB":
                has_explicit_subject = False
                for w in sentence.words:
                    if w.head == word.id and w.deprel == "nsubj":
                        has_explicit_subject = True
                        break

                if has_explicit_subject:
                    continue

                if not word.feats:
                    continue
                
                feats_dict = {}
                if word.feats:
                    for item in word.feats.split('|'):
                        if '=' in item:
                            key, value = item.split('=', 1)
                            feats_dict[key] = value
                
                tense = feats_dict.get('Tense', 'Unknown')
                person = feats_dict.get('Person', None)
                number = feats_dict.get('Number', None)
                gender = feats_dict.get('Gender', None)
                
                implied_pronoun = "Unknown"
                pronoun_lemma = None
                
                if person and number:
                    if person == '1':
                        if number == 'Sing':
                            implied_pronoun = "Я (I)"
                            pronoun_lemma = "я"
                        elif number == 'Plur':
                            implied_pronoun = "Ми (We)"
                            pronoun_lemma = "ми"
                    elif person == '2':
                        if number == 'Sing':
                            implied_pronoun = "Ти (You)"
                            pronoun_lemma = "ти"
                        elif number == 'Plur':
                            implied_pronoun = "Ви (You pl.)"
                            pronoun_lemma = "ви"
                    elif person == '3':
                        if number == 'Sing':
                            implied_pronoun = "Він/Вона/Воно (He/She/It)"
                            pronoun_lemma = "він/вона/воно"
                        elif number == 'Plur':
                            implied_pronoun = "Вони (They)"
                            pronoun_lemma = "вони"
                
                elif tense == 'Past' and gender:
                    if number == 'Sing':
                        if gender == 'Masc':
                            implied_pronoun = "Він (He) / Я (I) / Ти (You)"
                            pronoun_lemma = "він/я/ти"
                        elif gender == 'Fem':
                            implied_pronoun = "Вона (She) / Я (I) / Ти (You)"
                            pronoun_lemma = "вона/я/ти"
                        elif gender == 'Neut':
                            implied_pronoun = "Воно (It)"
                            pronoun_lemma = "воно"
                    elif number == 'Plur':
                        implied_pronoun = "Вони/Ми/Ви (They/We/You)"
                        pronoun_lemma = "вони/ми/ви"
                
                sentence_tokens = [w.text for w in sentence.words]
                verb_index = next((i for i, w in enumerate(sentence.words) if w.id == word.id), -1)
                window = 5
                start = max(0, verb_index - window)
                end = min(len(sentence_tokens), verb_index + window + 1)
                context = " ".join(sentence_tokens[start:end])
                
                results.append({
                    "pronoun": implied_pronoun,
                    "lemma": pronoun_lemma,
                    "position": verb_index,
                    "context": context,
                    "person": person,
                    "number": number,
                    "gender": gender,
                    "case": None,
                    "is_dropped": True,
                    "verb": word.text,
                    "verb_lemma": word.lemma,
                    "tense": tense,
                    "method": "stanza"
                })
    
    return results

def analyze_dropped_subjects_with_lmstudio(text, theme=None):
    if pd.isna(text) or not lmstudio_available:
        return []
    
    try:
        text_limited = str(text)[:300] if len(str(text)) > 300 else str(text)
        
        theme_context = ""
        if theme:
            theme_str = str(theme)
            if isinstance(theme, list):
                theme_str = ', '.join(theme)
            theme_context = f"\nTheme: {theme_str}"
        
        prompt = f"""Analyze Ukrainian poem for dropped subjects (pro-drop).

Rules:
- Present/Future: Person+Number marking (unambiguous)
- Past: Gender+Number (ambiguous for person)
  * Past.Masc.Sg (-в): Він/Я/Ти
  * Past.Fem.Sg (-ла): Вона/Я
  * Past.Plur (-ли): Вони/Ми/Ви
- Imperative: usually Ти/Ви
- Use context to resolve past tense ambiguity{theme_context}

Output JSON only:
{{
  "verbs": [
    {{
      "verb": "word",
      "subject_status": "DROPPED" | "EXPLICIT",
      "explicit_subject": null | "text",
      "implied_pronoun_primary": "Я|Ти|Він|Вона|Воно|Ми|Ви|Вони",
      "implied_pronoun_alternatives": ["..."],
      "confidence": "High|Medium|Low",
      "reasoning": "brief explanation",
      "sentence": "full sentence"
    }}
  ]
}}

Poem:
{text_limited}"""
        
        request_data = {
            "model": "qwen2.5-14b-instruct",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a Ukrainian linguist. Output JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": -1,
            "stream": False
        }
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    LMSTUDIO_API_URL,
                    json=request_data,
                    timeout=180
                )
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
                else:
                    print(f"LM Studio request timeout after {max_retries} attempts")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"LM Studio request error: {e}")
                return []
        
        if response.status_code != 200:
            try:
                error_detail = response.json()
                print(f"LM Studio API error {response.status_code}: {error_detail}")
            except Exception:
                print(f"LM Studio API error {response.status_code}: {response.text[:200]}")
            return []
        
        try:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            if not content.strip():
                return []
            
            if content.strip().startswith('{'):
                parsed = json.loads(content)
            else:
                parsed = json.loads(content)
            
            verbs_data = parsed.get('verbs', [])
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {content[:300] if 'content' in locals() else 'N/A'}")
            return []
        except KeyError as e:
            print(f"Response format error: {e}")
            return []
        
        results = []
        for item in verbs_data:
            if not isinstance(item, dict):
                continue
            
            verb = item.get('verb', '')
            subject_status = item.get('subject_status', 'UNKNOWN')
            is_dropped = (subject_status == 'DROPPED')
            
            if is_dropped:
                primary = item.get('implied_pronoun_primary', 'Unknown')
                alternatives = item.get('implied_pronoun_alternatives', [])
                
                pronoun_lemma = None
                pronoun_display = primary
                if primary:
                    lemma_map = {
                        'Я': 'я', 'I': 'я',
                        'Ти': 'ти', 'You': 'ти',
                        'Він': 'він', 'He': 'він',
                        'Вона': 'вона', 'She': 'вона',
                        'Воно': 'воно', 'It': 'воно',
                        'Ми': 'ми', 'We': 'ми',
                        'Ви': 'ви', 'You-pl': 'ви',
                        'Вони': 'вони', 'They': 'вони'
                    }
                    for key, val in lemma_map.items():
                        if key in primary:
                            pronoun_lemma = val
                            break
                
                if alternatives:
                    pronoun_display = f"{primary} (alt: {', '.join(alternatives)})"
            else:
                pronoun_display = item.get('explicit_subject', 'EXPLICIT')
                pronoun_lemma = None
            
            results.append({
                "pronoun": pronoun_display,
                "lemma": pronoun_lemma,
                "position": None,
                "context": item.get('sentence', text_limited[:100]),
                "person": None,
                "number": None,
                "gender": None,
                "case": None,
                "is_dropped": is_dropped,
                "verb": verb,
                "verb_form": item.get('verb_form', None),
                "verb_lemma": None,
                "tense": None,
                "confidence": item.get('confidence', 'Unknown'),
                "reasoning": item.get('reasoning', ''),
                "alternatives": alternatives if is_dropped else [],
                "method": "lmstudio_enhanced"
            })
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        if 'content' in locals():
            print(f"Raw response: {content[:200]}...")
        return []
    except Exception as e:
        print(f"LM Studio API error: {e}")
        return []


print("[INFO] Detecting explicit pronouns...")
df_uk["pronoun_details"] = df_uk["text"].apply(extract_pronoun_features)

dropped_subjects_list = []
if nlp_stanza is not None:
    print("[INFO] Detecting dropped subjects with Stanza...")
    df_uk["dropped_subjects_stanza"] = df_uk.apply(
        lambda row: analyze_dropped_subjects(
            row["text"], 
            row.get("Theme", None)
        ), 
        axis=1
    )
    dropped_subjects_list.append("dropped_subjects_stanza")
else:
    df_uk["dropped_subjects_stanza"] = df_uk.apply(lambda row: [], axis=1)

if lmstudio_available:
    print("[INFO] Enhancing with LM Studio...")
    BATCH_SIZE = 3
    progress_file = os.path.join(OUTPUT_DIR, "lmstudio_progress.json")
    
    start_idx = 0
    lmstudio_results = []
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                start_idx = progress.get('last_processed_idx', 0) + 1
                lmstudio_results = progress.get('results', [])
                print(f"[INFO] Resuming from index {start_idx} (found {len(lmstudio_results)} previous results)")
        except Exception as e:
            print(f"[INFO] Could not load progress: {e}, starting from beginning")
    
    total_batches = (len(df_uk) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_start in range(start_idx, len(df_uk), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df_uk))
        batch_num = (batch_start - start_idx) // BATCH_SIZE + 1
        print(f"Batch {batch_num}/{total_batches}: {batch_start+1}-{batch_end}/{len(df_uk)}")
        
        for idx in range(batch_start, batch_end):
            row = df_uk.iloc[idx]
            result = analyze_dropped_subjects_with_lmstudio(
                row.get("text", ""), 
                row.get("Theme", None)
            )
            lmstudio_results.append(result)
            
            if idx < batch_end - 1:
                time.sleep(0.1)
        
        progress = {
            'last_processed_idx': batch_end - 1,
            'results': lmstudio_results
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        
        if batch_end < len(df_uk):
            time.sleep(0.5)
    
    df_uk["dropped_subjects_lmstudio"] = lmstudio_results
    dropped_subjects_list.append("dropped_subjects_lmstudio")
    
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    print("[INFO] LM Studio analysis completed")
else:
    df_uk["dropped_subjects_lmstudio"] = df_uk.apply(lambda row: [], axis=1)

def merge_all_pronouns(row):
    all_results = row["pronoun_details"].copy()
    for col in dropped_subjects_list:
        if col in row and row[col]:
            all_results.extend(row[col])
    return all_results

df_uk["all_pronoun_details"] = df_uk.apply(merge_all_pronouns, axis=1)

df_expanded = df_uk.explode("all_pronoun_details").reset_index(drop=True)

cols_to_drop = ["pronoun_details", "all_pronoun_details"] + dropped_subjects_list
df_expanded = pd.concat(
    [
        df_expanded.drop(columns=[c for c in cols_to_drop if c in df_expanded.columns]),
        df_expanded["all_pronoun_details"].apply(pd.Series)
    ],
    axis=1
)

if 'Theme' not in df_expanded.columns and 'Theme' in df_uk.columns:
    theme_map = df_uk.set_index('ID')['Theme'].to_dict()
    df_expanded['Theme'] = df_expanded['ID'].map(theme_map)

freq = Counter(df_expanded["lemma"].dropna())
print("\n[Top 10 Lemmas]")
for p, c in freq.most_common(10):
    print(f"{p:<10} {c}")

if 'is_dropped' in df_expanded.columns:
    dropped_count = df_expanded['is_dropped'].sum()
    print(f"\n[Dropped Subjects] Found {dropped_count} dropped subjects out of {len(df_expanded)} total pronoun instances")

detailed_csv = OUTPUT_DIR / "ukrainian_pronouns_detailed.csv"
write_csv_artifact(df_expanded, detailed_csv, index=False, encoding="utf-8")
print(f"[INFO] Saved detailed results to {detailed_csv}")

summary = (
    df_expanded.groupby(["lemma", "person", "number"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
summary_csv = OUTPUT_DIR / "ukrainian_pronouns_summary.csv"
write_csv_artifact(summary, summary_csv, index=False, encoding="utf-8")
print(f"[INFO] Saved summary results to {summary_csv}")

summary_matrix = df_expanded.groupby(["person", "number"]).size().unstack(fill_value=0)
print("\n[Person × Number Table]\n", summary_matrix)

summary_matrix.plot(kind="bar", stacked=False, figsize=(6,4))
plt.title("Distribution of Pronouns by Person and Number")
plt.xlabel("Person")
plt.ylabel("Frequency")
plt.tight_layout()
plot_path = OUTPUT_DIR / "pronoun_person_number.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print("[DONE] Full pronoun detection and morphological tagging completed.")
