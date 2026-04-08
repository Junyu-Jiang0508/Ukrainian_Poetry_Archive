import pandas as pd
import spacy
import torch
from transformers import MarianMTModel, MarianTokenizer
from simalign import SentenceAligner
import logging
from tqdm import tqdm
import re
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

INPUT_FILE = "outputs/01_pronoun_detection/ukrainian_pronouns_detailed.csv"
OUTPUT_DIR = "outputs/01_pronoun_detection"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ukrainian_pronouns_projection_final.csv",
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class PronounProjectionPipeline:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logging.info(f"Using device: {self.device}")

        model_name = "Helsinki-NLP/opus-mt-uk-en"
        logging.info(f"Loading translation model: {model_name}...")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

        logging.info("Loading SimAligner (mBERT)...")
        self.aligner = SentenceAligner(
            model="bert",
            token_type="bpe",
            matching_methods="i",
            device=self.device,
        )

        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_uk = spacy.load("uk_core_news_sm")
        except OSError:
            logging.error(
                "Missing Spacy models. Run: python -m spacy download "
                "en_core_web_sm uk_core_news_sm"
            )
            raise

        self.uk_explicit_lemmas = {
            'я', 'ми', 'ти', 'ви', 'він', 'вона', 'воно', 'вони',
            'себе',
            'мій', 'твій', 'наш', 'ваш', 'їхній', 
            'свій', 
            'чий',  
            'цей', 'той', 
            'хто', 'що',
            'весь', 'все', 'кожен', 'інший',
            'його', 'її', 'їх'
            }

    def translate(self, text_list):
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        translated = self.model.generate(**inputs)
        return [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]

    def analyze_poem(self, poem_id, uk_text):
        if not uk_text or pd.isna(uk_text):
            return []

        clean_uk_text = re.sub(r'\s+', ' ', str(uk_text)).strip()

        if len(clean_uk_text.split()) > 400:
            logging.warning(
                f"Poem {poem_id} is very long; translation may be truncated."
            )

        try:
            en_text = self.translate([clean_uk_text])[0]
        except Exception as e:
            logging.error(f"Translation error ID {poem_id}: {e}")
            return []

        doc_en = self.nlp_en(en_text)
        doc_uk = self.nlp_uk(clean_uk_text)

        uk_tokens = [t.text for t in doc_uk]
        en_tokens = [t.text for t in doc_en]

        if not uk_tokens or not en_tokens:
            return []

        try:
            align_result = self.aligner.get_word_aligns(uk_tokens, en_tokens)
            alignments = align_result['itermax']
        except Exception as e:
            logging.error(f"Alignment error ID {poem_id}: {e}")
            return []

        results = []

        for en_tok in doc_en:
            if en_tok.pos_ == "PRON":
                if en_tok.lemma_.lower() in [
                    'that',
                    'which',
                    'what',
                    'who',
                    'whom',
                ]:
                    continue

                matched_uk_indices = [
                    uk_i for uk_i, en_i in alignments if en_i == en_tok.i
                ]

                is_dropped = False
                uk_word_text = None
                uk_lemma = None
                uk_pos = None
                uk_idx = None

                if not matched_uk_indices:
                    if en_tok.morph.get('Case') == ['Nom']:
                        is_dropped = True
                        uk_word_text = "<NIL>"
                    else:
                        continue
                else:
                    uk_idx = matched_uk_indices[0]
                    uk_token_obj = doc_uk[uk_idx]
                    uk_word_text = uk_token_obj.text
                    uk_lemma = uk_token_obj.lemma_
                    uk_pos = uk_token_obj.pos_

                    is_explicit = (
                        uk_token_obj.lemma_.lower() in self.uk_explicit_lemmas
                    )

                    if not is_explicit and uk_token_obj.pos_ == 'PUNCT':
                        continue

                    is_dropped = not is_explicit

                if uk_idx is None:
                    context_window = ""
                else:
                    start = max(0, uk_idx - 5)
                    end = min(len(uk_tokens), uk_idx + 6)
                    context_window = " ".join(uk_tokens[start:end])

                person = en_tok.morph.get('Person')
                number = en_tok.morph.get('Number')
                case = en_tok.morph.get('Case')

                person_str = person[0] if person else ''
                number_str = number[0] if number else ''
                case_str = case[0] if case else ''

                results.append({
                    'ID': poem_id,
                    'text': uk_text,
                    'pronoun': uk_word_text,
                    'lemma': uk_lemma,
                    'position': uk_idx,
                    'context': context_window,
                    'person': person_str,
                    'number': number_str,
                    'case': case_str,
                    'gender': en_tok.morph.get('Gender', [''])[0],
                    'is_dropped': is_dropped,
                    'en_reference': en_tok.text,
                    'uk_match_pos': uk_pos,
                    'method': 'projection_translation'
                })

        return results


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df_raw = pd.read_csv(INPUT_FILE)

    print("Deduplicating poems...")
    metadata_cols = ['ID', 'text', 'author', 'date', 'Theme', 'Language']
    cols_to_use = [c for c in metadata_cols if c in df_raw.columns]

    df_unique_poems = df_raw[cols_to_use].drop_duplicates(subset=['ID'])
    print(f"Found {len(df_unique_poems)} unique poems to process.")

    pipeline = PronounProjectionPipeline()

    all_results = []

    print("Starting projection pipeline...")
    for _, row in tqdm(df_unique_poems.iterrows(), total=len(df_unique_poems)):
        poem_id = row['ID']
        text = row['text']

        try:
            poem_results = pipeline.analyze_poem(poem_id, text)
            all_results.extend(poem_results)
        except Exception as e:
            logging.error(f"Failed to process poem {poem_id}: {e}")

    if not all_results:
        print("No pronouns detected.")
        return

    df_results = pd.DataFrame(all_results)

    print("Merging metadata...")
    df_meta = df_unique_poems.set_index('ID')
    cols_to_merge = [c for c in cols_to_use if c not in ['ID', 'text']]

    df_final = df_results.merge(df_meta[cols_to_merge], on='ID', how='left')

    desired_order = [
        'ID',
        'author',
        'date',
        'Language',
        'text',
        'Theme',
        'pronoun',
        'lemma',
        'uk_match_pos',
        'position',
        'context',
        'person',
        'number',
        'gender',
        'case',
        'is_dropped',
        'en_reference',
    ]

    final_cols = [c for c in desired_order if c in df_final.columns]
    df_final = df_final[final_cols]

    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("\nProcessing complete.")
    print(f"Total pronouns detected: {len(df_final)}")
    print(f"Explicit: {len(df_final[~df_final['is_dropped']])}")
    print(f"Dropped (Recovered): {len(df_final[df_final['is_dropped']])}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
