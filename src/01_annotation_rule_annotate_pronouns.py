import pandas as pd
import re

from utils.workspace import prepare_analysis_environment
from utils.stage_io import read_csv_artifact, stage_output_dir, write_csv_artifact

ROOT = prepare_analysis_environment(__file__, matplotlib_backend=None)
DEFAULT_STAGE_DIR = stage_output_dir("01_annotation_pronoun_detection", root=ROOT)
DEFAULT_INPUT_PATH = DEFAULT_STAGE_DIR / "ukrainian_pronouns_detailed.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_STAGE_DIR / "ukrainian_pronouns_annotated_50.csv"

df = read_csv_artifact(DEFAULT_INPUT_PATH, encoding="utf-8")

df_pronouns = df[df["pronoun"].notna()].copy()

df_50 = df_pronouns.head(50).copy()

df_50['referent_category'] = ''
df_50['referent_confidence'] = ''
df_50['we_inclusivity'] = ''
df_50['we_incl_confidence'] = ''
df_50['token_notes'] = ''
df_50['syntactic_position'] = ''
df_50['semantic_role'] = ''
df_50['discourse_function'] = ''
df_50['annotation_difficulty'] = ''
df_50['polyphony_type'] = ''
df_50['polyphony_notes'] = ''

def annotate_pronoun(row):
    pronoun = str(row['pronoun']).lower()
    context = str(row['context']).lower()
    text = str(row['text']).lower()
    person = row['person']
    number = row['number']
    date = str(row['date'])
    theme_str = str(row['Theme'])
    
    is_post_2022 = date >= "2022-01-01"

    referent = 'UNCERTAIN'
    confidence = 2
    inclusivity = ''
    incl_confidence = ''
    notes = ''
    difficulty = 2
    
    if 'наш' in pronoun or 'мій' in pronoun or 'твій' in pronoun or 'їх' in pronoun or 'його' in pronoun or 'її' in pronoun or 'ваш' in pronoun:
        syntactic = 'POSSESSIVE'
    elif re.search(r'\b(про|з|від|до|на|в|у|для|без|під|над|за|перед|між|поміж)\s+' + pronoun, context):
        syntactic = 'PREPOSITIONAL'
    else:
        if person == 2.0 or pronoun in ['ти', 'ви']:
            syntactic = 'SUBJECT'
        else:
            syntactic = 'SUBJECT'
    
    if person == 2.0:
        if "war" in theme_str or "війна" in text or "окупант" in text or "ворог" in text:
            referent = "INTIMATE"
            confidence = 2
            notes = "2nd person, war context: intimate or reader"
        else:
            referent = "INTIMATE"
            confidence = 3
            notes = "2nd person: intimate or addressee"

    elif pronoun in ["ми", "нас", "нам", "нами", "наш", "наша", "наше", "наші"]:

        national_markers = ['українці', 'народ', 'нація', 'україна', 'українська', 'український']
        enemy_markers = ['росіяни', 'окупанти', 'орки', 'москалі', 'рашисти', 'вони прийшли', 'вони бомблять', 'вони вбивають']
        intimate_markers = ['мама', 'тато', 'батько', 'діти', 'родина', 'кохан', 'дружина', 'чоловік', 'син', 'дочка']
        local_markers = ['київ', 'львів', 'харків', 'маріуполь', 'буча', 'місто', 'село', 'рота', 'батальйон', 'взвод']
        
        if any(marker in text for marker in national_markers) or any(marker in context for marker in national_markers):
            if is_post_2022:
                referent = 'NATION_POST2022'
            else:
                referent = 'NATION_PRE2022'
            confidence = 3
            inclusivity = 'INCLUSIVE'
            incl_confidence = 3
            notes = "explicit national collective"

        elif "war" in theme_str or "війна" in text:
            if any(marker in text for marker in enemy_markers):
                referent = 'ENEMY'
                confidence = 3
                notes = "explicit enemy referent"
            else:
                referent = "NATION"
                confidence = 2
                inclusivity = "AMBIGUOUS"
                incl_confidence = 2
                notes = "war context, collective"

        elif any(marker in text for marker in intimate_markers) or any(marker in context for marker in intimate_markers):
            referent = 'INTIMATE'
            confidence = 3
            inclusivity = 'EXCLUSIVE'
            incl_confidence = 3
            notes = "family/intimate circle"

        elif any(marker in text for marker in local_markers) or any(marker in context for marker in local_markers):
            referent = 'LOCAL'
            confidence = 2
            inclusivity = 'AMBIGUOUS'
            incl_confidence = 2
            notes = "local community"

        else:
            if 'death' in theme_str or 'home' in theme_str:
                referent = 'INTIMATE'
                confidence = 2
            else:
                referent = 'UNCERTAIN'
                confidence = 1
            inclusivity = 'AMBIGUOUS'
            incl_confidence = 1
            notes = "needs more context"
            difficulty = 3

    elif person == 1.0 and number == "Sing":
        referent = 'SELF'
        confidence = 3
        notes = "1sg lyric self"

    elif person == 3.0 or pronoun in ["він", "вона", "воно", "вони", "їх", "його", "її"]:
        enemy_markers = ['росіяни', 'окупанти', 'орки', 'москалі', 'рашисти']
        state_markers = ['уряд', 'президент', 'влада', 'держава', 'міністерство']
        
        if any(marker in text for marker in enemy_markers):
            referent = 'ENEMY_EXPLICIT'
            confidence = 3
            notes = "explicit enemy"
        elif any(marker in text for marker in state_markers):
            referent = "STATE"
            confidence = 2
            notes = "state institutions"
        elif "war" in theme_str and ("бомблять" in text or "вбивають" in text or "прийшли" in text):
            referent = "ENEMY_IMPLICIT"
            confidence = 2
            notes = "implicit enemy"
        else:
            referent = "UNCERTAIN"
            confidence = 1
            notes = "3rd person needs context"
            difficulty = 3

    action_verbs = ['будуємо', 'боремося', 'захищаємо', 'творимо', 'робимо', 'стоїмо']
    passive_verbs = ['бомблять', 'вбивають', 'руйнують', 'нищать']
    psych_verbs = ['відчуваємо', 'пам\'ятаємо', 'знаємо', 'розуміємо', 'бачимо']
    
    if any(verb in text for verb in action_verbs):
        semantic = 'AGENT'
    elif any(verb in text for verb in passive_verbs):
        semantic = 'PATIENT'
    elif any(verb in text for verb in psych_verbs):
        semantic = 'EXPERIENCER'
    else:
        semantic = 'AGENT'
    
    discourse = "TOPIC_CONTINUATION"
    
    return pd.Series({
        'referent_category': referent,
        'referent_confidence': confidence,
        'we_inclusivity': inclusivity,
        'we_incl_confidence': incl_confidence,
        'token_notes': notes,
        'syntactic_position': syntactic,
        'semantic_role': semantic,
        'discourse_function': discourse,
        'annotation_difficulty': difficulty,
        'polyphony_type': '',
        'polyphony_notes': ''
    })

annotations = df_50.apply(annotate_pronoun, axis=1)
for col in annotations.columns:
    df_50[col] = annotations[col]

write_csv_artifact(df_50, DEFAULT_OUTPUT_PATH, index=False, encoding="utf-8")

print(f"saved {DEFAULT_OUTPUT_PATH} rows={len(df_50)}")

