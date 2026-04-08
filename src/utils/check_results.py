import json
import io
import os
import sys
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent.parent)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('outputs/01_pronoun_detection/lmstudio_progress.json', 'r', encoding='utf-8') as f:
    progress = json.load(f)

print("=" * 60)
print("LM Studio Processing Progress")
print("=" * 60)
print(f"Last processed index: {progress.get('last_processed_idx', 0)}")
print(f"Total poems processed: {len(progress.get('results', []))}")

# Analyze results
results = progress.get('results', [])
all_items = []
dropped_count = 0
explicit_count = 0

for poem_results in results:
    for item in poem_results:
        all_items.append(item)
        if item.get('is_dropped'):
            dropped_count += 1
        else:
            explicit_count += 1

print(f"\nTotal verb instances analyzed: {len(all_items)}")
print(f"Dropped subjects found: {dropped_count}")
print(f"Explicit subjects: {explicit_count}")

# Show sample dropped subjects
print("\n" + "=" * 60)
print("Sample Dropped Subjects (first 10)")
print("=" * 60)
dropped_samples = [item for item in all_items if item.get('is_dropped')][:10]
for i, item in enumerate(dropped_samples, 1):
    verb = item.get('verb', 'N/A')
    pronoun = item.get('pronoun', 'N/A')
    conf = item.get('confidence', 'N/A')
    reason = item.get('reasoning', 'N/A')
    print(f"\n{i}. Verb: {verb[:40] if len(verb) > 40 else verb}")
    print(f"   Pronoun: {pronoun}")
    print(f"   Confidence: {conf}")
    print(f"   Reasoning: {reason[:80] if len(reason) > 80 else reason}")

# Check final output file
print("\n" + "=" * 60)
print("Final Output File Status")
print("=" * 60)
try:
    df = pd.read_csv('outputs/01_pronoun_detection/ukrainian_pronouns_detailed.csv')
    print(f"Total rows in detailed CSV: {len(df)}")
    if 'method' in df.columns:
        print("\nMethod distribution:")
        print(df['method'].value_counts())
    if 'is_dropped' in df.columns:
        print("\nDropped vs Explicit:")
        print(df['is_dropped'].value_counts())
except Exception as e:
    print(f"Error reading output file: {e}")

