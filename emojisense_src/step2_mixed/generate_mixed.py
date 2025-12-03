import pandas as pd
from datasets import load_dataset
import random
import emoji
import re
import os

# --- CONFIG ---
# We save the data into the 'data' subfolder
OUTPUT_PATH = "./data/text2emoji_100k.csv"
# We reference the cache from the parent folder to save space/time
CACHE_DIR = "../hf_cache" 

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

# 1. Load Original Dataset
print("Loading Text2Emoji dataset...")
dataset = load_dataset("KomeijiForce/Text2Emoji", split="train", cache_dir=CACHE_DIR)
df = dataset.to_pandas()

# Filter bad rows
df = df[df['text'].apply(lambda x: isinstance(x, str))]
df = df[df['emoji'].apply(lambda x: isinstance(x, str))]

# --- 2. SAMPLE ORIGINAL (Take 70k) ---
print("Sampling original data...")
df_original = df.sample(n=70000, random_state=42).reset_index(drop=True)

# --- 3. GENERATE MASHUPS (Target 15k) ---
print("Generating Mashups...")
shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
half = len(shuffled) // 2
part1 = shuffled.iloc[:half].reset_index(drop=True)
part2 = shuffled.iloc[half:2*half].reset_index(drop=True)

mashup_df = pd.DataFrame({
    'emoji': part1['emoji'] + part2['emoji'],
    'text': part1['text'] + " and " + part2['text'],
    'topic': ['mashup'] * len(part1)
})
mashup_df = mashup_df.head(15000)

# --- 4. MIXED MODALITY (Target 15k) ---
print("Generating Mixed Modality...")
word_to_emoji = {}
for emoji_char, data in emoji.EMOJI_DATA.items():
    if 'en' in data:
        clean_word = data['en'].strip(':').replace('_', ' ')
        word_to_emoji[clean_word] = emoji_char

def inject_emojis(text):
    words = text.split()
    new_words = []
    replaced = False
    for word in words:
        clean = re.sub(r'[^\w\s]', '', word).lower()
        if clean in word_to_emoji and random.random() < 0.3:
            new_words.append(word_to_emoji[clean])
            replaced = True
        else:
            new_words.append(word)
    return " ".join(new_words) if replaced else None

mixed_rows = []
candidates = df.sample(frac=1, random_state=456)['text'].tolist()

for text in candidates:
    new_input = inject_emojis(text)
    if new_input:
        mixed_rows.append({'emoji': new_input, 'text': text, 'topic': 'mixed'})
    if len(mixed_rows) >= 15000:
        break
        
mixed_df = pd.DataFrame(mixed_rows)

# --- 5. MERGE & SAVE ---
final_df = pd.concat([df_original, mashup_df, mixed_df]).sample(frac=1).reset_index(drop=True)
print(f"Final Dataset Size: {len(final_df)}")

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
