import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import os
import re

# --- CONFIG ---
CACHE_DIR = "../hf_cache" 
OUTPUT_PATH = "./data/boomer_20k.csv"
BASE_DATA_PATH = "../step2_mixed/data/text2emoji_100k.csv"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# 1. Load Source Data
df = pd.read_csv(BASE_DATA_PATH)
# We take a random 20k sample (it's okay if it overlaps with the Gen Z source rows)
candidates = df.sample(n=20000, random_state=999).reset_index(drop=True)

# Resume logic
if os.path.exists(OUTPUT_PATH):
    try:
        existing = pd.read_csv(OUTPUT_PATH)
        start_idx = len(existing)
        print(f"Resuming from row {start_idx}...")
    except:
        start_idx = 0
else:
    start_idx = 0
    pd.DataFrame(columns=['emoji', 'text', 'topic']).to_csv(OUTPUT_PATH, index=False)

candidates = candidates.iloc[start_idx:]

# 2. Setup Model
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

# 3. Prompt
def create_boomer_prompt(text):
    instr = "You are a Boomer translator. Rewrite the input sentence using formal language, capitalization, and enthusiastic punctuation."
    messages = [{"role": "user", "content": f"{instr}\nInput: \"{text}\"\nOutput ONLY the rewritten sentence:"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def clean_output(text):
    if not isinstance(text, str): return ""
    text = text.replace("<|end|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    text = text.replace("Here is the rewrite:", "").replace("Sure:", "")
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 4. Generation Loop
print("Generating 20k Boomer examples...")
buffer = []

for index, row in candidates.iterrows():
    prompt = create_boomer_prompt(row['text'])
    try:
        outputs = pipe(prompt, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        raw_output = outputs[0]['generated_text']
        
        if "<|assistant|>" in raw_output:
            gen_text = raw_output.split("<|assistant|>")[-1]
        else:
            gen_text = raw_output

        final_text = clean_output(gen_text)
        
        if len(final_text) > 2:
            buffer.append({
                'emoji': f"<BOOMER> {row['emoji']}",
                'text': final_text,
                'topic': "boomer"
            })
    except Exception as e:
        print(f"Error: {e}")

    # Save every 50
    if len(buffer) >= 50:
        pd.DataFrame(buffer).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        print(f"Generated {len(buffer)} (Batch saved)...")
        buffer = [] 

# Final Save
if buffer:
    pd.DataFrame(buffer).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)

print("Boomer Generation Complete!")
