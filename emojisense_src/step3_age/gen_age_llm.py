import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd
import os
import re

# --- CONFIG ---
CACHE_DIR = "../hf_cache" 
OUTPUT_PATH = "./data/age_style_20k.csv"
BASE_DATA_PATH = "../step2_mixed/data/text2emoji_100k.csv"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# --- 1. SETUP DATA & CHECKPOINTING ---
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing data at {OUTPUT_PATH}. Resuming...")
    try:
        existing_df = pd.read_csv(OUTPUT_PATH)
        completed_count = len(existing_df)
    except:
        completed_count = 0
    print(f"Already generated {completed_count} examples.")
else:
    print("Starting fresh...")
    completed_count = 0
    pd.DataFrame(columns=['emoji', 'text', 'topic']).to_csv(OUTPUT_PATH, index=False)

# Load Source Data
df = pd.read_csv(BASE_DATA_PATH)
# Target 20,000 total
candidates = df.sample(n=20000, random_state=42).reset_index(drop=True)
candidates = candidates.iloc[completed_count:] 

if len(candidates) == 0:
    print("All 20,000 examples are already generated! Exiting.")
    exit()

# --- 2. SETUP MODEL ---
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    cache_dir=CACHE_DIR
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

# --- 3. HELPER FUNCTIONS ---

def clean_output(text):
    """Aggressively cleans the LLM output."""
    if not isinstance(text, str): return ""
    
    # Remove system tokens
    text = text.replace("<|end|>", "").replace("<|assistant|>", "").replace("<|user|>", "")
    
    # Remove common chat intros
    text = text.replace("Here is the rewrite:", "").replace("Sure:", "")
    
    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'")
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_prompt(text, style):
    if style == "gen_z":
        sys_msg = "You are a Gen Z translator. Rewrite the input sentence using slang, lowercase, and irony."
    else: 
        sys_msg = "You are a Boomer translator. Rewrite the input sentence using formal language, capitalization, and enthusiastic punctuation."
    
    # STRICT instruction to output ONLY the result
    messages = [
        {"role": "user", "content": f"{sys_msg}\nInput: \"{text}\"\nOutput ONLY the rewritten sentence:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- 4. GENERATION LOOP ---
print(f"Starting generation for {len(candidates)} remaining rows...")

buffer = []
for index, row in candidates.iterrows():
    # Calculate global index (0-9999 = Gen Z, 10000+ = Boomer)
    true_index = completed_count + len(buffer)
    
    if true_index < 10000:
        style = "gen_z"
        tag = "<GEN_Z>"
    else:
        style = "boomer"
        tag = "<BOOMER>"

    prompt = create_prompt(row['text'], style)
    
    try:
        # Lower temp (0.7) reduces hallucinations like "worm blanket"
        outputs = pipe(prompt, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        
        # Robust Parsing: Split by the assistant tag to get ONLY the new text
        raw_output = outputs[0]['generated_text']
        if "<|assistant|>" in raw_output:
            generated_text = raw_output.split("<|assistant|>")[-1]
        else:
            # Fallback: try to just grab the end
            generated_text = raw_output
            
        final_text = clean_output(generated_text)
        
        # Validation: If it's empty or too short, skip/flag it
        if len(final_text) < 2:
            print(f"Warning: Empty generation for row {true_index}")
            continue

        buffer.append({
            'emoji': f"{tag} {row['emoji']}",
            'text': final_text,
            'topic': style
        })
        
    except Exception as e:
        print(f"Error on row {true_index}: {e}")

    # --- SAVE BATCH ---
    if len(buffer) >= 50: # Save more often (every 50)
        pd.DataFrame(buffer).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        print(f"Saved progress: {true_index + 1} / 20000 examples.")
        buffer = [] 

# Save leftovers
if buffer:
    pd.DataFrame(buffer).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)

print("Job Complete!")
