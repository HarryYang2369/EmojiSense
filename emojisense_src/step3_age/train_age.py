import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import pandas as pd
import emoji
import os

# --- CONFIG ---
STEP2_DATA_PATH = "../step2_mixed/data/text2emoji_100k.csv"
GENZ_DATA_PATH = "./data/gen_z_20k.csv"   # The renamed file
BOOMER_DATA_PATH = "./data/boomer_20k.csv" # The new file
MODEL_CHECKPOINT = "facebook/bart-large"
CACHE_DIR = "../hf_cache" 
OUTPUT_DIR = "./bart_age_final"

# --- 1. PREPARE SUPER DATASET ---
print("--- PREPARING DATA ---")

# A. Standard Data (80k)
print("Loading Standard data...")
df_std = pd.read_csv(STEP2_DATA_PATH).sample(n=80000, random_state=42).reset_index(drop=True)
df_std['emoji'] = "<STD> " + df_std['emoji'].astype(str)
df_std['topic'] = "standard"

# B. Gen Z Data (20k)
print("Loading Gen Z data...")
df_genz = pd.read_csv(GENZ_DATA_PATH)

# C. Boomer Data (20k)
print("Loading Boomer data...")
df_boomer = pd.read_csv(BOOMER_DATA_PATH)

# D. Merge All
print("Merging 80k STD + 20k GenZ + 20k Boomer...")
df_final = pd.concat([df_std, df_genz, df_boomer]).sample(frac=1).reset_index(drop=True)

print(f"Final Dataset Size: {len(df_final)}") # Should be ~120,000
print("Sample Input:", df_final.iloc[0]['emoji'])
print("Sample Target:", df_final.iloc[0]['text'])

dataset = Dataset.from_pandas(df_final)

# --- 2. SETUP MODEL ---
print("\n--- SETUP MODEL ---")
tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)

# --- 3. ADD SPECIAL TOKENS & SMART INITIALIZATION (THE FIX) ---
special_tokens_dict = {'additional_special_tokens': ['<ZWJ>', '<STD>', '<GEN_Z>', '<BOOMER>']}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added} special tokens: {special_tokens_dict['additional_special_tokens']}")

# Resize first to make room
model.resize_token_embeddings(len(tokenizer))

print("Applying SMART INITIALIZATION to prevent Model Collapse...")
# We copy embeddings from similar existing words so the model doesn't start from random noise
with torch.no_grad():
    embeds = model.get_input_embeddings().weight
    
    # Get IDs for our new tokens
    gen_z_id = tokenizer.convert_tokens_to_ids("<GEN_Z>")
    boomer_id = tokenizer.convert_tokens_to_ids("<BOOMER>")
    std_id = tokenizer.convert_tokens_to_ids("<STD>")
    
    # Get IDs for proxy words
    teen_id = tokenizer.convert_tokens_to_ids("teenager")
    formal_id = tokenizer.convert_tokens_to_ids("formal")
    standard_id = tokenizer.convert_tokens_to_ids("standard")
    
    # Clone weights
    embeds[gen_z_id] = embeds[teen_id].clone()
    embeds[boomer_id] = embeds[formal_id].clone()
    embeds[std_id] = embeds[standard_id].clone()

print("Smart Initialization Complete.")

# --- 4. ADD EMOJIS TO VOCAB ---
print("Scanning for unique emojis...")
all_text = "".join(df_final['emoji'].astype(str))
unique_chars = set(all_text)
emojis_to_add = [char for char in unique_chars if emoji.is_emoji(char)]
print(f"Adding {len(emojis_to_add)} emojis to tokenizer vocabulary.")
tokenizer.add_tokens(emojis_to_add)
model.resize_token_embeddings(len(tokenizer))

# --- 5. PREPROCESSING ---
def paper_style_emoji_split(text):
    processed_text = ""
    for char in text:
        if char == '\u200d':
            processed_text += " <ZWJ> "
        else:
            processed_text += char
    return processed_text

def preprocess_function(examples):
    inputs = [paper_style_emoji_split(ex) for ex in examples["emoji"]]
    targets = examples["text"]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=8)

# --- 6. TRAINING (GENTLE SETTINGS) ---
split_data = tokenized_datasets.train_test_split(test_size=0.05)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    
    # --- STABILITY FIXES ---
    learning_rate=1e-5,      # Lower LR to prevent shock
    weight_decay=0.05,       # Higher decay to keep weights distinct
    warmup_steps=1000,       # Slow warmup
    max_grad_norm=0.5,       # Gradient Clipping (Prevents explosion)
    num_train_epochs=5,      # Reduced epochs (Safe range)
    # -----------------------
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True, 
    report_to="none",
    dataloader_num_workers=8
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_data['train'],
    eval_dataset=split_data['test'],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

print("Starting Training...")
trainer.train()

# 7. Save Final Model
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Step 3 Training Complete!")
