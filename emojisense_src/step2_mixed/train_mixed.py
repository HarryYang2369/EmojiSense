import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import emoji
import os

# --- CONFIG ---
DATA_FILE = "./data/text2emoji_100k.csv"
MODEL_CHECKPOINT = "facebook/bart-large"
# Point to cache in the ROOT directory
CACHE_DIR = "../hf_cache" 
OUTPUT_DIR = "./bart_mixed_final"

# 1. Setup
tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
tokenizer.add_special_tokens({'additional_special_tokens': ['<ZWJ>']})

# 2. Load Data from CSV
print(f"Loading data from {DATA_FILE}...")
dataset = load_dataset("csv", data_files=DATA_FILE, cache_dir=CACHE_DIR)
dataset = dataset.filter(lambda x: isinstance(x["text"], str) and isinstance(x["emoji"], str))

# 3. Add Emojis to Vocabulary
print("Scanning for unique emojis...")
all_input_text = "".join(dataset['train']['emoji'])
unique_chars = set(all_input_text)
# CRITICAL: Only add characters that are actually emojis.
# Since our input now contains English letters (from mixed text), we must filter strictly.
emojis_to_add = [char for char in unique_chars if emoji.is_emoji(char)]

print(f"Adding {len(emojis_to_add)} emojis to tokenizer.")
tokenizer.add_tokens(emojis_to_add)
model.resize_token_embeddings(len(tokenizer))

# 4. Preprocessing
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

tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=8)

# 5. Training
split_data = tokenized_datasets['train'].train_test_split(test_size=0.05)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_steps=500,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True, 
    report_to="none",
    dataloader_num_workers=4
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

trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Training Complete!")
