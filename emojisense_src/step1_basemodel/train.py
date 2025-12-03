import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import emoji
import os

# --- CONFIGURATION ---
DATASET_NAME = "KomeijiForce/Text2Emoji" 
MODEL_CHECKPOINT = "facebook/bart-large"
# Save everything in your project folder, NOT your home folder
CACHE_DIR = "./hf_cache" 
OUTPUT_DIR = "./bart_emojilm_final"

print(f"Using cache directory: {CACHE_DIR}")

# 1. Setup Tokenizer & Model
tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)
tokenizer.add_special_tokens({'additional_special_tokens': ['<ZWJ>']})

# 2. Dataset & Clean up
print("Loading Dataset...")
dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
dataset = dataset.filter(lambda x: isinstance(x["text"], str) and isinstance(x["emoji"], str))

# 3. Add Emojis to Vocabulary
print("Scanning for unique emojis...")
all_emoji_text = "".join(dataset['train']['emoji'])
unique_emojis = set(all_emoji_text)
emojis_to_add = [char for char in unique_emojis if emoji.is_emoji(char)]
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

# Use 8 processes to speed up mapping
tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=8)

# 5. Training
split_data = tokenized_datasets['train'].train_test_split(test_size=0.05)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3, 
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
print("Done!")

