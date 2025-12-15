import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
import pandas as pd
from tqdm import tqdm
import os

# --- CONFIGURATION ---
MODEL_PATHS = {
    "Model 1 (Base) ": "./step1_basemodel/bart_emojilm_final/final",
    "Model 2 (Mixed)": "./step2_mixed/bart_mixed_final/final",
    "Model 3 (Age)  ": "./step3_age/bart_age_final/final"
}

# Paths to your data
GEN_Z_PATH = "./step3_age/data/gen_z_20k.csv"
BOOMER_PATH = "./step3_age/data/boomer_20k.csv"
STD_PATH = "./step2_mixed/data/text2emoji_100k.csv"

# Sample size per category (200 * 3 = 600 total test items)
SAMPLES_PER_STYLE = 500

# Load Metrics
print("Loading Metrics...")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

def load_balanced_test_set():
    print("--- Building Balanced Test Set ---")
    df_genz = pd.read_csv(GEN_Z_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    df_genz['topic'] = 'gen_z'
    
    df_boomer = pd.read_csv(BOOMER_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    df_boomer['topic'] = 'boomer'
    
    # df_std = pd.read_csv(STD_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    # df_std['topic'] = 'standard'
    
    test_df = pd.concat([df_genz, df_boomer]).sample(frac=1, random_state=123).reset_index(drop=True)
    print(f"Test Set Created: {len(test_df)} examples.")
    return test_df

def paper_style_emoji_split(text):
    """Splits emojis by ZWJ"""
    processed_text = ""
    for char in str(text):
        if char == '\u200d':
            processed_text += " <ZWJ> "
        else:
            processed_text += char
    return processed_text

def evaluate_model(model_name, model_path, test_data):
    print(f"\n==========================================")
    print(f"EVALUATING: {model_name}")
    print(f"==========================================")
    
    try:
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"Skipping {model_name}: {e}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    predictions = []
    references = []
    
    print("Generating predictions...")
    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        input_text = row['emoji']
        target_text = row['text']
        topic = row['topic']
        
        # --- INPUT PREP ---
        clean_input = paper_style_emoji_split(input_text)
        # Strip existing tags to be safe
        clean_input = clean_input.replace("<GEN_Z>", "").replace("<BOOMER>", "").replace("<STD>", "").strip()
        
        if "Model 3" in model_name or "Age" in model_name:
            if topic == 'gen_z':
                final_input = f"<GEN_Z> {clean_input}"
            elif topic == 'boomer':
                final_input = f"<BOOMER> {clean_input}"
            else:
                final_input = f"<STD> {clean_input}"
        else:
            final_input = clean_input # No tags for Models 1 & 2

        # Inference
        inputs = tokenizer(final_input, return_tensors="pt", max_length=128, truncation=True).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True
            )
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        predictions.append(pred)
        references.append(target_text)

    # --- CALCULATE METRICS ---
    print("\nComputing Scores...")
    
    # BLEU (Expects list of lists for references)
    bleu_refs = [[r] for r in references]
    res_bleu = bleu.compute(predictions=predictions, references=bleu_refs)
    
    # METEOR
    res_meteor = meteor.compute(predictions=predictions, references=references)
    
    # BERTScore (This takes a moment, downloads a mini-model)
    res_bert = bertscore.compute(predictions=predictions, references=references, lang="en")
    
    # Print Results
    print(f"\n FINAL RESULTS FOR {model_name}:")
    print(f"   BLEU Score:      {res_bleu['bleu'] * 100:.2f}")
    print(f"   METEOR Score:    {res_meteor['meteor'] * 100:.2f}")
    print(f"   BERTScore (F1):  {sum(res_bert['f1']) / len(res_bert['f1']) * 100:.2f}")

# --- EXECUTION ---
if __name__ == "__main__":
    final_test_set = load_balanced_test_set()
    for name, path in MODEL_PATHS.items():
        evaluate_model(name, path, final_test_set)