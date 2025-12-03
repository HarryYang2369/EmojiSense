import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Model Paths
MODEL_PATHS = {
    "Model 1 (Base) ": "./step1_basemodel/bart_emojilm_final/final",
    "Model 2 (Mixed)": "./step2_mixed/bart_mixed_final/final",
    "Model 3 (Age)  ": "./step3_age/bart_age_final/final"
}

# 2. Data Paths (Based on your screenshots/structure)
GEN_Z_PATH = "./step3_age/data/gen_z_20k.csv"
BOOMER_PATH = "./step3_age/data/boomer_20k.csv"
STD_PATH = "./step2_mixed/data/text2emoji_100k.csv"

# 3. Evaluation Size
# We will take 200 from each style to make a balanced 600-row test set
SAMPLES_PER_STYLE = 200 

# Load Metric
bleu = evaluate.load("bleu")

def load_balanced_test_set():
    print("--- Building Balanced Test Set ---")
    
    # A. Load Gen Z
    print(f"Loading Gen Z from {GEN_Z_PATH}...")
    df_genz = pd.read_csv(GEN_Z_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    # Ensure topic is set correctly
    df_genz['topic'] = 'gen_z'
    
    # B. Load Boomer
    print(f"Loading Boomer from {BOOMER_PATH}...")
    df_boomer = pd.read_csv(BOOMER_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    df_boomer['topic'] = 'boomer'
    
    # C. Load Standard (from Step 2 data)
    print(f"Loading Standard from {STD_PATH}...")
    df_std = pd.read_csv(STD_PATH).sample(n=SAMPLES_PER_STYLE, random_state=42)
    df_std['topic'] = 'standard'
    
    # Combine
    test_df = pd.concat([df_genz, df_boomer, df_std]).sample(frac=1, random_state=123).reset_index(drop=True)
    print(f"Test Set Created: {len(test_df)} examples (Balanced mix).")
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
    print(f"\n--- Evaluating {model_name} ---")
    
    try:
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"Skipping {model_name}: Could not load ({e})")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    predictions = []
    references = []
    
    print(f"Generating predictions...")
    
    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        input_text = row['emoji']
        target_text = row['text']
        topic = row['topic'] 
        
        # --- INPUT FORMATTING ---
        # 1. Basic ZWJ Split (All models need this)
        clean_input = paper_style_emoji_split(input_text)
        
        # 2. Tag Logic
        if "Model 3" in model_name or "Age" in model_name:
            # Model 3 EXPECTS tags. We must remove existing ones first to avoid double tagging
            # (e.g., if the csv already has <GEN_Z>, we strip it and add it fresh to be safe)
            clean_input = clean_input.replace("<GEN_Z>", "").replace("<BOOMER>", "").replace("<STD>", "").strip()
            
            if topic == 'gen_z':
                final_input = f"<GEN_Z> {clean_input}"
            elif topic == 'boomer':
                final_input = f"<BOOMER> {clean_input}"
            else:
                final_input = f"<STD> {clean_input}"
        else:
            # Models 1 & 2 DO NOT understand tags. Strip them out.
            final_input = clean_input.replace("<GEN_Z>", "").replace("<BOOMER>", "").replace("<STD>", "").strip()

        # Generate
        inputs = tokenizer(final_input, return_tensors="pt", max_length=128, truncation=True).to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=128, 
                num_beams=4,
                early_stopping=True
            )
            
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append([target_text]) 
        
    # Calculate BLEU
    results = bleu.compute(predictions=predictions, references=references)
    print(f"🔵 BLEU Score: {results['bleu'] * 100:.2f}")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Build the Data
    final_test_set = load_balanced_test_set()
    
    # 2. Run Eval
    for name, path in MODEL_PATHS.items():
        evaluate_model(name, path, final_test_set)
