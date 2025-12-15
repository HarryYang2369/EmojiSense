import os, json, subprocess, re
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util as st_util
from sacrebleu.metrics import CHRF

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "mistral"
OUT_PATH = "validated_ollama.jsonl"
MAX_ROWS = 8000            
SIM_THRESHOLD = 0.55

# Sentence similarity model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Disable CHRF for faster execution; enable if needed
chrf = None
# chrf = CHRF(word_order=2)

# Prompt for reverse translation
PROMPT = """You are an expert in interpreting emojis into natural English sentences.
Here are some examples:

🔥🎉🌙 → We're having a bonfire party tonight.
📄🦢⏰ → The origami crane took forever to fold.
💐💒😊 → I caught the bouquet at the wedding.
❄️🏔️🛣️🚫 → The avalanche blocked the mountain road.

Now interpret the following emoji sequence in a similar way.
Write one fluent English sentence.

Emoji: {emoji}
Sentence:"""

#another potential prompt

# PROMPT = """You are assisting in a dataset validation task.
# We are verifying how accurately emojis represent English sentences.
# Your job is to perform a *reverse translation*: given a sequence of emojis,
# produce one natural English sentence that expresses the same meaning.

# Guidelines:
# - Write in English only.
# - Use complete, grammatically correct sentences (no fragments).
# - Do not include emojis or explanations.
# - Do not mention that you are validating or comparing; just give the sentence.

# Here are some examples:

# 🔥🎉🌙 → We're having a bonfire party tonight.
# 📄🦢⏰ → The origami crane took forever to fold.
# 💐💒😊 → I caught the bouquet at the wedding.
# ❄️🏔️🛣️🚫 → The avalanche blocked the mountain road.
# ☕💔😩 → My espresso machine broke this morning.
# 🦎⬆️🪟 → The gecko climbed up the glass.

# Now translate the following emoji sequence according to the same style.

# Emoji: {emoji}
# Sentence:"""

# ----------------------------
# FUNCTIONS
# ----------------------------

def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """Call Ollama locally and return the model's response text."""
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    # Clean up artifacts like "### Response" or markdown formatting
    output = re.sub(r"^#+.*\n", "", output).strip()
    return output

def cosine(a: str, b: str) -> float:
    va = embedder.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    vb = embedder.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(st_util.cos_sim(va, vb).item())

def process_row(r):
    original = r["text"].strip()
    emoji_seq = r["emoji"].strip()

    reverse_text = call_ollama(PROMPT.format(emoji=emoji_seq))
    sim = cosine(original, reverse_text)
    # chrf_score = chrf.sentence_score(reverse_text, [original]).score / 100.0

    return {
        "original_text": original,
        "emoji_sequence": emoji_seq,
        "reverse_translation": reverse_text,
        "similarity_score": sim,
        # "chrf": chrf_score,
        "keep": sim >= SIM_THRESHOLD
    }

# ----------------------------
# MAIN LOOP
# ----------------------------

def main():
    ds = load_dataset("KomeijiForce/Text2Emoji", split="train")
    if MAX_ROWS > 0:
        ds = ds.select(range(min(MAX_ROWS, len(ds))))

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i in tqdm(range(len(ds)), desc="Validating"):
            try:
                r = process_row(ds[i])
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                out.write(json.dumps({"error": str(e), "index": i}) + "\n")
                out.flush()

    print(f"\n Done. Results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
