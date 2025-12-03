import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# --- CONFIG ---
MODEL_PATH = "./bart_age_final/final" 

print("Loading Emojisense Model... (This may take a minute)")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- HELPER FUNCTIONS ---
def paper_style_emoji_split(text):
    """Splits emojis by ZWJ to match training data format."""
    processed_text = ""
    for char in text:
        if char == '\u200d':
            processed_text += " <ZWJ> "
        else:
            processed_text += char
    return processed_text

def translate(input_text, age_style):
    # 1. Map User Choice to Special Token
    if age_style == "Gen Z (Slang)":
        tag = "<GEN_Z>"
    elif age_style == "Boomer (Formal)":
        tag = "<BOOMER>"
    else:
        tag = "<STD>"

    # 2. Pre-process Input
    # We split ZWJs first, then prepend the tag
    clean_emojis = paper_style_emoji_split(input_text)
    final_input = f"{tag} {clean_emojis}"

    # 3. Tokenize
    inputs = tokenizer(
        final_input, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True
    )

    # 4. Generate
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=5,           # Higher beams = smarter search
            max_length=128, 
            early_stopping=True,
            repetition_penalty=1.2 # Prevents repeating words
        )

    # 5. Decode
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output_text

# --- BUILD THE UI ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🔮 Emojisense Translator
        ### The Age-Aware Emoji-to-Text Interpreter
        Enter emojis or mixed text below, choose a persona, and see how the meaning shifts!
        """
    )
    
    with gr.Row():
        with gr.Column():
            # Left Column: Inputs
            input_box = gr.Textbox(
                label="Input Emojis / Text", 
                placeholder="e.g. 🍕📺 or I love 🍕",
                lines=2
            )
            style_dropdown = gr.Dropdown(
                choices=["Standard English", "Gen Z (Slang)", "Boomer (Formal)"],
                value="Standard English",
                label="Target Audience (Age Parameter)"
            )
            submit_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            # Right Column: Output
            output_box = gr.Textbox(label="Translation", lines=2)

    # Examples to help the user start quickly
    gr.Examples(
        examples=[
            ["This is crazy 💀", "Standard English"],
            ["This is crazy 💀", "Boomer (Formal)"],
            ["This is crazy 💀", "Gen Z (Slang)"],
            ["I love 🍕", "Standard English"],
            ["I love 🍕", "Boomer (Formal)"],
            ["I love 🍕", "Gen Z (Slang)"],
            ["My 🚗 broke down", "Standard English"],
            ["My 🚗 broke down", "Boomer (Formal)"],
            ["My 🚗 broke down", "Gen Z (Slang)"],
            ["💃🕺🎉", "Standard English"],
            ["💃🕺🎉", "Boomer (Formal)"],
            ["💃🕺🎉", "Gen Z (Slang)"]
        ],
        inputs=[input_box, style_dropdown]
    )

    # Connect the button to the function
    submit_btn.click(
        fn=translate, 
        inputs=[input_box, style_dropdown], 
        outputs=output_box
    )

# Launch the website
demo.launch()