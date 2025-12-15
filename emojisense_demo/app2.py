import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# --- CONFIG ---
MODEL_PATH = "./bart_age_final/final"

print("Loading Emojisense Model... (This may take a minute)")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- HELPER FUNCTIONS ---
def paper_style_emoji_split(text):
    processed_text = ""
    for char in text:
        if char == '\u200d':
            processed_text += " <ZWJ> "
        else:
            processed_text += char
    return processed_text

def translate(input_text, age_style):
    if age_style == "Gen Z (Slang)":
        tag = "<GEN_Z>"
    elif age_style == "Boomer (Formal)":
        tag = "<BOOMER>"
    else:
        tag = "<STD>"

    clean_emojis = paper_style_emoji_split(input_text)
    final_input = f"{tag} {clean_emojis}"

    inputs = tokenizer(
        final_input, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=5,
            max_length=128, 
            early_stopping=True,
            repetition_penalty=1.2
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- CUSTOM CSS ---
custom_css = """
#title_container {
    text-align: center;
    margin-bottom: 20px;
}
#emoji_header {
    font-size: 3em;
    margin-bottom: 0px;
}
#example_scroll_box {
    max-height: 350px; 
    overflow-y: auto; 
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px;
}
footer {visibility: hidden}
"""

# --- BUILD THE UI ---
# NOTE: No arguments passed to Blocks() in Gradio 6.0
with gr.Blocks() as demo:
    
    with gr.Column(elem_id="title_container"):
        gr.Markdown("<div id='emoji_header'>🔮</div>")
        gr.Markdown("# Emojisense Translator")
        gr.Markdown("### The Age-Aware Emoji Interpreter")

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="Input Emojis or Mixed Text", 
                placeholder="e.g. 🍕📺 or I love 🍕",
                lines=3
            )
            style_dropdown = gr.Dropdown(
                choices=["Standard English", "Gen Z (Slang)", "Boomer (Formal)"],
                value="Standard English",
                label="Target Persona"
            )
            submit_btn = gr.Button("✨ Translate", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Translation", 
                lines=6,
                interactive=False  # Removed show_copy_button to fix crash
            )

    with gr.Accordion("Try these examples (Scroll to see more)", open=True):
        with gr.Column(elem_id="example_scroll_box"):
            gr.Examples(
                examples=[
                    ["🍕📺", "Standard English"],
                    ["This is crazy 💀", "Standard English"],
                    ["I love 🍕", "Standard English"],
                    ["My 🚗 broke down", "Standard English"],
                    ["💃🕺🎉", "Standard English"],
                    ["turn the 🎶 up", "Standard English"]
                ],
                inputs=[input_box, style_dropdown],
                outputs=output_box,
                fn=translate,
                run_on_click=True,
                examples_per_page=50,
                cache_examples=False
            )

    submit_btn.click(
        fn=translate, 
        inputs=[input_box, style_dropdown], 
        outputs=output_box
    )

if __name__ == "__main__":
    # ARGUMENTS MOVED HERE for Gradio 6.0
    demo.launch(
        theme=gr.themes.Soft(primary_hue="purple"), 
        css=custom_css
    )