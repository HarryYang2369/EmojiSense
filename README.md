# EmojiSense

This repository contains the source code, training scripts, and demo for **EmojiSense**, a project that develops an age-aware emoji-to-text translation model based on the BART-Large sequence-to-sequence transformer.

## Project Structure

**emojisense_src**
Contains the core codebase. This includes:
* Training scripts for all three models (Emoji-to-Text, Prediction, and Style Transfer).
* Cross-LLM validation scripts used to construct our high quality dataset.

**emojisense_demo**
A local web demo for Model 3 (the Age-Aware Style Transfer model). You can run this to interactively test the Gen Z and Boomer personas.

**EmojiSense_presentation.pdf**
A slideshow covering the project motivation, methodology, and summary of results.

## Models & Weights

Due to GitHub's file size limits, our actual age-aware model is hosted on Hugging Face.

**Link to Model:** [JackInEdinburgh/emoji-sense](https://huggingface.co/JackInEdinburgh/emoji-sense)

### How to Load
The model is based on BART. You can load it directly in Python using the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer from Hugging Face
model_name = "JackInEdinburgh/emoji-sense"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example usage
input_text = "I love pizza 🍕"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
