from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import re
import pdfplumber
import pandas as pd
import base64
from docx import Document

app = Flask(__name__)

# Define the device, model, and tokenizer
device = "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

max_length = 1024
stride = 256
ai_perplexity_threshold = 55
human_ai_perplexity_threshold = 80


def get_perplexity(sentence):
    input_ids = tokenizer.encode(
        sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    total_nll = 0
    total_tokens = 0

    for start_pos in range(0, input_ids.shape[1], stride):
        end_pos = min(start_pos + max_length, input_ids.shape[1])
        target_len = end_pos - start_pos
        target_ids = input_ids[:, start_pos:end_pos].detach()
        target_ids[:, :-target_len].fill_(-100)
        outputs = model(input_ids[:, start_pos:end_pos], labels=target_ids)
        neg_log_likelihood = outputs.loss * target_len

        total_nll += neg_log_likelihood.sum()
        total_tokens += target_len
    if total_tokens == 0:
        perplexity = float('inf')
    else:
        perplexity = round(float(torch.exp(total_nll / total_tokens)), 2)

    return perplexity


def analyze_text(sentence):
    results = {}

    total_valid_char = sum(len(x)
                        for x in re.findall(r"[a-zA-Z0-9]+", sentence))

    if total_valid_char < 200:
        results["Label"] = -1
        results["Output"] = "Insufficient Content"
        results["Percent_ai"] = "-"
        results["Perplexity"] = "-"
        results["Burstiness"] = "-"

        return results

    lines = re.split(r'(?<=[.?!][ \[\(])|(?<=\n)\s*', sentence)
    lines = [line for line in lines if re.search(
        r"[a-zA-Z0-9]+", line) is not None]
    perplexities = []
    total_characters = 0
    ai_characters = 0
    for line in lines:
        total_characters += len(line)
        perplexity = get_perplexity(line)
        perplexities.append(perplexity)
        if perplexity < ai_perplexity_threshold:
            ai_characters += len(line)

    results["Percent_ai"] = str(
        round((ai_characters/total_characters)*100, 2))+"%"
    results["Perplexity"] = round(sum(perplexities) / len(perplexities), 2)
    results["Burstiness"] = round(np.var(perplexities), 2)

    if results["Perplexity"] <= ai_perplexity_threshold:
        results["Label"] = 0
        results["Output"] = "AI"
    elif results["Perplexity"] <= human_ai_perplexity_threshold:
        results["Label"] = 1
        results["Output"] = "Human + AI"
    else:
        results["Label"] = 2
        results["Output"] = "Human"

    return results


def process_text_file(file):
    if file.mimetype == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text()
                text += extracted_text if extracted_text is not None else ""

    elif file.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text
    else:
        return {"error": "Unsupported file format. Please upload a PDF or Word document."}

    results = analyze_text(text)
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    results = process_text_file(file)
    if "error" in results:
        return render_template('error.html', error_message=results["error"])

    results["file_name"] = file.filename

    return render_template('results.html', results=results)


if __name__ == "__main__":
    app.run(debug=True)
