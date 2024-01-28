

import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertForSequenceClassification
import openai

# TOPSIS implementation
class TOPSIS:
    def __init__(self, criteria, weights, data):
        self.criteria = criteria
        self.weights = weights
        self.data = data
        self.ideal_solution = None
        self.negative_ideal_solution = None
        self.relative_closeness = None

    def normalize_data(self):
        # Normalize data to a common scale (0-1)
        for i, criterion in enumerate(self.criteria):
            if criterion == "min":
                self.data[:, i] = (self.data[:, i] - 
       np.max(self.data[:, i])) / (np.max(self.data[:, i]) - np.min(self.data[:, i]))
            else:
                self.data[:, i] = (self.data[:, i] - np.min(self.data[:, i])) / (np.max(self.data[:, i]) - np.min(self.data[:, i]))

    def calculate_ideal_solution(self):
        # Find ideal and negative-ideal solutions for each criterion
        self.ideal_solution = np.max(self.data, axis=0)
        self.negative_ideal_solution = np.min(self.data, axis=0)

    def calculate_distances(self):
        # Calculate distances to ideal and negative-ideal solutions
        ideal_distances = np.linalg.norm(self.data - self.ideal_solution, axis=1)
        negative_ideal_distances = np.linalg.norm(self.data - self.negative_ideal_solution, axis=1)
        return ideal_distances, negative_ideal_distances

    def calculate_relative_closeness(self, ideal_distances, negative_ideal_distances):
        # Calculate relative closeness scores
        self.relative_closeness = negative_ideal_distances / (ideal_distances + negative_ideal_distances)

    def rank_models(self):
        # Rank models based on relative closeness scores
        ranking = np.argsort(self.relative_closeness)[::-1]
        return ranking

# Define evaluation functions (replace with your actual implementations)
def evaluate_fluency(generated_text):
    fluency_score = 0.8  
    return fluency_score

def evaluate_coherence(generated_text):
    coherence_score = 0.7  
    return coherence_score

def evaluate_originality(generated_text):
    originality_score = 0.9  
    return originality_score

def evaluate_accuracy(generated_text, reference_text):
    accuracy_score = 0.85  
    return accuracy_score

# Model definitions
def generate_text_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  
        prompt=prompt,
        max_tokens=150  
    )
    return response.choices[0].text.strip()

def generate_text_with_t5(prompt):
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()

def generate_text_with_bert(prompt):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return tokenizer.decode(predicted_class, skip_special_tokens=True)

def main():
    # Define criteria, weights, prompts, and references (adjust as needed)
    criteria = ["fluency", "coherence", "originality", "accuracy"]
    weights = [0.3, 0.2, 0.4, 0.1]
    prompts = ["Once upon a time", "Translate the following English text to French: 'Hello, how are you?'", "Some other prompt"]
    references = ["...", "...", "..."]  # reference texts for accuracy evaluation

    # Evaluate models on each criterion
    data = []
    for model_name in ["gpt3", "t5", "bert"]:
        model_scores = []
        for i, prompt in enumerate(prompts):
            generated_text = None
            if model_name == "gpt3":
                generated_text = generate_text_with_gpt3(prompt)
            elif model_name == "t5":
                generated_text = generate_text_with_t5(prompt)
