import os
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline
from fuzzywuzzy import fuzz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load the model and tokenizer
model_name = "gpt2"  # Using GPT-2 as an example
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize data and models
np.random.seed(0)
banking_data = {
    'income': np.random.randint(30000, 150000, 100),
    'spending': np.random.randint(1000, 5000, 100)
}
insurance_data = {
    'user_id': np.arange(1, 101),
    'income': np.random.randint(30000, 150000, 100),
    # 1: Health, 2: Life, 3: Auto, 4: Home
    'insurance_pref': np.random.choice([1, 2, 3, 4], 100)
}

banking_df = pd.DataFrame(banking_data)
insurance_df = pd.DataFrame(insurance_data)

# Train models
kmeans = KMeans(n_clusters=3)
banking_df['cluster'] = kmeans.fit_predict(banking_df[['income', 'spending']])

insurance_model = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(
    insurance_df[['insurance_pref', 'income']])
distances, indices = insurance_model.kneighbors(
    insurance_df[['insurance_pref', 'income']])


def recommend_banking_services(cluster):
    if cluster == 0:
        return "Recommend savings account"
    elif cluster == 1:
        return "Recommend investment options"
    else:
        return "Recommend premium credit cards"


banking_df['recommendation'] = banking_df['cluster'].apply(
    recommend_banking_services)


def recommend_insurance(user_index):
    similar_users = indices[user_index][1:]
    similar_prefs = insurance_df.iloc[similar_users]['insurance_pref']
    recommendation = similar_prefs.mode()[0]
    return recommendation


insurance_df['insurance_recommendation'] = insurance_df.index.to_series().apply(
    recommend_insurance)


@app.route('/recommend/banking', methods=['POST'])
def recommend_banking():
    data = request.json
    income = data['income']
    spending = data['spending']
    cluster = kmeans.predict([[income, spending]])[0]
    recommendation = recommend_banking_services(cluster)
    return jsonify({'recommendation': recommendation})


@app.route('/recommend/car-insurance', methods=['POST'])
def recommend_car_insurance():
    data = request.json
    car_model = data['carModel']
    distance_driven = data['distanceDriven']
    monthly_income = data['monthlyIncome']
    insurance_budget = data['insuranceBudget']

    # Example recommendation logic based on input data
    recommendation = f"Recommend car insurance for {car_model} based on income {monthly_income} and driving {distance_driven} km/month."

    return jsonify({'recommendation': recommendation})


# Sample dataset
qa_data = [
    {
        "question": "What types of savings accounts do you offer?",
        "answer": "We offer Demand Savings Account, Notice Savings Account, Tax-free Demand Savings Account, and Fixed Deposit Account."
    },
    {
        "question": "Tell me about your credit cards.",
        "answer": "We have Gold Card, Black Card, and Platinum Card."
    },
    {
        "question": "What is a Gold Card?",
        "answer": "The Gold Card offers premium benefits including travel insurance, concierge services, and higher credit limits."
    }
]

# Load your pre-trained GPT-2 model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


def find_best_match(question):
    best_match = None
    best_score = 0

    for qa in qa_data:
        score = fuzz.ratio(question.lower(), qa['question'].lower())
        if score > best_score:
            best_score = score
            best_match = qa

    # Adjusted threshold for better matching
    return best_match if best_score > 85 else None


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Handle common greetings separately
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(greet in question.lower() for greet in greetings):
        return jsonify({"response": "Hello! How can I assist you today?"})

    # First try to find a match in the dataset
    best_match = find_best_match(question)

    if best_match:
        response = best_match['answer']
    else:
        # Use the GPT-2 model to generate a response
        inputs = tokenizer.encode(question, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
