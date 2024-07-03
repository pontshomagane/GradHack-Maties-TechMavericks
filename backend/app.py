from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

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


if __name__ == '__main__':
    app.run(debug=True)
