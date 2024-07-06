from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Vertex AI
vertexai.init(project="gradhack24jnb-610", location="us-central1")

# Load data from text file
data_file = os.path.join(os.path.dirname(__file__), 'data.txt')
try:
    with open(data_file, 'r', encoding='utf-8') as file:
        textsi_1 = file.read()
except FileNotFoundError:
    textsi_1 = ""

model = GenerativeModel(
    "gemini-1.5-flash-001",
    system_instruction=[textsi_1]
)

generation_config = {
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/data', methods=['GET'])
def get_data():
    try:
        with open(data_file, 'r', encoding='utf-8') as file:
            data = file.read()
        return jsonify({"data": data})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']

    try:
        responses = model.generate_content(
            [question],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        response_text = "".join([response.text for response in responses])
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'response': f"Error generating response: {str(e)}"})


# Endpoint for banking recommendations (you'll integrate your logic here)
@app.route('/recommend/banking', methods=['POST'])
def get_banking_recommendation():
    data = request.json
    income = data.get('income')
    spending = data.get('spending')

    # Replace with actual recommendation logic based on income and spending
    recommendation = f"Recommendation for banking based on income {income} and spending {spending}"

    return jsonify({'recommendation': recommendation})


# Endpoint for car insurance recommendations (you'll integrate your logic here)
@app.route('/recommend/car-insurance', methods=['POST'])
def get_car_insurance_recommendation():
    data = request.json
    car_model = data.get('carModel')
    distance_driven = data.get('distanceDriven')
    monthly_income = data.get('monthlyIncome')
    insurance_budget = data.get('insuranceBudget')

    # Replace with actual recommendation logic based on car details and financials
    recommendation = f"Recommendation for car insurance based on car model {car_model}, distance {distance_driven}, income {monthly_income}, and budget {insurance_budget}"

    return jsonify({'recommendation': recommendation})


# Endpoint for travel recommendations (you'll integrate your logic here)
@app.route('/recommend/travel', methods=['POST'])
def get_travel_recommendation():
    data = request.json
    destination = data.get('destination')
    budget = data.get('budget')

    # Replace with actual recommendation logic based on destination and budget
    recommendation = f"Recommendation for travel based on destination {destination} and budget {budget}"

    return jsonify({'recommendation': recommendation})


if __name__ == '__main__':
    app.run(debug=True)
