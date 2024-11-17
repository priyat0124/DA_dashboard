from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os


# Load the trained model, scaler, and encoders
model = joblib.load('logistic_regression_model.pkl')  # Corrected to your model file
scaler = joblib.load('scaler.pkl')                    # Loading the scaler
label_encoders = joblib.load('label_encoders.pkl')    # Loading the label encoders

# Initialize the Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the request
    data = request.form
    
    # Extract and preprocess the input data
    try:
        input_data = [
            float(data['Age']),
            float(data['RestingBP']),
            float(data['Cholesterol']),
            float(data['FastingBS']),
            float(data['MaxHR']),
            float(data['Oldpeak']),
            label_encoders['Sex'].transform([data['Sex']])[0],
            label_encoders['ChestPainType'].transform([data['ChestPainType']])[0],
            label_encoders['RestingECG'].transform([data['RestingECG']])[0],
            label_encoders['ExerciseAngina'].transform([data['ExerciseAngina']])[0],
            label_encoders['ST_Slope'].transform([data['ST_Slope']])[0]
        ]

        # Scale the input data
        input_data_scaled = scaler.transform([input_data])

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        prediction_text = 'Positive for heart disease' if prediction[0] == 1 else 'Negative for heart disease'

        return jsonify({'prediction': prediction_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
