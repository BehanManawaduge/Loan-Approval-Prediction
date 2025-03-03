import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("model/loan_approval_model.h5")
scaler = StandardScaler()


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Render a simple HTML form for user input


# API route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.get_json(force=True)

        # Extract features from the incoming request
        features = np.array([data["features"]])  # Expecting a list of features

        # Preprocessing the features
        features_scaled = scaler.fit_transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_binary = (prediction > 0.5).astype(int)

        # Return the prediction
        response = {"loan_status": prediction_binary.flatten().tolist()}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


# Route to get prediction via an HTML form (if front-end is added)
@app.route('/predict_form', methods=['POST'])
def predict_form():
    # Collect the features from form input (these should correspond to the 26 features expected by the model)
    features = [float(x) for x in request.form.values()]

    # Ensure that the input features are the same as during training (e.g., using scaling, encoding, etc.)
    # Preprocess the features (this should match your training preprocessing, e.g., scaling)
    features_scaled = scaler.transform([features])  # Use scaler from the training phase to scale

    # Predict
    prediction = model.predict(features_scaled)
    prediction_binary = (prediction > 0.5).astype(int)

    # Return prediction result
    return render_template('index.html', prediction_text=f"Loan Status: {'Approved' if prediction_binary[0][0] == 1 else 'Denied'}")



# Run the app
if __name__ == "__main__":
    app.run(debug=True)
