This is the main Python file that will handle the backend logic, including loading the model, processing user inputs, and returning predictions.

python
Copy
from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(_name_)

# Load the model, scaler, and label encoder
model = load_model('model/loan_approval_model.h5')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')

# Define a route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = [
            float(request.form['age']),
            float(request.form['income']),
            float(request.form['loan_amount']),
            float(request.form['credit_score']),
            request.form['employment_status'],
            request.form['housing_type']
        ]

        # Preprocess the input data
        input_data_encoded = np.array(input_data).reshape(1, -1)

        # One-hot encode the categorical data (employment_status and housing_type)
        input_data_encoded = np.array(input_data_encoded, dtype=str)
        employment_status = input_data_encoded[0, 4]
        housing_type = input_data_encoded[0, 5]

        input_data_encoded = np.delete(input_data_encoded, [4, 5], axis=1)
        employment_status_encoded = [1 if x == employment_status else 0 for x in ["Employed", "Self-Employed", "Unemployed"]]
        housing_type_encoded = [1 if x == housing_type else 0 for x in ["Rent", "Own"]]
        
        input_data_encoded = np.concatenate((input_data_encoded, employment_status_encoded, housing_type_encoded), axis=1)

        # Scale the input data using the same scaler used during training
        input_data_scaled = scaler.transform(input_data_encoded)

        # Make the prediction
        prediction = model.predict(input_data_scaled)

        # Convert the prediction to loan status
        loan_status = le.inverse_transform([int(prediction > 0.5)])[0]

        # Return the result to the user
        return render_template('index.html', loan_status=loan_status)

    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(debug=True)
