
from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load ANN model
model = load_model("heart_disease_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Home page
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

# Prediction page
@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# Predict heart disease
@app.route('/predict', methods=['POST'])
def predict():

    # Get patient details
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    # Arrange input data
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # ANN prediction
    prediction_output = model.predict(input_scaled, verbose=0)
    probability = float(prediction_output[0][0])

    # Display result
    if probability >= 0.5:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    return render_template(
        'result.html',
        result=result,
        probability=round(probability * 100, 2)
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
