import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify

print("STARTING APP...")
app = Flask(__name__)

# Load model
print("Loading model...")
model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")
print("Model loaded")

@app.route("/")
def home():
    return "Employee Attrition Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # ⚠️ IMPORTANT: must match training preprocessing
    df = pd.get_dummies(df)

    # Align with model features
    columns = joblib.load("../model/columns.joblib")
    model_features = columns
    df = df.reindex(columns=model_features, fill_value=0)

    # Prediction
    prob = model.predict_proba(df)[:, 1][0]
    pred = 1 if prob > 0.45 else 0

    return jsonify({
        "Attrition": "Yes" if pred == 1 else "No",
        "Probability": float(prob)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)