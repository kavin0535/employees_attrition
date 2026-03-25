import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify
i

print("STARTING APP...")
app = Flask(__name__)

# Load model
print("Loading model...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
columns_path = os.path.join(BASE_DIR, "models", "columns.pkl")

print("Loading model from:", model_path)

model = joblib.load(model_path)
columns = joblib.load(columns_path)

print("Model loaded")
print("Model loaded")

@app.route("/")
def home():
    return "Employee Attrition Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "Attrition": "Yes" if prediction == 1 else "No",
            "Probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)