import pandas as pd
import joblib

model = joblib.load("../models/model.pkl")
columns = joblib.load("../models/columns.pkl")

def predict_employee(data_dict):
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return "Leave" if prediction == 1 else "Stay"

if __name__ == "__main__":
    sample = {
        "Age": 35,
        "MonthlyIncome": 5000,
        "DistanceFromHome": 10,
        "YearsAtCompany": 5,
        "JobLevel": 2,
        "OverTime_Yes": 1
    }

    print("Prediction:", predict_employee(sample))