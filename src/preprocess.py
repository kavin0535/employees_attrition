import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Convert target
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Drop useless columns
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=drop_cols, errors='ignore')

    # 🔹 HANDLE MISSING VALUES (STRONG FIX)

    # Numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 🔹 One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # 🔥 FINAL SAFETY CHECK (IMPORTANT)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def split_data(df):
    from sklearn.model_selection import train_test_split

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return train_test_split(X, y, test_size=0.2, random_state=42)