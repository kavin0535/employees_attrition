import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_data, preprocess_data, split_data

# ✅ UPDATED HERE
df = load_data("../data/Final dataset Attrition.csv")

df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# DEBUG: Check if any NaN still exists
print("NaN count:\n", X_train.isnull().sum().sum())


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:\n", feature_importance.head(10))

y_prob = model.predict_proba(X_test)[:, 1]

# 🔥 Adjust threshold (IMPORTANT)
y_pred = (y_prob > 0.45).astype(int)
for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    y_temp = (y_prob > t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    print(f"Threshold {t}: Precision={precision_score(y_test, y_temp):.2f}, Recall={recall_score(y_test, y_temp):.2f}")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import os
import joblib

# Create folder
os.makedirs("../model", exist_ok=True)

# Save model
joblib.dump(model, "../model/model.joblib")

# Save columns (VERY IMPORTANT for app.py)
joblib.dump(X_train.columns.tolist(), "../model/columns.joblib")

print("Model saved at ../model/")