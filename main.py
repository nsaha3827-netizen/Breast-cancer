# Import libraries
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "breast+cancer+wisconsin+diagnostic")
DATA_FILE = os.path.join(DATA_DIR, "wdbc.data")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Data file not found at {DATA_FILE}. "
        "Download the UCI dataset and unzip it into the workspace."
    )

# ── Load dataset ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE, header=None)

columns = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
df.columns = columns

# ── Preprocessing ──────────────────────────────────────────────────────────────
# Encode target
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# Drop ID
df.drop(columns=["ID"], inplace=True)

# Drop highly correlated features (>0.90)
corr_matrix = df.drop("Diagnosis", axis=1).corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
df.drop(columns=to_drop, inplace=True)

# Keep only features correlated with target (>0.1)
corr_with_target = df.corr()["Diagnosis"].abs()
important_features = corr_with_target[corr_with_target > 0.1].index
df = df[important_features]

# ── Features / target ──────────────────────────────────────────────────────────
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# ── Handle class imbalance ─────────────────────────────────────────────────────
rs = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rs.fit_resample(X, y)

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train model ────────────────────────────────────────────────────────────────
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = lr.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print("\nCross Validation Scores:", scores)
print("Average CV Score:", scores.mean())

# ── Save artefacts ─────────────────────────────────────────────────────────────
model_file        = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
scaler_file       = os.path.join(BASE_DIR, "scaler.pkl")
feature_list_file = os.path.join(BASE_DIR, "feature_names.json")
input_csv_file    = os.path.join(BASE_DIR, "input_test_data.csv")
output_csv_file   = os.path.join(BASE_DIR, "predictions_vs_actual.csv")

joblib.dump(lr, model_file)
print(f"\nModel saved to:   {model_file}")

joblib.dump(scaler, scaler_file)
print(f"Scaler saved to:  {scaler_file}")

with open(feature_list_file, "w") as f:
    json.dump(X.columns.tolist(), f)
print(f"Feature names saved to: {feature_list_file}")

# Input CSV  – scaled test features + actual labels
input_csv = pd.DataFrame(X_test_scaled, columns=X.columns)
input_csv["Actual_Diagnosis"] = y_test.reset_index(drop=True)
input_csv.to_csv(input_csv_file, index=False)
print(f"Input CSV saved to:  {input_csv_file}")

# Output CSV – predictions vs actuals
output_csv = pd.DataFrame({
    "Actual_Diagnosis":    y_test.reset_index(drop=True),
    "Predicted_Diagnosis": y_pred,
    "Prediction_Correct":  (y_test.reset_index(drop=True) == y_pred),
})
output_csv.to_csv(output_csv_file, index=False)
print(f"Output CSV saved to: {output_csv_file}")

print("\n--- Comparison Summary (first 10 rows) ---")
print(output_csv.head(10))