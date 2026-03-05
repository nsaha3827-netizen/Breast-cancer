import joblib
import pandas as pd
import numpy as np
import os

def load_model_and_scaler():
    """Load the trained model and scaler"""
    model_dir = os.path.dirname(__file__)
    model_file = os.path.join(model_dir, "logistic_regression_model.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def predict_diagnosis(features_dict):
    """
    Predict breast cancer diagnosis from user input
    
    Args:
        features_dict (dict): Dictionary with feature names and values
                             Example: {
                                 'radius_mean': 16.5,
                                 'texture_mean': 23.4,
                                 ...
                             }
    
    Returns:
        dict: Prediction result with diagnosis and probability
    """
    model, scaler = load_model_and_scaler()
    
    # Ensure the features match the training set
    required_feats = get_feature_names()
    provided_feats = list(features_dict.keys())
    missing = [f for f in required_feats if f not in provided_feats]
    extra = [f for f in provided_feats if f not in required_feats]
    
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")
    if extra:
        # drop extras silently
        for key in extra:
            features_dict.pop(key, None)
    
    # Convert to DataFrame with required order
    input_df = pd.DataFrame([features_dict], columns=required_feats)
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Map to diagnosis names
    diagnosis_map = {0: "Benign (B)", 1: "Malignant (M)"}
    
    return {
        "diagnosis": diagnosis_map[prediction],
        "probability_benign": probability[0],
        "probability_malignant": probability[1],
        "raw_prediction": prediction
    }

import json

def get_feature_names():
    """Get the list of required features loaded from training.
    Falls back to default if the file is missing."""
    model_dir = os.path.dirname(__file__)
    feature_list_file = os.path.join(model_dir, "feature_names.json")
    try:
        with open(feature_list_file, 'r') as f:
            features = json.load(f)
            return features
    except Exception:
        # fallback to full set if file not available
        return [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
            "smoothness_mean", "compactness_mean", "concavity_mean", 
            "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", 
            "smoothness_se", "compactness_se", "concavity_se", 
            "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", 
            "smoothness_worst", "compactness_worst", "concavity_worst", 
            "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

if __name__ == "__main__":
    # Example usage
    print("\n=== Breast Cancer Prediction System ===\n")
    
    # Example input (you can modify these values)
    example_input = {
        'radius_mean': 13.0,
        'texture_mean': 25.4,
        'perimeter_mean': 84.3,
        'area_mean': 540.4,
        'smoothness_mean': 0.105,
        'compactness_mean': 0.131,
        'concavity_mean': 0.051,
        'concave_points_mean': 0.028,
        'symmetry_mean': 0.181,
        'fractal_dimension_mean': 0.064,
        'radius_se': 0.364,
        'texture_se': 1.411,
        'perimeter_se': 2.589,
        'area_se': 20.62,
        'smoothness_se': 0.004,
        'compactness_se': 0.011,
        'concavity_se': 0.011,
        'concave_points_se': 0.005,
        'symmetry_se': 0.018,
        'fractal_dimension_se': 0.001,
        'radius_worst': 15.61,
        'texture_worst': 30.73,
        'perimeter_worst': 106.2,
        'area_worst': 819.7,
        'smoothness_worst': 0.1644,
        'compactness_worst': 0.2956,
        'concavity_worst': 0.2213,
        'concave_points_worst': 0.0820,
        'symmetry_worst': 0.2830,
        'fractal_dimension_worst': 0.0805
    }
    
    result = predict_diagnosis(example_input)
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Probability Benign: {result['probability_benign']:.4f}")
    print(f"Probability Malignant: {result['probability_malignant']:.4f}")
