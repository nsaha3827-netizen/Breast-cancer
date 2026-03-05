# Breast Cancer Prediction - User Guide

## Overview
This project provides **3 ways** for users to get predictions from your trained ML model.

---

## **Method 1: Simple Python Function** ✅ (Easiest to Integrate)

### How to Use:
```python
from predict import predict_diagnosis

# Input data as dictionary
user_data = {
    'radius_mean': 16.5,
    'texture_mean': 23.4,
    'perimeter_mean': 107.5,
    'area_mean': 860.0,
    # ... add all 30 features
}

# Get prediction
result = predict_diagnosis(user_data)
print(result)
# Output:
# {
#     'diagnosis': 'Malignant (M)',
#     'probability_benign': 0.1234,
#     'probability_malignant': 0.8766,
#     'raw_prediction': 1
# }
```

### Run directly:
```bash
python predict.py
```

---

## **Method 2: Web Interface** 🌐 (Best for Users)

### Setup:
1. Install Streamlit:
```bash
pip install streamlit
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open browser at: `http://localhost:8501`

### Features:
- ✏️ **Manual Input**: Type measurements directly
- 📊 **CSV Upload**: Batch predictions from multiple samples
- 📋 **Sample Data**: Test with built-in sample

---

## **Method 3: REST API** 🔌 (For External Applications)

If users want to call your model from different applications (web apps, mobile, etc.):

```bash
pip install flask
python flask_api.py
```

Then users can call:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"radius_mean": 16.5, "texture_mean": 23.4, ...}'
```

---

## **Input Data Requirements**

The set of features used for prediction is determined during preprocessing when the model is trained. Highly correlated features and those with low correlation to the target are dropped, so the actual list may be shorter than the original 30.

After running `main.py`, a file named `feature_names.json` is created; the prediction utilities (and the web/API interfaces) automatically load this list. You can inspect it directly or call the `/features` API endpoint.

For reference, the original dataset contains 30 measurements split into three categories:

### Mean Features (10, may be reduced)
- radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
- compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean

### Standard Error Features (10, may be reduced)
- radius_se, texture_se, perimeter_se, area_se, smoothness_se
- compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se

### Worst Features (10, may be reduced)
- radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst
- compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst

> **Tip:** After training, run `python -c "import json; print(json.load(open('feature_names.json')))"` to see exactly what features are required.  

---

## **How the Prediction Works**

1. **User inputs 30 measurements**
2. **Features are scaled** (using your saved scaler)
3. **Model predicts**: Benign (0) or Malignant (1)
4. **Results show**: 
   - Diagnosis category
   - Probability scores
   - Confidence level

---

## **Quick Example CSV for Batch Prediction**

Create `patients.csv`:
```
radius_mean,texture_mean,perimeter_mean,...,fractal_dimension_worst
13.0,25.4,84.3,...,0.0805
14.5,26.1,93.2,...,0.0890
```

Upload to the Streamlit app or use Python API!

---

## **Recommendation**

- **For simple use**: Use **Method 1** (Python function)
- **For your users**: Use **Method 2** (Streamlit app) - No coding needed!
- **For integration**: Use **Method 3** (Flask API)

---

## **Files Created**

- `predict.py` - Core prediction function
- `app.py` - Streamlit web interface
- `flask_api.py` - REST API (optional)

All use your existing saved model and scaler! ✅
