import streamlit as st
import pandas as pd
from predict import predict_diagnosis, get_feature_names

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🏥 Breast Cancer Diagnosis Prediction")
st.write("Enter patient measurements to predict diagnosis (Benign or Malignant)")

# Get feature names (from training). may vary depending on preprocessing
features = get_feature_names()

# categorize features by suffix to group inputs
mean_features = [f for f in features if "mean" in f]
se_features = [f for f in features if "se" in f and "mean" not in f]
worst_features = [f for f in features if "worst" in f]

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📊 CSV Upload", "📋 Sample Data"])

with tab1:
    st.subheader("Enter Patient Measurements")
    
    # Organize features in columns
    col1, col2, col3 = st.columns(3)
    user_input = {}
    
    # mean_features, se_features, worst_features already computed above
    
    # create input sections only if non-empty
    
    if mean_features:
        st.write("**Mean Features:**")
        cols = st.columns(min(4, len(mean_features)))
        for i, feature in enumerate(mean_features):
            with cols[i % len(cols)]:
                user_input[feature] = st.number_input(
                    label=feature.replace("_", " ").title(),
                    value=0.0,
                    step=0.1,
                    key=f"mean_{i}"
                )

    if se_features:
        st.write("**Standard Error Features:**")
        cols = st.columns(min(4, len(se_features)))
        for i, feature in enumerate(se_features):
            with cols[i % len(cols)]:
                user_input[feature] = st.number_input(
                    label=feature.replace("_", " ").title(),
                    value=0.0,
                    step=0.01,
                    key=f"se_{i}"
                )

    if worst_features:
        st.write("**Worst Features:**")
        cols = st.columns(min(4, len(worst_features)))
        for i, feature in enumerate(worst_features):
            with cols[i % len(cols)]:
                user_input[feature] = st.number_input(
                    label=feature.replace("_", " ").title(),
                    value=0.0,
                    step=0.1,
                    key=f"worst_{i}"
                )
    
    if st.button("🔮 Predict Diagnosis", key="predict_manual"):
        result = predict_diagnosis(user_input)
        
        st.success("### Prediction Result")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Diagnosis",
                result['diagnosis'],
                delta=None
            )
        
        with col2:
            st.metric(
                "Benign Probability",
                f"{result['probability_benign']:.2%}"
            )
        
        with col3:
            st.metric(
                "Malignant Probability",
                f"{result['probability_malignant']:.2%}"
            )
        
        # Confidence indicator
        confidence = max(result['probability_benign'], result['probability_malignant'])
        st.info(f"**Confidence Level:** {confidence:.2%}")

with tab2:
    st.subheader("Upload CSV File")
    st.write("Upload a CSV file with patient measurements (one sample per row)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("🔮 Predict All Samples", key="predict_csv"):
            predictions = []
            
            for idx, row in df.iterrows():
                input_dict = row.to_dict()
                result = predict_diagnosis(input_dict)
                predictions.append({
                    'Sample': idx + 1,
                    'Diagnosis': result['diagnosis'],
                    'Prob_Benign': f"{result['probability_benign']:.4f}",
                    'Prob_Malignant': f"{result['probability_malignant']:.4f}"
                })
            
            results_df = pd.DataFrame(predictions)
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

with tab3:
    st.subheader("Sample Data")
    st.write("Click below to load a zero‑valued sample or use the first row from the training data")
    
    # build default sample with zeros for each feature
    sample_data = {feat: 0.0 for feat in features}
    
    if st.button("Use Zero Sample", key="use_sample"):
        result = predict_diagnosis(sample_data)
        
        st.success("### Prediction Result")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Diagnosis", result['diagnosis'])
        with col2:
            st.metric("Benign Probability", f"{result['probability_benign']:.2%}")
        with col3:
            st.metric("Malignant Probability", f"{result['probability_malignant']:.2%}")
    
    # try loading first row from input_test_data.csv if available
    try:
        demo_df = pd.read_csv("input_test_data.csv")
        if not demo_df.empty:
            if st.button("Use First Test Row", key="use_test"):
                row = demo_df.iloc[0].to_dict()
                # drop Actual_Diagnosis if present
                row.pop("Actual_Diagnosis", None)
                result = predict_diagnosis(row)
                st.success("### Prediction Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Diagnosis", result['diagnosis'])
                with col2:
                    st.metric("Benign Probability", f"{result['probability_benign']:.2%}")
                with col3:
                    st.metric("Malignant Probability", f"{result['probability_malignant']:.2%}")
    except FileNotFoundError:
        pass

# Information section
st.divider()
st.sidebar.info(
    f"""
    ### About This App
    - **Model**: Logistic Regression
    - **Features**: {len(features)} trained measurements (loaded dynamically)
    - **Accuracy**: Check your model's test accuracy
    
    ### Feature Categories
    - **Mean**: Average measurements
    - **SE**: Standard Error
    - **Worst**: Worst measurements
    """
)
