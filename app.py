
import streamlit as st
import joblib
import numpy as np

# Load saved files
model = joblib.load("Random Forest_churn_model.pkl")  # Replace with your best model file
scaler = joblib.load("scaler.pkl")
top_features = joblib.load("top_10_features.pkl")
all_features = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="🔮")
st.title("🔮 Telco Customer Churn Predictor")

st.markdown("Provide values for the 10 most important customer features to predict churn:")

# Input fields
user_inputs = []
for feature in top_features:
    if "Yes" in feature or "No" in feature:
        val = st.selectbox(f"{feature}", [0, 1])
    elif "InternetService" in feature:
        val = st.selectbox(f"{feature}", [0, 1])
    else:
        val = st.number_input(f"{feature}", value=0.0)
    user_inputs.append(val)

if st.button("Predict Churn"):
    input_array = np.array(user_inputs).reshape(1, -1)

    # Empty full input
    full_input = np.zeros((1, len(all_features)))
    
    # Fill top 10 values
    for idx, feature in enumerate(top_features):
        pos = all_features.index(feature)
        full_input[0][pos] = input_array[0][idx]

    # Scale
    full_input_scaled = scaler.transform(full_input)
    
    # Predict
    prediction = model.predict(full_input_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ Prediction: Customer is likely to churn.")
    else:
        st.success("✅ Prediction: Customer is not likely to churn.")
