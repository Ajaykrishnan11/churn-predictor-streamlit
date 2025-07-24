import streamlit as st

import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# ✅ Dropbox direct download links (from you)
model_url = "https://www.dropbox.com/scl/fi/x7sgzoyseu3t6b665qkw0/Random-Forest_churn_model.pkl?rlkey=h9e9ohqp67808xxpiecqwbxfb&st=w4421666&dl=1"
scaler_url = "https://www.dropbox.com/scl/fi/your_scaler_file.pkl?rlkey=xxxx&dl=1"  # Replace this
features_url = "https://www.dropbox.com/scl/fi/your_feature_file.pkl?rlkey=xxxx&dl=1"  # Replace this

# ✅ File names
model_file = "RandomForest_churn_model.pkl"
scaler_file = "scaler.pkl"
features_file = "feature_columns.pkl"

# ✅ Download files if not present
def download_file(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

download_file(model_url, model_file)
download_file(scaler_url, scaler_file)
download_file(features_url, features_file)

# ✅ Load files
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
selected_features = joblib.load(features_file)

# ✅ Streamlit UI
st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer details to predict churn risk.")

# Collect user inputs
user_input = {}
for feature in selected_features:
    user_input[feature] = st.text_input(f"{feature}", "0")

if st.button("Predict Churn"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.astype(float)

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to Churn. Probability: {probability:.2f}")
        else:
            st.success(f"✅ Likely to Stay. Probability: {1 - probability:.2f}")
    except Exception as e:
        st.warning(f"Input error: {e}")
