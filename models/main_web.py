# streamlit_app.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

st.set_page_config(page_title="Bid Anomaly Detection (XGBoost)", layout="wide")
st.title("🚨 Government Bids Anomaly Detector (XGBoost)")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("modelXGB_09-01-2020.pkl")
    return model

model = load_model()

# File upload
st.sidebar.header("📁 Upload Your Bids Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with bid features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    if st.button("🔍 Detect Anomalies / Predict"):
        try:
            # Predict
            X = df.select_dtypes(include=[np.number])  # Only numeric columns
            preds = model.predict(X)
            try:
                scores = model.predict_proba(X)[:, 1]
            except:
                scores = None

            df["Prediction"] = preds
            if scores is not None:
                df["Anomaly_Score"] = scores

            st.success("✅ Detection Complete")
            st.dataframe(df)

            st.subheader("📈 Prediction Distribution")
            st.bar_chart(df["Prediction"].value_counts())

            if scores is not None:
                st.subheader("📉 Anomaly Score Distribution")
                st.line_chart(df["Anomaly_Score"])

            st.subheader("🚨 Detected Anomalies (Prediction = 1)")
            st.dataframe(df[df["Prediction"] == 1])

        except Exception as e:
            st.error(f"Error during processing: {e}")
else:
    st.info("👈 Upload a valid CSV file to start anomaly detection.")

st.markdown("---")
st.markdown("Developed by [Your Name] • Powered by Streamlit & XGBoost")
