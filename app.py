
import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("app/model.pkl")
columns_used = np.load("app/columns.npy", allow_pickle=True)

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📞 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn based on their service details.")

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
Contract_Two_year = st.selectbox("Is the contract for two years?", [0, 1])
InternetService_Fiber_optic = st.selectbox("Uses Fiber Optic Internet?", [0, 1])

input_df = pd.DataFrame({
    'SeniorCitizen': [SeniorCitizen],
    'Partner_Yes': [1 if Partner == "Yes" else 0],
    'Dependents_Yes': [1 if Dependents == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'Contract_Two year': [Contract_Two_year],
    'InternetService_Fiber optic': [InternetService_Fiber_optic],
})

input_df_aligned = input_df.reindex(columns=columns_used, fill_value=0)

if st.button("Predict Churn"):
    prediction = model.predict(input_df_aligned)[0]
    if prediction == 1:
        st.error("⚠️ Prediction: The customer is likely to **CHURN**.")
    else:
        st.success("✅ Prediction: The customer is likely to **STAY**.")

st.download_button(
    label="📥 Download This Input as CSV",
    data=input_df.to_csv(index=False),
    file_name="customer_input.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("💖 Built by Pratu Baby's Assistant with Streamlit and ML")
