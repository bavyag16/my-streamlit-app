import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("models/churn_model.pkl")

# ---------------- PAGE ---------------- #
st.title("📊 Customer Churn Prediction System")
st.caption("Machine Learning powered risk prediction dashboard")

# ---------------- INPUT ---------------- #
name = st.text_input("Customer Name")
tenure = st.number_input("Tenure", 0, 120)
monthly = st.number_input("Monthly Charges", 0.0, 1000.0)
total = tenure * monthly

st.metric("Total Charges", round(total, 2))

# ---------------- SIMPLE EXPLANATION LOGIC ---------------- #
if tenure < 6:
    reason = "Low tenure increases churn risk"
elif monthly > 80:
    reason = "High monthly charges increase churn risk"
else:
    reason = "Customer behavior is stable"

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Churn"):

    input_data = [[tenure, monthly, total]]

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.error(f"⚠ {name if name else 'Customer'} WILL CHURN (Risk: {prob:.2f})")
    else:
        st.success(f"✅ {name if name else 'Customer'} will STAY (Risk: {prob:.2f})")

    # Risk level
    if prob > 0.7:
        st.write("🔴 Risk Level: HIGH")
    elif prob > 0.4:
        st.write("🟡 Risk Level: MEDIUM")
    else:
        st.write("🟢 Risk Level: LOW")

    # Explanation
    st.write("💡 Explanation:", reason)

# ---------------- MODEL INSIGHTS ---------------- #
st.subheader("📊 Model Insights")
st.image("models/feature_importance.png")
