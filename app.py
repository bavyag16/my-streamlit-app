import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Churn Intelligence System", layout="wide")

# ---------------- CLEAN UI ---------------- #
st.markdown("""
<style>
.stApp { background-color: #F1F5F9 !important; }

h1, h2, h3 {
    color: #1E3A8A !important;
}

.stButton > button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1D4ED8 !important;
}

section[data-testid="stSidebar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("models/churn_model.pkl")

# ---------------- TITLE ---------------- #
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Intelligence System</h1>", unsafe_allow_html=True)

# ---------------- NAVIGATION ---------------- #
page = st.radio("", ["Home", "Single Prediction", "Bulk Prediction"], horizontal=True)

# ---------------- HOME ---------------- #
if page == "Home":
    st.markdown("""
    ### Predict. Understand. Retain.

    Multi-domain churn prediction system for customer retention.

    👉 Single Prediction – Check one customer  
    👉 Bulk Prediction – Upload CSV  
    """)

# ---------------- SINGLE ---------------- #
elif page == "Single Prediction":

    st.subheader("🔍 Customer Prediction")

    # ✅ Domain selector
    domain = st.selectbox("Select Industry", ["Telecom", "Banking", "E-commerce"])

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Customer Name")
        gender = st.selectbox("Gender", ["Male", "Female"])

        age = st.number_input("Age", 0, 100)
        senior = 1 if age >= 60 else 0

    # Dynamic labels
    if domain == "Telecom":
        tenure_label = "Tenure (months)"
        monthly_label = "Monthly Charges"
        total_label = "Total Charges"

    elif domain == "Banking":
        tenure_label = "Account Duration (months)"
        monthly_label = "Monthly Transactions"
        total_label = "Total Balance"

    else:
        tenure_label = "Customer Duration (months)"
        monthly_label = "Monthly Spend"
        total_label = "Total Spend"

    with col2:
        tenure = st.number_input(tenure_label, 0, 120)
        monthly = st.number_input(monthly_label, 0.0, 1000.0)

    with col3:
        total = st.number_input(total_label, 0.0, 50000.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    # ---------------- PREPARE INPUT ---------------- #
    input_data = pd.DataFrame({
        "gender": [gender],
        "seniorcitizen": [senior],
        "tenure": [tenure],
        "monthlycharges": [monthly],
        "totalcharges": [total],
        "contract": [contract],
        "paymentmethod": [payment]
    })

    input_data.columns = input_data.columns.str.lower()
    input_data = pd.get_dummies(input_data)

    # ✅ SAFE MODEL COLUMNS
    try:
        model_columns = model.feature_names_in_
    except:
        model_columns = model.get_booster().feature_names

    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    st.markdown("---")

    if st.button("🚀 Predict Churn"):

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("### 📌 Prediction Result")

        if pred == 1:
            st.error(f"⚠ {name if name else 'Customer'} WILL CHURN (Risk: {prob:.2f})")
        else:
            st.success(f"✅ {name if name else 'Customer'} will STAY (Risk: {prob:.2f})")

        # ---------------- RISK LEVEL ---------------- #
        if prob > 0.7:
            st.write("🔴 Risk Level: HIGH")
        elif prob > 0.4:
            st.write("🟡 Risk Level: MEDIUM")
        else:
            st.write("🟢 Risk Level: LOW")

        # ---------------- EXPLANATION ---------------- #
        st.markdown("### 💡 Explanation")

        if pred == 1:
            if domain == "Telecom":
                st.warning("Customer may churn due to high charges or short usage.")
            elif domain == "Banking":
                st.warning("Customer may leave due to low engagement or high transactions.")
            else:
                st.warning("Customer may stop purchasing due to inconsistent activity.")
        else:
            st.success("Customer is stable based on current behavior.")

        # ---------------- SHAP ---------------- #
        st.markdown("---")
        st.subheader("🔍 Feature Impact")

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)

            shap_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Impact": shap_values.values[0]
            })

            shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

            shap_df["Feature"] = shap_df["Feature"].replace({
                "gender_Male": "Gender",
                "tenure": "Tenure",
                "monthlycharges": "Monthly Spend",
                "totalcharges": "Total Spend"
            })

            colA, colB = st.columns(2)

            with colA:
                st.dataframe(shap_df.head(10))

            with colB:
                fig, ax = plt.subplots()
                ax.barh(shap_df["Feature"][:10], shap_df["Impact"][:10])
                ax.invert_yaxis()
                st.pyplot(fig)

        except:
            st.warning("SHAP explanation not available for this model.")

# ---------------- BULK ---------------- #
elif page == "Bulk Prediction":

    st.subheader("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        df.columns = df.columns.str.lower()
        df_processed = pd.get_dummies(df)

        try:
            model_columns = model.feature_names_in_
        except:
            model_columns = model.get_booster().feature_names

        for col in model_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        df_processed = df_processed[model_columns]

        preds = model.predict(df_processed)
        probs = model.predict_proba(df_processed)[:, 1]

        df["Prediction"] = preds
        df["Churn Probability"] = probs

        st.markdown("### 📊 Results")
        st.dataframe(df)

        # Graph
        churn_counts = df["Prediction"].value_counts().reindex([0,1], fill_value=0)

        fig, ax = plt.subplots()
        bars = ax.bar(["No Churn", "Churn"], churn_counts)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    int(bar.get_height()),
                    ha='center')

        st.pyplot(fig)