import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

    domain = st.selectbox("Select Industry", ["Telecom", "Banking", "E-commerce"])

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Customer Name")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 0, 100)
        senior = 1 if age >= 60 else 0

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

    st.markdown("---")

    if st.button("🚀 Predict Churn"):

        # ---------------- SMART DEMO LOGIC ---------------- #
        prob = 0.3

        if tenure < 6:
            prob += 0.3
        if monthly > 500:
            prob += 0.2
        if contract == "Month-to-month":
            prob += 0.2
        if senior == 1:
            prob += 0.1

        prob = min(prob, 0.95)
        pred = 1 if prob > 0.5 else 0

        st.markdown("### 📌 Prediction Result")

        if pred == 1:
            st.error(f"⚠ {name if name else 'Customer'} WILL CHURN (Risk: {prob})")
        else:
            st.success(f"✅ {name if name else 'Customer'} will STAY (Risk: {prob})")

        # Risk level
        if prob > 0.7:
            st.write("🔴 Risk Level: HIGH")
        elif prob > 0.4:
            st.write("🟡 Risk Level: MEDIUM")
        else:
            st.write("🟢 Risk Level: LOW")

        # Explanation
        st.markdown("### 💡 Explanation")

        if pred == 1:
            st.warning("Customer shows potential churn behavior based on inputs like low tenure, high spending, or flexible contract.")
        else:
            st.success("Customer appears stable with long-term or consistent engagement.")

        # ---------------- VISUAL ---------------- #
        st.markdown("---")
        st.subheader("📊 Feature Overview")

        features = ["Tenure", "Monthly", "Total", "Age"]
        values = [tenure, monthly, total, age]

        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.invert_yaxis()
        st.pyplot(fig)

# ---------------- BULK ---------------- #
elif page == "Bulk Prediction":

    st.subheader("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        # ---------------- DEMO BULK LOGIC ---------------- #
        predictions = []
        probabilities = []

        for _, row in df.iterrows():
            prob = 0.3

            if "tenure" in df.columns and row["tenure"] < 6:
                prob += 0.3
            if "monthly" in df.columns and row["monthly"] > 500:
                prob += 0.2

            prob = min(prob, 0.95)
            pred = "Churn" if prob > 0.5 else "No Churn"

            predictions.append(pred)
            probabilities.append(round(prob, 2))

        df["Prediction"] = predictions
        df["Churn Probability"] = probabilities

        st.markdown("### 📊 Results")
        st.dataframe(df)

        churn_counts = df["Prediction"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(churn_counts.index, churn_counts.values)

        st.pyplot(fig)
