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

    # ✅ AUTO CALCULATED TOTAL
    with col3:
        total = tenure * monthly
        st.metric(label=total_label, value=round(total, 2))

        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.markdown("---")

    # ✅ DEMO PREDICTION (NO MODEL)
    if st.button("🚀 Predict Churn"):

        # Fake logic for demo
        if monthly > 500 or tenure < 6:
            pred = 1
            prob = 0.78
        else:
            pred = 0
            prob = 0.22

        st.markdown("### 📌 Prediction Result")

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
        st.markdown("### 💡 Explanation")

        if pred == 1:
            st.warning("Customer may churn due to high spend or short engagement.")
        else:
            st.success("Customer is stable based on current behavior.")

        # Simple visualization
        st.markdown("---")
        st.subheader("📊 Feature Impact (Demo)")

        features = ["Tenure", "Monthly", "Total"]
        values = [tenure, monthly, total]

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

        # Auto-calc total if columns exist
        if "tenure" in df.columns and "monthlycharges" in df.columns:
            df["totalcharges"] = df["tenure"] * df["monthlycharges"]

        # Demo predictions
        df["Prediction"] = df["monthlycharges"].apply(lambda x: 1 if x > 500 else 0)
        df["Churn Probability"] = df["Prediction"].apply(lambda x: 0.75 if x == 1 else 0.25)

        st.markdown("### 📊 Results")
        st.dataframe(df)

        # Graph
        churn_counts = df["Prediction"].value_counts().reindex([0,1], fill_value=0)

        fig, ax = plt.subplots()
        ax.bar(["No Churn", "Churn"], churn_counts)
        st.pyplot(fig)
