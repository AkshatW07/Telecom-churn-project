import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Telecom Churn Prediction App",
    page_icon="📊",
    layout="centered"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("📊 Telecom Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on service and contract details.")
st.markdown("---")


st.write("Enter customer details to predict churn risk.")

# Input fields
st.sidebar.header("Enter Customer Details")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Load feature template
feature_template = pd.read_csv(
    os.path.join(BASE_DIR, "..", "Data", "churn_features.csv")
)


feature_template = feature_template.drop("Churn", axis=1)

input_data = pd.DataFrame(
    np.zeros((1, feature_template.shape[1])),
    columns=feature_template.columns
)
# Add missing columns if not present
for col in ["SeniorCitizen", "TotalCharges", "Unnamed: 0"]:
    if col not in input_data.columns:
        input_data[col] = 0


# Fill user inputs
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges

input_data = input_data[model.feature_names_in_]


if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data["InternetService_No"] = 1

if payment_method == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_data["PaymentMethod_Mailed check"] = 1
elif payment_method == "Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"] = 1

# Scale numeric features
input_data[["tenure", "MonthlyCharges"]] = scaler.transform(
    input_data[["tenure", "MonthlyCharges"]]
)

# Predict
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Churn Risk\n\nProbability: {probability:.2f}")

