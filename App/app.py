from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI(title="Telecom Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Load feature template
feature_template = pd.read_csv(
    os.path.join(BASE_DIR, "..", "Data", "churn_features.csv")
)

feature_template = feature_template.drop("Churn", axis=1)

# ---------------------------
# Input Schema
# ---------------------------
class CustomerInput(BaseModel):
    contract: str
    tenure: int
    monthly_charges: float
    internet_service: str
    payment_method: str


# ---------------------------
# Helper Function
# ---------------------------
def preprocess_input(data: CustomerInput):

    input_df = pd.DataFrame(
        np.zeros((1, feature_template.shape[1])),
        columns=feature_template.columns
    )

    # Add missing columns
    for col in ["SeniorCitizen", "TotalCharges", "Unnamed: 0"]:
        if col not in input_df.columns:
            input_df[col] = 0

    # Fill numeric values
    input_df["tenure"] = data.tenure
    input_df["MonthlyCharges"] = data.monthly_charges

    # Contract
    if data.contract == "One year":
        input_df["Contract_One year"] = 1
    elif data.contract == "Two year":
        input_df["Contract_Two year"] = 1

    # Internet Service
    if data.internet_service == "Fiber optic":
        input_df["InternetService_Fiber optic"] = 1
    elif data.internet_service == "No":
        input_df["InternetService_No"] = 1

    # Payment Method
    if data.payment_method == "Electronic check":
        input_df["PaymentMethod_Electronic check"] = 1
    elif data.payment_method == "Mailed check":
        input_df["PaymentMethod_Mailed check"] = 1
    elif data.payment_method == "Credit card (automatic)":
        input_df["PaymentMethod_Credit card (automatic)"] = 1

    # Keep same feature order as model
    input_df = input_df[model.feature_names_in_]

    # Scale numeric features
    input_df[["tenure", "MonthlyCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges"]]
    )

    return input_df


# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
def home():
    return {"message": "Telecom Churn Prediction API is running!"}


@app.post("/predict")
def predict_churn(customer: CustomerInput):

    input_data = preprocess_input(customer)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = {
        "churn_prediction": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(float(probability), 2)
    }

    return result
