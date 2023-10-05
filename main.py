from fastapi import FastAPI, Query
import pandas as pd
import joblib

app = FastAPI()

# Load the model from the correct file name
classifier = joblib.load('LRR.pkl')

def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"

@app.get("/")
async def read_root():
    return {"message": "Customer Churn Prediction For Vodafone PLC using FastAPI"}

@app.get("/predict/")
async def predict_churn(
    SeniorCitizen: int = Query(..., description="Select 1 for Yes and 0 for No"),
    Partner: str = Query(..., description="Do You Have a Partner? (Yes/No)"),
    Dependents: str = Query(..., description="Do You Have a Dependent? (Yes/No)"),
    tenure: int = Query(..., description="How Long Have You Been with Vodafone in Months?"),
    InternetService: str = Query(..., description="Internet Service Type (DSL/Fiber optic/No)"),
    OnlineSecurity: str = Query(..., description="Online Security (Yes/No/No internet service)"),
    OnlineBackup: str = Query(..., description="Online Backup (Yes/No/No internet service)"),
    DeviceProtection: str = Query(..., description="Device Protection (Yes/No/No internet service)"),
    TechSupport: str = Query(..., description="Tech Support (Yes/No/No internet service)"),
    StreamingTV: str = Query(..., description="Streaming TV (Yes/No/No internet service)"),
    StreamingMovies: str = Query(..., description="Streaming Movies (Yes/No/No internet service)"),
    Contract: str = Query(..., description="Contract Type (Month-to-month/One year/Two year)"),
    PaperlessBilling: str = Query(..., description="Paperless Billing (Yes/No)"),
    PaymentMethod: str = Query(..., description="Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))"),
    MonthlyCharges: float = Query(..., description="Monthly Charges"),
    TotalCharges: float = Query(..., description="Total Charges")
):
    input_data = [
        SeniorCitizen, Partner, Dependents, tenure, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges
    ]

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    pred = classifier.predict(input_df)
    output = classify(pred[0])

    response = {
        "prediction": output
    }

    return response

