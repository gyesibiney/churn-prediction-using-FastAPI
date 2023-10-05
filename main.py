from fastapi import FastAPI, Query
from enum import Enum
import pandas as pd
import joblib

app = FastAPI()

# Load the model from the correct file name
classifier = joblib.load('LR.pkl')

def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"

class YesNoEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class InternetServiceEnum(str, Enum):
    DSL = "DSL"
    FiberOptic = "Fiber optic"
    No = "No"

class ContractEnum(str, Enum):
    MonthToMonth = "Month-to-month"
    OneYear = "One year"
    TwoYear = "Two year"

class PaymentMethodEnum(str, Enum):
    ElectronicCheck = "Electronic check"
    MailedCheck = "Mailed check"
    BankTransfer = "Bank transfer (automatic)"
    CreditCard = "Credit card (automatic)"

@app.get("/")
async def read_root():
    return {"message": "Customer Churn Prediction For Vodafone PLC using FastAPI"}

@app.get("/predict/")
async def predict_churn(
    SeniorCitizen: int = Query(..., description="Select 1 for Yes and 0 for No"),
    Partner: YesNoEnum = Query(..., description="Do You Have a Partner?"),
    Dependents: YesNoEnum = Query(..., description="Do You Have a Dependent?"),
    tenure: int = Query(..., description="How Long Have You Been with Vodafone in Months?"),
    InternetService: InternetServiceEnum = Query(..., description="Internet Service Type"),
    OnlineSecurity: InternetServiceEnum = Query(..., description="Online Security"),
    OnlineBackup: InternetServiceEnum = Query(..., description="Online Backup"),
    DeviceProtection: InternetServiceEnum = Query(..., description="Device Protection"),
    TechSupport: InternetServiceEnum = Query(..., description="Tech Support"),
    StreamingTV: InternetServiceEnum = Query(..., description="Streaming TV"),
    StreamingMovies: InternetServiceEnum = Query(..., description="Streaming Movies"),
    Contract: ContractEnum = Query(..., description="Contract Type"),
    PaperlessBilling: YesNoEnum = Query(..., description="Paperless Billing"),
    PaymentMethod: PaymentMethodEnum = Query(..., description="Payment Method"),
    MonthlyCharges: float = Query(..., description="Monthly Charges"),
    TotalCharges: float = Query(..., description="Total Charges")
):
    input_data = [
        SeniorCitizen, Partner.value, Dependents.value, tenure, InternetService.value,
        OnlineSecurity.value, OnlineBackup.value, DeviceProtection.value, TechSupport.value,
        StreamingTV.value, StreamingMovies.value, Contract.value, PaperlessBilling.value,
        PaymentMethod.value, MonthlyCharges, TotalCharges
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
