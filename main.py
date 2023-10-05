from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

import joblib
import pandas as pd

app = FastAPI()

# Load the model from the correct file name
classifier = joblib.load('LR.pkl')


class ChurnPredictionInput(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ChurnPredictionOutput(BaseModel):
    prediction: int
    result: str


def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"


@app.post("/predict/")
def predict_churn(inputs: ChurnPredictionInput) -> ChurnPredictionOutput:
    input_data = [
        inputs.SeniorCitizen, inputs.Partner, inputs.Dependents, inputs.tenure,
        inputs.InternetService, inputs.OnlineSecurity, inputs.OnlineBackup,
        inputs.DeviceProtection, inputs.TechSupport, inputs.StreamingTV,
        inputs.StreamingMovies, inputs.Contract, inputs.PaperlessBilling,
        inputs.PaymentMethod, inputs.MonthlyCharges, inputs.TotalCharges
    ]

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    pred = classifier.predict(input_df)
    output = classify(pred[0])

    return {"prediction": pred[0], "result": output}
