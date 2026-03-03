from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_fraud, load_model_and_scaler

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

try:
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: TransactionData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    # Convert Pydantic model to DataFrame
    transaction_dict = data.dict()
    transaction_df = pd.DataFrame([transaction_dict])
    
    # Get prediction
    results = predict_fraud(transaction_df, model, scaler)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
