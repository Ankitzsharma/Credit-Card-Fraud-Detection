import joblib
import pandas as pd
import numpy as np
import os

def load_model_and_scaler(model_path: str = 'models/best_model.pkl', scaler_path: str = 'models/scaler.pkl'):
    """
    Load the trained model and scaler for inference.
    
    Args:
        model_path (str): Path to the saved model.
        scaler_path (str): Path to the saved scaler.
        
    Returns:
        tuple: (model, scaler)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_fraud(data: pd.DataFrame, model, scaler):
    """
    Predict fraud probability and class for given transaction data.
    
    Args:
        data (pd.DataFrame): Input transaction features.
        model: Trained XGBoost model.
        scaler: Trained StandardScaler.
        
    Returns:
        dict: Prediction results.
    """
    # Preprocess features (Time and Amount)
    data[['Amount', 'Time']] = scaler.transform(data[['Amount', 'Time']])
    
    # Predict probability and class
    prob = model.predict_proba(data)[:, 1][0]
    prediction = "Fraud" if prob > 0.5 else "Not Fraud"
    
    # Risk level based on probability
    if prob < 0.3:
        risk_level = "Low"
    elif prob < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        "fraud_probability": float(prob),
        "prediction": prediction,
        "risk_level": risk_level
    }

if __name__ == "__main__":
    # Example prediction
    # model, scaler = load_model_and_scaler()
    # sample_data = pd.DataFrame(...)
    # results = predict_fraud(sample_data, model, scaler)
    # print(results)
    pass
