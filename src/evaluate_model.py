from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import os
import pandas as pd
from data_preprocessing import load_data, preprocess_data

def evaluate_model(model_path: str, X_test, y_test):
    """
    Evaluate the model on test data and print metrics.
    
    Args:
        model_path (str): Path to the saved model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        
    Returns:
        None
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Evaluate model
    evaluate_model('models/best_model.pkl', X_test, y_test)
