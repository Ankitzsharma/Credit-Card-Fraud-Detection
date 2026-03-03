import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load credit card transaction data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame, save_scaler: bool = True, scaler_path: str = 'models/scaler.pkl'):
    """
    Preprocess the dataset: scaling and splitting.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        save_scaler (bool): Whether to save the scaler for production use.
        scaler_path (str): Path to save the scaler.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test

def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        
    Returns:
        tuple: (X_train_res, y_train_res)
    """
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

if __name__ == "__main__":
    # Example usage
    data = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    print(f"Original shape: {X_train.shape}, Resampled shape: {X_train_res.shape}")
