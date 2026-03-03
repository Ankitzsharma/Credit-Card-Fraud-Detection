from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import os
import pandas as pd
from data_preprocessing import load_data, preprocess_data, handle_imbalance

def train_best_model(X_train, y_train, model_path: str = 'models/best_model.pkl'):
    """
    Train the best model using GridSearchCV and save it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_path (str): Path to save the trained model.
        
    Returns:
        XGBClassifier: Best estimator.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    
    grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, scoring='f1', cv=StratifiedKFold(n_splits=3), verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/creditcard.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    
    # Train and save the best model
    best_model = train_best_model(X_train_res, y_train_res)
