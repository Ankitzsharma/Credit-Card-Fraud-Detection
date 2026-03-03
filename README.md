# 💳 Credit Card Fraud Detection System

A production-ready machine learning system to detect fraudulent credit card transactions in real-time. This project was developed as a comprehensive solution for financial institutions to minimize losses due to fraudulent activities.

---

## 🚀 Project Overview

This project aims to build a robust, end-to-end fraud detection pipeline. It includes data exploration, handling extreme class imbalance, model training with XGBoost, and a real-time inference system using FastAPI and Docker.

### Key Objectives
- **Catch Fraud Early**: Implement a high-recall model to identify as many fraudulent transactions as possible.
- **Minimize False Alarms**: Maintain high precision to ensure a smooth user experience for legitimate customers.
- **Real-time Performance**: Achieve sub-200ms inference latency for seamless integration into transaction flows.

---

## 🛠️ Architecture

```text
+------------------+       +-------------------------+       +-------------------+
|  Raw Data (CSV)  | ----> | EDA & Preprocessing     | ----> | Model Training    |
|                  |       | (SMOTE, Scaling)        |       | (XGBoost, Tuning) |
+------------------+       +-------------------------+       +-------------------+
                                                                     |
                                                                     v
+------------------+       +-------------------------+       +-------------------+
|  Docker Container| <---- | FastAPI Application     | <---- | Best Model (.pkl) |
|  (Deployment)    |       | (POST /predict)         |       |                   |
+------------------+       +-------------------------+       +-------------------+
```

---

## 🔎 Exploratory Data Analysis

- Dataset contains 284,807 transactions with only 0.17% fraud cases, highlighting extreme class imbalance.
- Fraudulent transactions show higher variance in specific PCA components (V14, V17).
- Transaction amount distribution indicates fraud cases often cluster in mid-range values rather than extreme amounts.
- Strong need for imbalance handling due to skewed class distribution.

## 📈 Model Performance

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|--------|----------|---------|
| Logistic Regression| 0.89      | 0.85   | 0.87     | 0.94    |
| Random Forest      | 0.91      | 0.88   | 0.89     | 0.96    |
| XGBoost (Final)    | 0.93      | 0.91   | 0.92     | 0.97    |

The final XGBoost model was selected based on superior recall and balanced precision, minimizing financial risk while reducing customer friction.

## 🔄 Model Lifecycle Strategy

- Model versioning using artifact storage.
- Prediction logging for audit and fraud investigation.
- Periodic retraining with fresh labeled transaction data.
- Performance monitoring dashboard integration.

## 📊 Business Impact

- **Loss Prevention**: By achieving ≥ 90% recall, the system can potentially save millions in fraudulent chargebacks.
- **Operational Efficiency**: Automated detection reduces the need for manual review of every transaction.
- **Customer Trust**: Maintaining high precision (≥ 90%) ensures legitimate transactions aren't blocked, preserving customer satisfaction.

---

## ⚙️ Technical Stack

- **Python 3.10**: Core programming language.
- **Scikit-learn & XGBoost**: Machine learning framework and gradient boosting.
- **Pandas & NumPy**: Data manipulation and numerical computation.
- **FastAPI**: High-performance web framework for the API.
- **Docker**: Containerization for consistent deployment.
- **Pytest**: Automated testing for API and model logic.

---

## 📁 Project Structure

```text
📁 data/                # Raw transaction dataset
📁 notebooks/           # Jupyter notebooks for EDA and experimentation
📁 src/                 # Production-grade modular Python scripts
📁 models/              # Saved model and scaler artifacts
📁 app/                 # FastAPI application code
📁 tests/               # Automated unit and integration tests
Dockerfile              # Docker configuration
requirements.txt        # Project dependencies
README.md               # Project documentation
```

### 💻 User Interface (Streamlit)
For a professional, interactive experience, we've included a Streamlit UI that connects to the FastAPI backend.
- **Run Locally:**
  ```bash
  python -m streamlit run app_ui.py
  ```
- **Features:** 
  - Real-time transaction risk scoring.
  - Interactive inputs for amount and time.
  - Expandable advanced feature (PCA) section.
  - Visual risk categorization (Low, Medium, High).
  - Model performance summary and business impact overview.

---

## 🌐 Live Demo & Deployment

This project is fully containerized and ready for deployment to cloud platforms like Render, Railway, or Heroku.

### 1. Build and Run via Docker
To run both the API and the UI in a single container:
```bash
docker build -t fraud-detection-system .
docker run -p 8000:8000 -p 8501:8501 fraud-detection-system
```
- **FastAPI API:** [http://localhost:8000](http://localhost:8000)
- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

### 2. Live Demo Deployment (Example)
- **Render:** Connect this repository, set the build command to `pip install -r requirements.txt`, and the start command to `./start.sh`.
- **Railway:** Similar to Render, Railway will automatically detect the Dockerfile and deploy the services.

---

## 📈 Monitoring & Maintenance Plan

### 1. Data Drift Monitoring
- **Approach**: Monitor the distribution of input features (especially PCA components) over time using statistical tests (e.g., Kolmogorov-Smirnov).
- **Tooling**: Integration with tools like Evidently AI or custom monitoring scripts.

### 2. Model Retraining Strategy
- **Frequency**: Retrain every 3 months or when performance drops below a predefined threshold (e.g., F1-score < 0.85).
- **Process**: Automated pipeline to ingest new labeled data, retrain with SMOTE, and validate against a champion model.

### 3. Threshold Adjustment
- **Strategy**: Dynamically adjust the classification threshold based on business priorities. If fraud losses increase, lower the threshold to prioritize recall.

### 4. Fraud-Loss Reduction Simulation
- Periodically run simulations on historical data to estimate the financial impact of the model's predictions versus the cost of false positives.

---

## 🛠️ Installation & Usage

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

### Docker Setup
1. Build the image:
   ```bash
   docker build -t fraud-detection-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 fraud-detection-api
   ```

---

## 📝 ATS-Optimized Highlights

- **Machine Learning**: Developed an end-to-end Fraud Detection System using Python, Scikit-learn, and XGBoost, achieving **90%+ precision and recall**.
- **Imbalanced Data**: Successfully handled extreme class imbalance (0.17% fraud) using **SMOTE** and stratified sampling to improve model robustness.
- **Model Deployment**: Deployed a production-ready **FastAPI** application containerized with **Docker**, ensuring scalable and consistent environment setup.
- **Real-time Inference**: Optimized inference pipeline to achieve **sub-200ms latency**, meeting production requirements for near-real-time transaction monitoring.
- **Production Engineering**: Built a modular project structure with automated testing using **Pytest**, ensuring high code quality and maintainability.

---

## 💬 Interview Talking Points

- **Why XGBoost?**: Chosen for its superior handling of non-linear relationships and its built-in regularization which prevents overfitting on imbalanced data.
- **Handling Imbalance**: Used SMOTE for oversampling the minority class in the training set, while keeping the test set representative of the real-world distribution.
- **Precision vs Recall**: In fraud detection, recall is often prioritized to catch all fraud, but high precision is maintained to avoid "customer friction" from false positives.
- **Production Challenges**: One major challenge was ensuring feature scaling was applied consistently between training and real-time inference, solved by saving the scaler object.
- **Deployment Reasoning**: FastAPI was chosen for its asynchronous capabilities and automatic Swagger documentation, making it ideal for high-throughput ML services.
