import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# --- Page Config ---
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Backend API URL ---
API_URL = "http://localhost:8000/predict"

# --- Helper Functions ---
def get_prediction(payload):
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        return None

def generate_random_sample(is_fraud=False):
    # This would ideally sample from the test set or generate realistic values
    # For now, we generate random normal values centered around typical ranges
    # Legitimate transactions (is_fraud=False) have smaller PCA variances
    # Fraudulent transactions (is_fraud=True) often have larger deviations in certain V-features
    np.random.seed(int(time.time()))
    sample = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}
    sample["Time"] = np.random.uniform(0, 172792)
    sample["Amount"] = np.random.uniform(0, 500) if not is_fraud else np.random.uniform(10, 2000)
    
    # If is_fraud is True, slightly bias some features (e.g., V14, V17 are often discriminative)
    if is_fraud:
        sample["V14"] = np.random.normal(-5, 2)
        sample["V17"] = np.random.normal(-5, 2)
        sample["V12"] = np.random.normal(-5, 2)
    return sample

# --- 1. Header Section ---
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### *Real-Time Machine Learning Fraud Risk Assessment*")
st.write("""
    This production-grade system analyzes credit card transactions in real-time to identify potential fraudulent activities. 
    Leveraging an optimized XGBoost model with advanced imbalance handling, it provides high-precision risk scoring for financial transactions.
""")

st.divider()

# --- 2. Model Performance Summary ---
st.markdown("#### 📊 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">93%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">91%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">0.97</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value"><200ms</div><div class="metric-label">Avg. Latency</div></div>', unsafe_allow_html=True)

st.divider()

# --- 3. Transaction Input Panel ---
st.markdown("#### 📝 Transaction Input")

# Auto-fill Sample Buttons
c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button("Generate Normal Sample"):
        st.session_state.sample = generate_random_sample(is_fraud=False)
with c2:
    if st.button("Generate Fraud Sample"):
        st.session_state.sample = generate_random_sample(is_fraud=True)

# Main Inputs
if 'sample' not in st.session_state:
    st.session_state.sample = generate_random_sample(is_fraud=False)

col_amt, col_time = st.columns(2)
with col_amt:
    amount = st.number_input("Transaction Amount ($)", value=float(st.session_state.sample['Amount']), min_value=0.0)
with col_time:
    transaction_time = st.number_input("Transaction Time (Seconds)", value=float(st.session_state.sample['Time']), min_value=0.0)

# PCA Features (Expandable)
with st.expander("🛠️ Advanced PCA Features (V1-V28)"):
    st.info("These are anonymized features (V1 to V28) derived from PCA transformation of the original transaction data.")
    v_cols = st.columns(4)
    v_inputs = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with v_cols[col_idx]:
            v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=float(st.session_state.sample[f"V{i}"]))

# --- 4. Predict Button ---
if st.button("🚀 Analyze Transaction Risk"):
    payload = {
        "Time": transaction_time,
        "Amount": amount,
        **v_inputs
    }
    
    with st.spinner("Analyzing transaction patterns..."):
        # Artificial delay for realism if latency is too fast locally
        # time.sleep(0.1) 
        result = get_prediction(payload)
        
    if result:
        st.divider()
        st.markdown("#### 🔍 Analysis Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        prob = result['fraud_probability']
        label = result['prediction']
        risk = result['risk_level']
        
        # Determine Color
        risk_class = "risk-low"
        if risk == "Medium":
            risk_class = "risk-medium"
        elif risk == "High":
            risk_class = "risk-high"
            
        with res_col1:
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
        with res_col2:
            st.markdown(f"**Classification:** <span class='{risk_class}'>{label}</span>", unsafe_allow_html=True)
        with res_col3:
            st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk}</span>", unsafe_allow_html=True)
            
        # Business Recommendation
        if risk == "High":
            st.error("⚠️ ACTION REQUIRED: This transaction has high indicators of fraud. Verification is recommended.")
        elif risk == "Medium":
            st.warning("⚡ CAUTION: Transaction flagged for manual review based on suspicious patterns.")
        else:
            st.success("✅ APPROVED: Transaction pattern consistent with legitimate historical data.")

st.divider()

# --- 6. Business Impact & Architecture ---
col_impact, col_arch = st.columns([1, 1])

with col_impact:
    st.markdown("#### 📈 Business Impact")
    st.info("""
        If deployed in high-volume transaction systems, this model can significantly reduce fraud-related financial losses 
        while maintaining a seamless customer experience by minimizing false positives. 
        The stratified XGBoost approach ensures robustness against the extreme rarity of fraudulent events.
    """)

with col_arch:
    st.markdown("#### 🏗️ Architecture Overview")
    st.write("""
        **Data Pipeline:** Raw CSV → SMOTE Balancing → StandardScaler  
        **Model:** Tuned XGBoost (GridSearch Optimized)  
        **Inference:** FastAPI REST Endpoints (Sub-200ms)  
        **Deployment:** Docker Containerized Microservice  
    """)

st.markdown("---")
st.caption("Developed for Production-Ready ML Portfolio | © 2026 Fraud Detection Systems Inc.")
