#!/bin/bash

# Start FastAPI in the background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
python -m streamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0
