import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import os
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# -------- CONFIG -------
BUCKET_NAME = "practice-mlops-oppe"
MODEL_BLOB_PATH = "models/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

# -------- FASTAPI APP --------
app = FastAPI(title="Fraud Detection API")
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

model = None


# -------- INPUT SCHEMA --------
class Transaction(BaseModel):
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


# -------- STARTUP: LOAD MODEL FROM GCS --------
@app.on_event("startup")
def load_model():
    global model
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BLOB_PATH)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        model = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Model loaded from GCS")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# -------- PREDICTION ENDPOINT --------
@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = pd.DataFrame([transaction.dict()])

    with tracer.start_as_current_span("model_inference"):
        probability = model.predict_proba(data)[0][1]

    prediction = int(probability >= 0.5)


    return {
        "prediction": prediction,
        "probability": probability
    }
