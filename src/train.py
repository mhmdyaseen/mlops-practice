import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from google.cloud import storage

# ... (rest of your existing code for GCS functions) ...
BUCKET_NAME = "practice-mlops-oppe"
DATA_BLOB = "data/data.csv"
LOCAL_DATA_PATH = "transactions_data_2022.csv"
MODEL_LOCAL_PATH = "model.pkl"
MODEL_BLOB_PATH = "models/model.pkl"


def download_data_from_gcs():
    # ... (your existing download function) ...
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DATA_BLOB)
    blob.download_to_filename(LOCAL_DATA_PATH)
    print(f"✅ Downloaded data from gs://{BUCKET_NAME}/{DATA_BLOB}")


def upload_model_to_gcs():
    # ... (your existing upload function) ...
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_PATH)
    blob.upload_from_filename(MODEL_LOCAL_PATH)
    print(f"✅ Uploaded model to gs://{BUCKET_NAME}/{MODEL_BLOB_PATH}")


def train_model():
    # Step 1: Download data
    download_data_from_gcs()

    # Step 2: Load data
    df = pd.read_csv(LOCAL_DATA_PATH)

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )

    # --- FIX IS HERE ---
    # Set the tracking URI to your MLflow server address
    # Replace the placeholder URL below with your actual MLflow server URL
    mlflow.set_tracking_uri("http://34.29.148.119:8100/") 
    
    mlflow.set_experiment("fraud_detection_gcs")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("f1_score", f1)

        # The model logging part is correct
        mlflow.sklearn.log_model(model, name="model")

        print(f"✅ Validation F1-score: {f1:.4f}")

    # Step 3: Save model locally (Optional if using MLflow logging)
    joblib.dump(model, MODEL_LOCAL_PATH)
    
    # Step 4: Upload model to GCS (Optional if using MLflow logging)
    upload_model_to_gcs()


if __name__ == "__main__":
    train_model()

