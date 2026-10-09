import joblib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.pkl"


def predict_single_patient(patient_data: dict):
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    df = pd.DataFrame([patient_data])
    prob = float(pipeline.predict_proba(df)[0, 1])
    prediction = int(prob >= threshold)

    return {
        "heart_disease_probability": round(prob, 4),
        "prediction": prediction,
        "risk_level": "High" if prob >= 0.7 else ("Moderate" if prob >= 0.4 else "Low"),
    }


if __name__ == "__main__":
    sample_patient = {
        "age": 63, "sex": "Male", "dataset": "Cleveland", "cp": "typical angina",
        "trestbps": 145, "chol": 233, "fbs": True,
        "restecg": "lv hypertrophy", "thalch": 150, "exang": False,
        "oldpeak": 2.3, "slope": "downsloping", "ca": 0.0, "thal": "fixed defect",
    }
    result = predict_single_patient(sample_patient)
    print("Inference result:", result)
