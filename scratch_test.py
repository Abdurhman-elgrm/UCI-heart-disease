import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/heart_model.pkl")
model = joblib.load(MODEL_PATH)

payload = {
    "age": 55,
    "sex": "Male",
    "cp": "typical angina",
    "trestbps": 130,
    "chol": 200,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150,
    "exang": False,
    "oldpeak": 1.0,
    "slope": "flat",
    "ca": 0.0,
    "thal": "normal",
    "dataset": "Cleveland",
}

df = pd.DataFrame([payload])
print("DataFrame created:")
print(df)

try:
    prob = model["pipeline"].predict_proba(df)
    print("Probability:", prob)
except Exception as e:
    import traceback
    traceback.print_exc()
