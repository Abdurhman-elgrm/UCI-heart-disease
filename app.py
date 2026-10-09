# app.py
import joblib
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# --------------------------------------------------------------
# 1️⃣ Load the model (cached for the whole session)
# --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.pkl"

@st.cache_resource
def load_model():
    """Load the XGBoost model once and cache it."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"❌ Could not load model from {MODEL_PATH}: {exc}")
        st.stop()

model = load_model()

# --------------------------------------------------------------
# 2️⃣ Page layout
# --------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Predictor")
st.write(
    "Enter patient data below and click **Predict**. The model runs locally – no external API call is needed."
)

# --------------------------------------------------------------
# 3️⃣ Input widgets – order must match training columns
# --------------------------------------------------------------
def get_user_input():
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", options=["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[False, True], format_func=lambda x: "Yes" if x else "No")
    restecg = st.selectbox("Resting ECG Results", options=["normal", "lv hypertrophy", "st-t abnormality"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[False, True], format_func=lambda x: "Yes" if x else "No")
    oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["flat", "downsloping", "upsloping"])
    ca = st.selectbox("Number of Major Vessels (0‑3)", options=[0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia", options=["normal", "fixed defect", "reversable defect"])
    dataset = st.selectbox("Dataset Source", options=["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    return {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "dataset": dataset,
    }

payload = get_user_input()

# --------------------------------------------------------------
# 4️⃣ Prediction
# --------------------------------------------------------------
if st.button("🔮 Predict"):
    with st.spinner("Running model…"):
        X = pd.DataFrame([payload])
        try:
            pipeline = model["pipeline"]
            threshold = model.get("threshold", 0.5)
            prob = pipeline.predict_proba(X)[0, 1]
            label = int(prob >= threshold)
            st.success(f"**Probability of heart disease:** {prob:.2%}")
            st.info(f"**Prediction (optimal threshold {threshold:.2f}):** {'🛑 Disease' if label else '✅ No disease'}")
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
