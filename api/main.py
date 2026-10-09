import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Load the trained model (saved by src/train.py)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}") from exc

# ----------------------------------------------------------------------
# Request schema – must match the column order used during training
# ----------------------------------------------------------------------
class PatientRecord(BaseModel):
    age: int = Field(..., description="Age in years")
    sex: int = Field(..., description="1 = male, 0 = female")
    cp: int = Field(..., description="Chest pain type (0‑3)")
    trestbps: int = Field(..., description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., description="Resting ECG results (0‑2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of the peak exercise ST segment (0‑2)")
    ca: int = Field(..., description="Number of major vessels (0‑3) colored by fluoroscopy")
    thal: int = Field(..., description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)")
    dataset: int = Field(..., description="Dataset source (0‑2)")

app = FastAPI(
    title="UCI Heart Disease Predictor",
    version="0.1.0",
    description="A tiny API that returns the probability of heart disease using a pre‑trained XGBoost model."
)

@app.post("/predict")
def predict(record: PatientRecord):
    """Return the probability that the patient has heart disease."""
    X = [list(record.dict().values())]
    try:
        prob = model.predict_proba(X)[0, 1]
        label = int(prob >= 0.5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"probability": prob, "prediction": label}

