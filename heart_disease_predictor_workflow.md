# UCI Heart Disease Predictor — End-to-End Workflow

A complete project designed to teach you medical data cleaning, transformation, and production machine learning using the classic UCI Heart Disease dataset.

---

## 0. Define the Problem Before Touching Data

Be specific about what you're predicting — it determines your pipeline, metrics, and API schemas:

- **Binary Classification (Standard & Recommended):** Heart Disease Present vs. Absent (0 = No Disease, 1 = Disease).
- **Multiclass Classification:** Disease Severity (0: No disease, 1: Mild, 2: Moderate, 3: Severe, 4: Critical).
- **Risk Score / Probability Estimation:** Outputting calibrated probability (0.0 to 1.0) so doctors can assess clinical risk thresholds.

**Recommendation:** Start with **Binary Classification** with predicted probabilities. In clinical domains, **Recall (Sensitivity)** and **ROC-AUC** matter far more than raw accuracy because false negatives (missing a sick patient) are dangerous.

---

## 1. Data Understanding & Column Dictionary

The dataset lives in `data/raw/heart_disease_uci.csv`. It combines patient records from 4 institutions (Cleveland, Hungarian, Switzerland, Long Beach VA).

| Feature | Type | Description & Values |
|---|---|---|
| `id` | Identifier | Patient ID (drop before modeling) |
| `age` | Numeric | Age in years |
| `sex` | Categorical | Sex (`Male`, `Female` or `1`, `0`) |
| `dataset` | Categorical | Origin hospital (Cleveland, Hungary, VA, Switzerland) |
| `cp` | Categorical | Chest pain type (`typical angina`, `atypical angina`, `non-anginal`, `asymptomatic`) |
| `trestbps` | Numeric | Resting blood pressure (in mm Hg on admission) |
| `chol` | Numeric | Serum cholesterol in mg/dl (watch out for 0s) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl (`TRUE`, `FALSE`) |
| `restecg` | Categorical | Resting ECG (`normal`, `st-t abnormality`, `lv hypertrophy`) |
| `thalach` | Numeric | Maximum heart rate achieved during exercise |
| `exang` | Binary | Exercise-induced angina (`TRUE`, `FALSE`) |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment (`upsloping`, `flat`, `downsloping`) |
| `ca` | Numeric/Cat | Major vessels (0–3) colored by fluoroscopy |
| `thal` | Categorical | Thalassemia (`normal`, `fixed defect`, `reversable defect`) |
| `num` | Target | Diagnosis integer (0 = healthy, 1–4 = disease) |

---

## 2. Project Architecture & File Structure

Here is how your existing repository structure maps to every phase:

```text
UCI heart disease/
├── data/
│   ├── raw/
│   │   └── heart_disease_uci.csv   # untouched original data
│   └── processed/
│       └── cleaned_heart.csv       # model-ready transformed data
├── notebooks/
│   └── EDA.ipynb                   # exploratory analysis & visualization
├── src/
│   ├── inguest.py                  # ingestion & raw schema validation
│   ├── clean.ipynb                 # dirty data cleaning & imputation
│   ├── feature.py                  # encoders, scalers & transformation pipeline
│   ├── trian.py                    # model training, evaluation & artifact saving
│   └── predict.py                  # offline inference verification
├── models/
│   └── heart_model.pkl             # saved pipeline artifact
├── api/
│   └── main.py                     # FastAPI REST API serving /predict
├── requirements.txt
└── .env
```

---

## 3. Data Ingestion (`src/inguest.py`)

- **Role:** Safely read the raw data from `data/raw/heart_disease_uci.csv`, verify schema integrity, and provide a single source of truth for loading.
- **Key Actions:**
  1. Check file existence and load with pandas.
  2. Standardize column names (lowercase, strip whitespace).
  3. Validate expected columns are present.
  4. Expose a helper function (e.g. `load_raw_data()`).

```python
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/heart_disease_uci.csv")

def load_raw_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found at {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
```

---

## 4. Exploratory Data Analysis (`notebooks/EDA.ipynb`)

Open `notebooks/EDA.ipynb` before making any modeling or cleaning decisions.

### 4.1 What to investigate
1. **Target distribution:** What percentage of patients have `num == 0` vs `num > 0`?
2. **Missing data patterns:** The 4 institutions have very different missingness (e.g., Hungarian and VA have many missing `ca` and `thal` values).
3. **Biological impossibilities:** Look for `chol == 0` or resting blood pressure `trestbps == 0`. (A cholesterol of 0 mg/dl is clinically impossible — it means missing measurement!).
4. **Key clinical correlations:**
   - How does `thalach` (max heart rate) correlate with heart disease? (Usually lower max HR = higher risk).
   - How does `oldpeak` (ST depression) correlate? (Higher ST depression = higher risk).
   - Relationship between chest pain type (`cp == asymptomatic`) and actual disease presence.

---

## 5. Data Cleaning (`src/clean.ipynb`)

This is where you make critical clinical data cleaning decisions. Document each choice:

### 5.1 Step 1: Drop identifiers
Drop `id` — it carries zero predictive signal and causes data leakage if ordered.

### 5.2 Step 2: Fix impossible zeroes (Biological reality)
In tabular medical data, zeroes are often encoded instead of `NaN`.
```python
import numpy as np

# A resting blood pressure or cholesterol of 0 is missing data
df['chol'] = df['chol'].replace(0, np.nan)
df['trestbps'] = df['trestbps'].replace(0, np.nan)
```

### 5.3 Step 3: Target binarization
Convert the multiclass target `num` into binary `target`:
```python
# 0 = No Disease, 1 = Disease (values 1, 2, 3, 4)
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns=['num'])
```

### 5.4 Step 4: Handle missing values (`NaN`)
- Numeric columns (`chol`, `trestbps`, `thalach`, `oldpeak`): Impute using **median**.
- Categorical columns (`ca`, `thal`, `slope`): Impute using **mode** or add a separate `'missing'` category to avoid throwing away Hungarian/VA rows.
- Save output: export clean dataframe to `data/processed/cleaned_heart.csv`.

---

## 6. Feature Engineering & Preprocessing (`src/feature.py`)

Transform clean raw values into numeric matrices that algorithms can digest.

### 6.1 Transformations
- **Categoricals:** One-Hot Encode nominal variables (`cp`, `restecg`, `slope`, `thal`, `sex`).
- **Numerics:** Scale features with `StandardScaler` (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).

### 6.2 Golden Rule: Build a Scikit-Learn `ColumnTransformer`
Never fit scalers on the entire dataset before splitting — that leaks test data! Always package transformations in a reusable transformer:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

---

## 7. Model Training & Evaluation (`src/trian.py`)

### 7.1 Train / Test Split
Use **Stratified Split** because target class proportions must remain balanced across both sets:
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 7.2 Models to Benchmark
1. **Logistic Regression:** Clinical baseline; coefficients are directly interpretable by doctors as odds ratios.
2. **Random Forest Classifier:** Handles non-linear feature interactions and categorical splits.
3. **XGBoost / LightGBM:** Modern tabular benchmark for highest predictive accuracy.

### 7.3 Clinical Evaluation Metrics (Not just Accuracy!)
- **Recall / Sensitivity:** $\frac{TP}{TP + FN}$ — Critical! You want to minimize False Negatives (never tell a heart disease patient they are healthy).
- **ROC-AUC:** Measures discrimination capability across all classification probability thresholds.
- **Log-Loss / Brier Score:** Evaluates whether predicted probabilities reflect true probabilities.
- **Confusion Matrix:** Inspect True Negatives, False Positives, False Negatives, True Positives.

### 7.4 Save the Full Pipeline
Combine preprocessing + model into a single scikit-learn `Pipeline` and persist to `models/heart_model.pkl`:
```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save the unified artifact
joblib.dump(pipeline, "models/heart_model.pkl")
```

---

## 8. Offline Inference Verification (`src/predict.py`)

Before deploying to FastAPI, write a sanity-check script to verify that:
1. `models/heart_model.pkl` loads cleanly.
2. It accepts raw, un-transformed patient dictionaries and executes end-to-end preprocessing + prediction without errors.

```python
import joblib
import pandas as pd

def predict_single_patient(patient_data: dict):
    model = joblib.load("models/heart_model.pkl")
    df = pd.DataFrame([patient_data])
    
    prob = model.predict_proba(df)[0, 1]
    prediction = int(prob >= 0.5)
    
    return {
        "heart_disease_probability": round(float(prob), 4),
        "prediction": prediction,
        "risk_level": "High" if prob >= 0.7 else ("Moderate" if prob >= 0.4 else "Low")
    }

if __name__ == "__main__":
    sample_patient = {
        "age": 63, "sex": "Male", "cp": "typical angina",
        "trestbps": 145, "chol": 233, "fbs": True,
        "restecg": "lv hypertrophy", "thalach": 150, "exang": False,
        "oldpeak": 2.3, "slope": "downsloping", "ca": 0.0, "thal": "fixed defect"
    }
    result = predict_single_patient(sample_patient)
    print("Inference result:", result)
```

---

## 9. Production API Serving (`api/main.py`)

Expose your machine learning pipeline as a high-performance REST API with automated validation using FastAPI and Pydantic.

### 9.1 Implementation
In `api/main.py`:
- Define a Pydantic `PatientInput` schema specifying ranges (e.g. `age > 0`, `trestbps > 0`).
- Load the model once at startup.
- Expose `GET /` or `GET /health` for container health probes.
- Expose `POST /predict` returning prediction and probability.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="UCI Heart Disease Prediction API", version="1.0.0")

MODEL_PATH = Path("models/heart_model.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, example=58)
    sex: str = Field(..., example="Male")
    cp: str = Field(..., example="asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, example=130.0)
    chol: float = Field(..., ge=80, le=600, example=240.0)
    fbs: bool = Field(..., example=False)
    restecg: str = Field(..., example="normal")
    thalach: float = Field(..., ge=40, le=250, example=160.0)
    exang: bool = Field(..., example=False)
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=1.2)
    slope: str = Field(..., example="flat")
    ca: float = Field(0.0, ge=0.0, le=3.0, example=0.0)
    thal: str = Field(..., example="normal")

class PredictionOutput(BaseModel):
    has_heart_disease: bool
    risk_probability: float
    risk_level: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
def predict_heart_disease(patient: PatientInput):
    try:
        input_df = pd.DataFrame([patient.model_dump()])
        prob = float(model.predict_proba(input_df)[0, 1])
        prediction = bool(prob >= 0.5)
        risk = "High" if prob >= 0.7 else ("Moderate" if prob >= 0.4 else "Low")
        
        return PredictionOutput(
            has_heart_disease=prediction,
            risk_probability=round(prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 9.2 Launching the Server
Run from project root:
```powershell
uvicorn api.main:app --reload
```
Access the interactive Swagger UI documentation at:  
👉 **`http://127.0.0.1:8000/docs`**

---

## 10. Suggested Build Order Checklist

Follow this checklist step-by-step:

- [ ] **1. Ingest:** Implement `src/inguest.py` to safely load raw CSV from `data/raw/`.
- [ ] **2. Explore:** Run `notebooks/EDA.ipynb` to inspect target balance, impossible 0s, and feature correlations.
- [ ] **3. Clean:** Implement `src/clean.ipynb` to fix missing values/0s, binarize target, and save `data/processed/cleaned_heart.csv`.
- [ ] **4. Features:** Write reusable `ColumnTransformer` inside `src/feature.py`.
- [ ] **5. Train:** Build training pipeline in `src/trian.py`, evaluate Recall & ROC-AUC, and save `models/heart_model.pkl`.
- [ ] **6. Sanity Check:** Run `src/predict.py` with mock patient data to verify pipeline loading.
- [ ] **7. API:** Implement `api/main.py` with Pydantic validation and `POST /predict`.
- [ ] **8. Run Server:** Start `uvicorn api.main:app --reload` and test live via `/docs`.

---

## Where You'll Learn the Most in this Medical Dataset

1. **Spotting biological domain traps:** Catching that `chol = 0` is missing data, not super-human arteries.
2. **Clinical metric priorities:** Learning why 90% accuracy with low recall is a dangerous model in healthcare.
3. **Scikit-Learn Pipeline unification:** Packaging preprocessing + model into one artifact so your FastAPI app never suffers from train/serving skew.

