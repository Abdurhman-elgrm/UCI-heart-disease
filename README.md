# ❤️ UCI Heart Disease Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

An end-to-end machine learning project for predicting the presence of heart disease from clinical measurements, built on the classic **UCI Heart Disease** dataset for a Kaggle competition. It covers data ingestion, cleaning, exploratory analysis, model selection with `GridSearchCV`, threshold tuning, and a Streamlit app for live predictions.

Because missing a sick patient is far more costly than a false alarm, the project optimizes for **Recall (sensitivity)** and **ROC-AUC** rather than raw accuracy.

---

## 📌 Table of Contents

- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Notes and Limitations](#-notes-and-limitations)
- [License](#-license)

---

## 📊 Dataset

The data lives in `data/raw/heart_disease_uci.csv` and combines patient records from four institutions.

| Property | Value |
|---|---|
| Records | 920 patients |
| Sources | Cleveland (304), Hungary (293), VA Long Beach (200), Switzerland (123) |
| Task | Binary classification: `0` = no disease, `1` = disease |
| Class balance (after binarizing) | 509 disease / 411 healthy |

**Features**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age in years |
| `sex` | Categorical | `Male` / `Female` |
| `dataset` | Categorical | Source hospital |
| `cp` | Categorical | Chest pain type (`typical angina`, `atypical angina`, `non-anginal`, `asymptomatic`) |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) |
| `chol` | Numeric | Serum cholesterol (mg/dl) |
| `fbs` | Boolean | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting ECG (`normal`, `st-t abnormality`, `lv hypertrophy`) |
| `thalch` | Numeric | Maximum heart rate achieved |
| `exang` | Boolean | Exercise-induced angina |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of the peak exercise ST segment |
| `ca` | Numeric | Major vessels (0–3) colored by fluoroscopy |
| `thal` | Categorical | Thalassemia (`normal`, `fixed defect`, `reversable defect`) |
| `num` → `target` | Target | Original 0–4 diagnosis, binarized to `0` / `1` (any value > 0 is disease) |

---

## 🗂 Project Structure

```text
UCI-heart-disease/
├── data/
│   ├── raw/heart_disease_uci.csv        # original, untouched data
│   └── processed/cleaned_heart.csv      # cleaned, model-ready data
├── notebooks/
│   └── EDA.ipynb                        # exploratory data analysis
├── src/
│   ├── ingest.py                        # safe loading + schema validation
│   ├── clean.ipynb                      # cleaning and imputation
│   ├── feature.py                       # scaler / encoder ColumnTransformer
│   ├── train.py                         # GridSearchCV, evaluation, saving the model
│   └── predict.py                       # offline single-patient inference
├── models/
│   └── heart_model.pkl                  # saved pipeline + decision threshold
├── api/
│   └── main.py                          # early FastAPI prototype (see notes)
├── app.py                               # Streamlit web app
├── heart_disease_predictor_workflow.md  # detailed step-by-step workflow guide
├── scratch_test.py                      # quick inference smoke test
└── requirements.txt
```

---

## 🔬 Pipeline Overview

1. **Ingestion** (`src/ingest.py`): loads the CSV, normalizes column names, and validates that all expected columns exist.
2. **EDA** (`notebooks/EDA.ipynb`): target distribution, missing values per source hospital, and feature correlations with the target.
3. **Cleaning** (`src/clean.ipynb`):
   - Drop the `id` column.
   - Treat `chol == 0` and `trestbps == 0` as missing (biologically impossible).
   - Binarize `num` into `target`.
   - Impute numeric columns with the **median** and categorical columns with the **mode**.
   - Save to `data/processed/cleaned_heart.csv`.
4. **Preprocessing** (`ColumnTransformer` inside the model pipeline):
   - Numeric (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`): `StandardScaler`.
   - Categorical (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`, `dataset`): `OneHotEncoder(handle_unknown="ignore")`.
5. **Model selection** (`src/train.py`):
   - 80/20 stratified train/test split (`random_state=42`).
   - `GridSearchCV` with 5-fold cross-validation, scored on **Recall**, over three models: Logistic Regression, Random Forest, XGBoost.
   - The model with the best CV recall is selected automatically.
6. **Threshold tuning**: the decision threshold is searched between 0.10 and 0.50 to maximize a recall/precision-style score on the test set.
7. **Saving**: the fitted pipeline and chosen threshold are stored together as a dictionary in `models/heart_model.pkl`.

---

## 🏆 Results

### GridSearchCV (5-fold, scoring = Recall)

| Model | CV Recall | Best Hyperparameters |
|---|---|---|
| Logistic Regression | 0.8598 | `C=0.01`, `class_weight=None` |
| **Random Forest** ✅ | **0.8770** | `max_depth=5`, `n_estimators=50` |
| XGBoost | 0.8379 | `max_depth=3`, `n_estimators=50` |

**Best model: Random Forest (CV Recall: 0.8770)**

**Optimal decision threshold: 0.44** (F1: 0.9008)

### Detailed Evaluation: Random Forest (held-out test set, 184 patients)

| Metric | Score |
|---|---|
| Accuracy | 0.8478 |
| Precision | 0.8033 |
| **Recall (Sensitivity)** | **0.9608** |
| Specificity | 0.7073 |
| F1-Score | 0.8750 |
| ROC-AUC | 0.9342 |
| Log-Loss | 0.3749 |

### Confusion Matrix

|  | Predicted Disease | Predicted Healthy |
|---|---|---|
| **Actually Disease** | TP = 98 (sick caught) | FN = 4 (sick missed) |
| **Actually Healthy** | FP = 24 (healthy flagged) | TN = 58 (healthy correct) |

The model catches **98 of 102** patients with heart disease (96.1% recall), missing only 4. The trade-off is 24 false alarms among 82 healthy patients, which is the intended bias for a screening-style model where a missed diagnosis is the costlier error.

Raw output from `src/train.py`:

```text
GridSearchCV Results (5-fold, Recall):
----------------------------------------
Logistic Regression: 0.8598 | {'classifier__C': 0.01, 'classifier__class_weight': None}
Random Forest: 0.8770 | {'classifier__max_depth': 5, 'classifier__n_estimators': 50}
XGBoost: 0.8379 | {'classifier__max_depth': 3, 'classifier__n_estimators': 50}

Best model: Random Forest (CV Recall: 0.8770)

Optimal threshold: 0.44 (F1: 0.9008)

==================================================
DETAILED EVALUATION — Random Forest
==================================================
Accuracy:    0.8478
Precision:   0.8033
Recall:      0.9608
Specificity: 0.7073
F1-Score:    0.8750
ROC-AUC:     0.9342
Log-Loss:    0.3749

Confusion Matrix:
  TP (sick caught):     98
  TN (healthy correct): 58
  FP (healthy flagged): 24
  FN (sick missed):     4
==================================================
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Abdurhman-elgrm/UCI-heart-disease.git
cd UCI-heart-disease

# 2. (Optional) create a virtual environment
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

**Train the model** (re-runs the grid search and overwrites `models/heart_model.pkl`):

```bash
python src/train.py
```

**Run the Streamlit app** at `http://localhost:8501`:

```bash
streamlit run app.py
```

Enter the patient's clinical values and click **Predict** to get the disease probability and a label based on the tuned threshold.

**Predict from Python:**

```python
from src.predict import predict_single_patient

patient = {
    "age": 63, "sex": "Male", "dataset": "Cleveland", "cp": "typical angina",
    "trestbps": 145, "chol": 233, "fbs": True, "restecg": "lv hypertrophy",
    "thalch": 150, "exang": False, "oldpeak": 2.3, "slope": "downsloping",
    "ca": 0.0, "thal": "fixed defect",
}
print(predict_single_patient(patient))
```

---

## ⚠️ Notes and Limitations

- **Not a medical device.** This project is for education and competition purposes only and must not be used for real clinical decisions.
- The decision threshold is tuned on the same test set used for the final evaluation, so the reported test metrics are slightly optimistic. A separate validation split would give a cleaner estimate.
- Imputation in `clean.ipynb` is done on the full dataset before the train/test split, which can leak a small amount of information. Moving imputation into the pipeline would avoid this.
- `api/main.py` is an early FastAPI prototype that expects integer-encoded inputs and does not match the current string-based pipeline. Use `app.py` or `src/predict.py` instead.
- The notebooks contain absolute Windows paths to the author's machine and need adjusting before they will run elsewhere.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository, Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (Janosi, Steinbrunn, Pfisterer, Detrano).
- Built for a Kaggle competition.
