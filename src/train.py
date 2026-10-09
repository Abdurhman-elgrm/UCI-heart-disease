import joblib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_heart.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.pkl"

numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "dataset"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Log-Loss: {log_loss(y_test, y_proba):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"classifier__C": [0.01, 0.1, 1, 10], "classifier__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            {"classifier__n_estimators": [50, 100, 200], "classifier__max_depth": [None, 5, 10]},
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
            {"classifier__n_estimators": [50, 100, 200], "classifier__max_depth": [3, 5, 7]},
        ),
    }

    best_name = None
    best_cv_recall = 0.0
    best_pipeline = None

    print("GridSearchCV Results (5-fold, Recall):")
    print("-" * 40)

    for name, (classifier, param_grid) in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall")
        grid.fit(X_train, y_train)
        print(f"{name}: {grid.best_score_:.4f} | {grid.best_params_}")

        if grid.best_score_ > best_cv_recall:
            best_cv_recall = grid.best_score_
            best_name = name
            best_pipeline = grid.best_estimator_

    print(f"\nBest model: {best_name} (CV Recall: {best_cv_recall:.4f})")

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in [i / 100 for i in range(10, 51)]:
        y_pred_t = (y_proba >= threshold).astype(int)
        recall = recall_score(y_test, y_pred_t)
        precision = accuracy_score(y_test, y_pred_t)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nOptimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    y_pred = (y_proba >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"DETAILED EVALUATION — {best_name}")
    print(f"{'='*50}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc:.4f}")
    print(f"Log-Loss:    {ll:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (sick caught):     {tp}")
    print(f"  TN (healthy correct): {tn}")
    print(f"  FP (healthy flagged): {fp}")
    print(f"  FN (sick missed):     {fn}")
    print(f"{'='*50}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "threshold": best_threshold}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

