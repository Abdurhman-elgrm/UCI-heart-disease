import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"

EXPECTED_COLUMNS = [
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]


def load_raw_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found at {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    print(df.head())
