
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml", encoding="utf-8") as file:
    PARAMS = yaml.safe_load(file)

FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]

NUMERIC_FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
]

CATEGORICAL_FEATURES = ["brand_tier", "city_zone"]


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_model():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=PARAMS["model"]["n_estimators"],
                    max_depth=PARAMS["model"]["max_depth"],
                    random_state=PARAMS["training"]["random_state"],
                    n_jobs=PARAMS["model"]["n_jobs"],
                ),
            ),
        ]
    )


def main():
    later_path = ROOT / PARAMS["data"]["later_path"]
    raw_path = ROOT / PARAMS["data"]["raw_path"]

    later = pd.read_csv(later_path)
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    split = int(len(later) * PARAMS["monitoring"]["retrain_fraction"])

    train_data = later.iloc[:split].copy()
    check_data = later.iloc[split:].copy()

    model = build_model()

    model.fit(
        train_data[FEATURES],
        train_data[PARAMS["data"]["target"]],
    )

    predictions = model.predict(check_data[FEATURES])

    v2_mae = mean_absolute_error(
        check_data[PARAMS["data"]["target"]],
        predictions,
    )

    output_dir = ROOT / "models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "model_v2_candidate.joblib"
    joblib.dump(model, model_path)

    data_hash = calculate_sha256(raw_path)

    print("\nV2 RETRAINING RESULTS")
    print("=" * 50)
    print("Training rows:", len(train_data))
    print("Check rows:", len(check_data))
    print("Training period:", train_data["date"].min(), "to", train_data["date"].max())
    print("Check period:", check_data["date"].min(), "to", check_data["date"].max())
    print("V2 MAE:", round(v2_mae, 6))
    print("Original train SHA-256:", data_hash)
    print("Candidate model saved:", model_path)


if __name__ == "__main__":
    main()
