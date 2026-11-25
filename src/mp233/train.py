
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "data" / "raw" / "train.csv"
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP233_Used_Two_Wheeler"
MODEL_NAME = "mp233_used_two_wheeler"


# --------------------------------------------------
# 2. Calculate original dataset SHA-256
# --------------------------------------------------

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 3. Load cleaned dataset
# --------------------------------------------------

def main():

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEAN_DATA_PATH}")

    data_hash = calculate_sha256(RAW_DATA_PATH)

    df = pd.read_csv(CLEAN_DATA_PATH)

    features = [
        "age_years",
        "km_driven",
        "engine_cc",
        "owners",
        "service_records",
        "accidents",
        "brand_tier",
        "city_zone",
    ]

    target = "resale_thousand_rs"

    X = df[features]
    y = df[target]

    # --------------------------------------------------
    # 4. Train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------
    # 5. Preprocessing
    # --------------------------------------------------

    numeric_features = [
        "age_years",
        "km_driven",
        "engine_cc",
        "owners",
        "service_records",
        "accidents",
    ]

    categorical_features = [
        "brand_tier",
        "city_zone",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # --------------------------------------------------
    # 6. Random Forest model
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    # --------------------------------------------------
    # 7. Configure MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --------------------------------------------------
    # 8. Train and log experiment
    # --------------------------------------------------

    with mlflow.start_run(run_name="baseline_v1"):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
            "test_size": 0.20,
            "imputation": "median_numeric_most_frequent_categorical",
        })

        # Log metrics
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # Log original raw dataset hash
        mlflow.log_param("data_sha256", data_hash)

        # Prepare input example
        input_example = X_test.head(5).copy()

        # Convert numeric columns to float64
        # This avoids MLflow input schema warnings
        input_example[numeric_features] = (
            input_example[numeric_features].astype("float64")
        )

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print("\nMODEL TRAINING COMPLETED")
        print("------------------------")
        print("Training rows:", len(X_train))
        print("Testing rows:", len(X_test))
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))
        print("Data SHA-256:", data_hash)
        print("Registered model:", MODEL_NAME)


if __name__ == "__main__":
    main()
