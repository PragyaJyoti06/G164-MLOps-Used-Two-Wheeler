from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml


ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "params.yaml", encoding="utf-8") as file:
    PARAMS = yaml.safe_load(file)

MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]
MODEL_PATH = ROOT / "models" / "model.joblib"


def main():

    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    print("Champion version:", champion.version)
    print("Champion run ID:", champion.run_id)

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)

    print("Model exported successfully")
    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
