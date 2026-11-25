
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"


def test_cleaned_data_has_no_duplicates():
    df = pd.read_csv(CLEAN_DATA_PATH)
    assert not df.duplicated().any()


def test_target_column_exists_and_has_no_missing_values():
    df = pd.read_csv(CLEAN_DATA_PATH)

    assert "resale_thousand_rs" in df.columns
    assert df["resale_thousand_rs"].notna().all()


def test_champion_model_is_version_one():
    client = MlflowClient(
        tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    champion = client.get_model_version_by_alias(
        name="mp233_used_two_wheeler",
        alias="champion"
    )

    assert champion.version == 1
