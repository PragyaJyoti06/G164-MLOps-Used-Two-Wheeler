from pathlib import Path

import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient
from src.mp233.cleaning import clean_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cleaned_data():
    record_ids = [
        "R00040",
        "R00561",
        "R00615",
        "R00836",
        "R00977",
        "R00509",
        "R00543",
        "R00331",
        "R00772",
        "R00830",
        "R00904",
    ]

    df = pd.DataFrame({
        "record_id": record_ids,
        "date": "2026-03-25",
        "age_years": 5.0,
        "km_driven": 80060,
        "engine_cc": 150,
        "owners": 1,
        "service_records": -1,
        "accidents": 0,
        "brand_tier": "budget",
        "city_zone": "North",
        "final_price_from_rc_transfer": 80.0,
        "resale_thousand_rs": 800.0,
    })

    # Add one duplicate row for testing
    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True
    )

    cleaned, _ = clean_data(df)

    return cleaned


def test_cleaned_data_has_no_duplicates(cleaned_data):
    assert not cleaned_data.duplicated().any()
    assert len(cleaned_data) == 11


def test_target_column_exists_and_has_no_missing_values(cleaned_data):
    assert "resale_thousand_rs" in cleaned_data.columns
    assert cleaned_data["resale_thousand_rs"].notna().all()


def test_leakage_column_removed(cleaned_data):
    assert "final_price_from_rc_transfer" not in cleaned_data.columns


def test_target_typos_corrected(cleaned_data):
    assert (cleaned_data["resale_thousand_rs"] == 80.0).all()


def test_sentinel_values_handled(cleaned_data):
    assert cleaned_data["service_records"].isna().all()


def test_stuck_sensor_values_handled(cleaned_data):
    assert cleaned_data["km_driven"].isna().all()


def test_champion_model_registry():
    with open(ROOT / "reports" / "findings.yaml", encoding="utf-8") as file:
        tracking = yaml.safe_load(file)["tracking"]

    client = MlflowClient(
        tracking_uri=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    champion = client.get_model_version_by_alias(
        name=tracking["registered_model_name"],
        alias="champion"
    )

    assert int(champion.version) == int(tracking["champion_version"])

    v1 = client.get_model_version(
        name=tracking["registered_model_name"],
        version="1"
    )

    assert v1.run_id == tracking["best_run_id"]
