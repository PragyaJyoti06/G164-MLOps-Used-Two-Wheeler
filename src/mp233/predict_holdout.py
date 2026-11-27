from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
OUTPUT_PATH = ROOT / "predictions" / "holdout_predictions.csv"

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

def main():
    holdout = pd.read_csv(HOLDOUT_PATH)

    required = ["record_id", "date"] + FEATURES
    missing = [column for column in required if column not in holdout.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(holdout[FEATURES])

    output = holdout[["record_id", "date"]].copy()
    output["predicted_resale_thousand_rs"] = predictions

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("HOLDOUT PREDICTION SUMMARY")
    print("=" * 40)
    print(f"Input rows: {len(holdout)}")
    print(f"Predictions generated: {len(output)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nFirst 5 predictions:")
    print(output.head().to_string(index=False))
    print("\nMissing predictions:", output["predicted_resale_thousand_rs"].isna().sum())

if __name__ == "__main__":
    main()
