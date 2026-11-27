import joblib
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "age_years", "km_driven", "engine_cc", "owners",
    "service_records", "accidents", "brand_tier", "city_zone",
]

NUMERIC_FEATURES = [
    "age_years", "km_driven", "engine_cc",
    "owners", "service_records", "accidents",
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce")
    current = pd.to_numeric(current, errors="coerce")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(
        reference.dropna().quantile(quantiles).to_numpy()
    )

    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_bins = pd.cut(
        reference, bins=edges, labels=False
    ).astype("float").fillna(-1).astype(int)

    current_bins = pd.cut(
        current, bins=edges, labels=False
    ).astype("float").fillna(-1).astype(int)

    all_bins = sorted(
        set(reference_bins.unique()) |
        set(current_bins.unique())
    )

    reference_pct = (
        reference_bins.value_counts(normalize=True)
        .reindex(all_bins, fill_value=0)
    )
    current_pct = (
        current_bins.value_counts(normalize=True)
        .reindex(all_bins, fill_value=0)
    )

    epsilon = 0.000001

    psi = (
        (current_pct + epsilon) -
        (reference_pct + epsilon)
    ) * np.log(
        (current_pct + epsilon) /
        (reference_pct + epsilon)
    )

    return float(psi.sum())


def main():
    train = pd.read_csv("data/processed/train_clean.csv")
    later = pd.read_csv("data/raw/later.csv")

    v1_path = "models/model_v1_baseline.skops"
    trusted = sio.get_untrusted_types(file=v1_path)
    v1_model = sio.load(file=v1_path, trusted=trusted)

    v2_model = joblib.load("models/model.joblib")

    print("MP233 MODEL MONITORING")
    print("=" * 55)

    print("\n1. POPULATION STABILITY INDEX")
    psi_results = {}

    for feature in NUMERIC_FEATURES:
        value = calculate_psi(
            train[feature], later[feature], bins=10
        )
        psi_results[feature] = value
        print(f"{feature:20} {value:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)
    print("Most drifted feature:", most_drifted)

    print("\n2. V1 PERFORMANCE BY CITY ZONE")
    v1_predictions = v1_model.predict(later[FEATURES])

    results = later[["city_zone"]].copy()
    results["absolute_error"] = (
        later["resale_thousand_rs"].to_numpy() - v1_predictions
    )
    results["absolute_error"] = results["absolute_error"].abs()

    mae_by_group = (
        results.groupby("city_zone")["absolute_error"]
        .mean()
        .sort_values(ascending=False)
    )

    for group, value in mae_by_group.items():
        print(f"{group:20} {value:.6f}")

    print("Worst-served group:", mae_by_group.index[0])
    print(
        "Overall V1 MAE:",
        f"{mean_absolute_error(later['resale_thousand_rs'], v1_predictions):.6f}",
    )

    print("\n3. CHRONOLOGICAL V1 VS V2 EVALUATION")

    chronological = later.copy()
    chronological["date"] = pd.to_datetime(
        chronological["date"], errors="coerce"
    )
    chronological = chronological.sort_values(
        "date", kind="mergesort"
    ).reset_index(drop=True)

    split_index = int(len(chronological) * 0.70)
    check_data = chronological.iloc[split_index:]

    X_check = check_data[FEATURES]
    y_check = check_data["resale_thousand_rs"]

    v1_check_predictions = v1_model.predict(X_check)
    v2_check_predictions = v2_model.predict(X_check)

    v1_mae = mean_absolute_error(y_check, v1_check_predictions)
    v2_mae = mean_absolute_error(y_check, v2_check_predictions)

    improvement = (v1_mae - v2_mae) / v1_mae
    required_improvement = 0.08

    print("Training portion rows:", split_index)
    print("Check portion rows:", len(check_data))
    print("Check period:", check_data["date"].min(), "to", check_data["date"].max())
    print(f"V1 check MAE: {v1_mae:.6f}")
    print(f"V2 check MAE: {v2_mae:.6f}")
    print(f"MAE improvement: {improvement:.2%}")
    print(f"Required improvement: {required_improvement:.0%}")

    decision = (
        "PROMOTE V2"
        if improvement >= required_improvement
        else "RETAIN V1"
    )
    print("Promotion decision:", decision)


if __name__ == "__main__":
    main()
