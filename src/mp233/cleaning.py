from pathlib import Path
import pandas as pd

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_FILE = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "train_clean.csv"
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "cleaning_summary.txt"


def clean_data(df):
    """Clean the training dataset and return the cleaned dataframe."""

    df = df.copy()

    summary = []

    summary.append("DATA CLEANING SUMMARY")
    summary.append("=" * 50)
    summary.append(f"Original rows: {len(df)}")

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = before - len(df)

    summary.append(f"1. Exact duplicate rows removed: {duplicates_removed}")

    # 2. Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Handle sentinel values in service_records
    sentinel_count = (df["service_records"] == -1).sum()

    df.loc[df["service_records"] == -1, "service_records"] = pd.NA

    summary.append(f"2. service_records (-1) converted to missing: {sentinel_count}")

    # 4. Handle suspicious stuck km_driven sequence
    stuck_mask = (
        (df["km_driven"] == 80060)
        & (df["date"] >= "2026-03-21")
        & (df["date"] <= "2026-04-12")
    )

    stuck_count = stuck_mask.sum()

    df.loc[stuck_mask, "km_driven"] = pd.NA

    summary.append(
        f"3. Suspicious km_driven values converted to missing: {stuck_count}"
    )

    # 5. Correct suspected 10x target typos
    suspicious_target_ids = [
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

    found_ids = set(df["record_id"].astype(str))
    missing_ids = set(suspicious_target_ids) - found_ids

    if missing_ids:
        raise ValueError(f"Target typo record IDs not found: {missing_ids}")

    target_mask = df["record_id"].isin(suspicious_target_ids)

    # Verify that these records contain unusually large targets
    if not (df.loc[target_mask, "resale_thousand_rs"] > 500).all():
        raise ValueError(
            "One or more suspected target typo values are not above 500. "
            "Review the records before correcting them."
        )

    corrected_count = target_mask.sum()

    df.loc[target_mask, "resale_thousand_rs"] = (
        df.loc[target_mask, "resale_thousand_rs"] / 10
    )

    summary.append(f"4. Suspected target typo values divided by 10: {corrected_count}")

    # 6. Remove potential leakage column from model dataset
    leakage_column = "final_price_from_rc_transfer"

    if leakage_column in df.columns:
        df = df.drop(columns=[leakage_column])
        summary.append("5. Potential leakage column removed: " + leakage_column)

    # 7. Report remaining missing values
    summary.append("\nRemaining missing values:")
    summary.append(df.isnull().sum().to_string())

    summary.append("\n" + "=" * 50)
    summary.append(f"Final rows: {len(df)}")
    summary.append(f"Final columns: {len(df.columns)}")

    return df, "\n".join(summary)


def main():
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Read original training dataset
    df = pd.read_csv(INPUT_FILE)

    # Clean dataset
    cleaned_df, summary = clean_data(df)

    # Save cleaned dataset
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    # Save cleaning report
    REPORT_FILE.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
    print(f"Cleaning report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
