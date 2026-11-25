
from pathlib import Path
import pandas as pd
import numpy as np
from difflib import get_close_matches

# Project folders
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

# Read the datasets. We do not modify the original files.
train = pd.read_csv(DATA / "train.csv")
later = pd.read_csv(DATA / "later.csv")
holdout = pd.read_csv(DATA / "holdout_features.csv")

output = []

def section(title):
    output.append("\n" + "=" * 65)
    output.append(title)
    output.append("=" * 65)

target = "resale_thousand_rs"

# -------------------------------------------------
# 1. DATASET INFORMATION
# -------------------------------------------------
section("DATASET INFORMATION")

output.append(f"Train rows: {len(train)}")
output.append(f"Train columns: {len(train.columns)}")
output.append(f"Later rows: {len(later)}")
output.append(f"Holdout rows: {len(holdout)}")
output.append("\nTrain column types:")
output.append(train.dtypes.to_string())

# -------------------------------------------------
# 2. LEAKING COLUMN
# -------------------------------------------------
section("1. LEAKING COLUMN CHECK")

missing_from_holdout = [
    col for col in train.columns if col not in holdout.columns
]

output.append(f"Columns missing from holdout: {missing_from_holdout}")

for col in missing_from_holdout:
    if col == target:
        continue

    if pd.api.types.is_numeric_dtype(train[col]):
        correlation = train[col].corr(train[target])
        output.append(
            f"{col}: correlation with target = {correlation:.4f}"
        )

# -------------------------------------------------
# 3. UNIT MIXUP
# -------------------------------------------------
section("2. UNIT MIXUP CHECK")

if "city_zone" in train.columns:
    numeric_cols = train.select_dtypes(include="number").columns
    numeric_cols = [
        c for c in numeric_cols if c != "record_id"
    ]

    for col in numeric_cols:
        stats = train.groupby("city_zone")[col].agg(
            ["min", "mean", "max", "median"]
        )
        output.append(f"\nColumn: {col}")
        output.append(stats.to_string())

        # Flag possible factors of 12 or 1000 for manual checking.
        medians = stats["median"].dropna()
        medians = medians[medians > 0]

        if len(medians) >= 2:
            ratio = medians.max() / medians.min()

            for factor in (12, 1000):
                if abs(np.log10(ratio / factor)) < 0.08:
                    output.append(
                        f"POSSIBLE UNIT MIXUP: {col}, "
                        f"group median ratio approximately {ratio:.2f}"
                    )

# -------------------------------------------------
# 4. SENTINEL VALUES
# -------------------------------------------------
section("3. SENTINEL VALUE CHECK")

sentinels = [-1, -999, 999, 9999]

for col in train.select_dtypes(include="number").columns:
    for value in sentinels:
        count = (train[col] == value).sum()

        if count > 0:
            output.append(
                f"{col}: value {value} occurs {count} times"
            )

# -------------------------------------------------
# 5. DUPLICATE ROWS
# -------------------------------------------------
section("4. DUPLICATE ROW CHECK")

duplicate_count = train.duplicated().sum()
output.append(f"Exact duplicate rows: {duplicate_count}")

# -------------------------------------------------
# 6. STUCK SENSOR
# -------------------------------------------------
section("5. STUCK SENSOR CHECK")

ordered = train.copy()

if "date" in ordered.columns:
    ordered["date"] = pd.to_datetime(
        ordered["date"], errors="coerce"
    )
    ordered = ordered.sort_values("date")

excluded = {
    "record_id", "date", target,
    "final_price_from_rc_transfer"
}

for col in ordered.columns:
    if col in excluded:
        continue

    values = ordered[col].astype("string").fillna("<MISSING>")
    run_ids = values.ne(values.shift()).cumsum()
    run_sizes = values.groupby(run_ids).size()

    if len(run_sizes) == 0:
        continue

    longest_id = run_sizes.idxmax()
    longest_length = run_sizes.max()

    if longest_length >= 5:
        run_rows = ordered.loc[run_ids == longest_id]

        first_date = (
            run_rows["date"].iloc[0]
            if "date" in run_rows.columns else "unknown"
        )
        last_date = (
            run_rows["date"].iloc[-1]
            if "date" in run_rows.columns else "unknown"
        )

        output.append(
            f"{col}: value {values.loc[run_rows.index[0]]!r} "
            f"repeats for {longest_length} consecutive rows; "
            f"from {first_date} to {last_date}"
        )

# -------------------------------------------------
# 7. TARGET TYPOS
# -------------------------------------------------
section("6. TARGET TYPO CHECK")

if target in train.columns:
    y = pd.to_numeric(train[target], errors="coerce")
    median_target = y[y > 0].median()

    suspicious = y >= median_target * 10

    output.append(f"Target median: {median_target}")
    output.append("Possible 10x target values:")

    columns_to_show = [target]
    if "record_id" in train.columns:
        columns_to_show.insert(0, "record_id")

    output.append(
        train.loc[suspicious, columns_to_show].to_string(index=False)
    )

# -------------------------------------------------
# 8. SPELLING VARIANTS
# -------------------------------------------------
section("7. SPELLING VARIANT CHECK")

for col in ["brand_tier", "city_zone"]:
    if col not in train.columns or col not in later.columns:
        continue

    train_values = train[col].dropna().astype(str).unique().tolist()
    clean_values = later[col].dropna().astype(str).unique().tolist()

    output.append(f"\nColumn: {col}")
    output.append(f"Train value counts:\n{train[col].value_counts().to_string()}")
    output.append(f"Later value counts:\n{later[col].value_counts().to_string()}")
    output.append(f"True levels in later.csv: {later[col].nunique()}")

    for value in train_values:
        matches = get_close_matches(
            value, clean_values, n=1, cutoff=0.80
        )

        if matches and value != matches[0]:
            output.append(
                f"Possible spelling variant: {value!r} "
                f"may correspond to {matches[0]!r}"
            )

# -------------------------------------------------
# SAVE THE REPORT
# -------------------------------------------------
REPORTS.mkdir(parents=True, exist_ok=True)

report_path = REPORTS / "data_quality_report.txt"
report_path.write_text("\n".join(output), encoding="utf-8")

print("\n".join(output))
print(f"\nReport saved to: {report_path}")
