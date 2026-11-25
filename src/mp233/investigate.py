
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Read original training data
df = pd.read_csv(ROOT / "data/raw/train.csv")

# Create reports folder
report_dir = ROOT / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

output_file = report_dir / "investigation_output.txt"

with open(output_file, "w", encoding="utf-8") as f:

    # Investigate suspicious target values
    ids = ["R00040", "R00561", "R00615", "R00836", "R00977"]

    f.write("--- Suspicious target records ---\n")
    f.write(
        df[df["record_id"].isin(ids)].to_string(index=False)
    )
    f.write("\n\n")

    # Investigate km_driven stuck sequence
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    mask = df["km_driven"] == 80060

    f.write("--- km_driven = 80060 records ---\n")
    f.write(df.loc[mask].to_string(index=False))
    f.write("\n\n")

    # Investigate duplicates
    f.write("--- Duplicate rows ---\n")
    f.write(df[df.duplicated(keep=False)].to_string(index=False))
    f.write("\n")

print(f"Investigation output saved to: {output_file}")
