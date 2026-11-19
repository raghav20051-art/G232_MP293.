
from pathlib import Path

import pandas as pd


# Project root: G232_MP293
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_data():
    """Load train, later, and holdout datasets."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    later = pd.read_csv(DATA_DIR / "later.csv")
    holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")

    return train, later, holdout


def investigate_data():
    train, later, holdout = load_data()

    print("\n========== BASIC DATA INFORMATION ==========")
    print("Train shape:", train.shape)
    print("Later shape:", later.shape)
    print("Holdout shape:", holdout.shape)

    print("\n========== 1. LEAKING COLUMN CHECK ==========")
    missing_columns = set(train.columns) - set(holdout.columns)
    print("Columns missing from holdout:", missing_columns)
    print(
        "Feature columns missing from holdout:",
        missing_columns - {"withdrawal_thousand_rs"}
    )

    print("\n========== 2. UNIT MIX-UP CHECK ==========")
    print(
        train.groupby("area")["distance_to_market_km"]
        .describe()
        .round(2)
    )

    print("\n========== 3. SENTINEL VALUE CHECK ==========")
    sentinel_count = (train["avg_ticket_rs"] == -999).sum()
    print("avg_ticket_rs = -999 count:", sentinel_count)

    print("\n========== 4. DUPLICATE ROW CHECK ==========")
    duplicate_count = train.duplicated().sum()
    print("Duplicate rows:", duplicate_count)

    print("\n========== 5. STUCK SENSOR CHECK ==========")
    train["date"] = pd.to_datetime(train["date"])

    daily = train.groupby("date")["days_to_salary"].agg(
        ["min", "max", "nunique", "count"]
    )

    print("\nDaily readings:")
    print(daily.to_string())

    stuck_dates = daily[daily["nunique"] == 1]
    print("\nDates with one unique reading:")
    print(stuck_dates.to_string())

    print("\n========== 6. TARGET TYPO CHECK ==========")
    target = train["withdrawal_thousand_rs"]

    sorted_target = target.sort_values().reset_index(drop=True)
    gaps = sorted_target.diff()

    print("\nTarget statistics:")
    print(target.describe())

    print("\nLargest target values:")
    print(target.nlargest(30).to_string(index=False))

    print("\nLargest gaps between sorted target values:")
    print(gaps.nlargest(20).to_string())

    print("\nRecords with largest target values:")
    print(
        train.nlargest(30, "withdrawal_thousand_rs")[
            ["record_id", "date", "area", "atm_type",
             "withdrawal_thousand_rs"]
        ].to_string(index=False)
    )

    print("\nTarget statistics by area:")
    print(
        train.groupby("area")["withdrawal_thousand_rs"]
        .describe()
        .round(2)
    )

    print("\nTarget statistics by area and ATM type:")
    print(
        train.groupby(["area", "atm_type"])["withdrawal_thousand_rs"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )

    print("\n========== 7. SPELLING VARIANTS CHECK ==========")
    print("\nTrain ATM types:")
    print(train["atm_type"].value_counts(dropna=False).to_string())

    print("\nLater ATM types:")
    print(later["atm_type"].value_counts(dropna=False).to_string())

    print("\n========== ADDITIONAL DATA CHECKS ==========")
    print("Train missing values:")
    print(train.isna().sum().to_string())

    print("\nTrain duplicate record IDs:")
    print(train["record_id"].duplicated().sum())

    print("\n========== INVESTIGATION COMPLETE ==========")


if __name__ == "__main__":
    investigate_data()
