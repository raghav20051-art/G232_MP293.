
from pathlib import Path

import pandas as pd


# Project directory: .../G232_MP293
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TARGET_COLUMN = "withdrawal_thousand_rs"

FEATURE_COLUMNS = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
    "atm_type",
    "area",
]

METADATA_COLUMNS = ["record_id", "date"]

TRAIN_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
HOLDOUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def load_csv(file_path):
    """Load a CSV file and raise a clear error if it cannot be read."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        raise ValueError(
            f"Could not read {file_path.name}: {error}"
        ) from error


def validate_columns(df, required_columns, dataset_name):
    """Check whether all required columns exist."""
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )


def validate_record_ids(df, dataset_name):
    """Check that record IDs are present and unique."""
    if df["record_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing record_id values."
        )

    duplicate_count = df["record_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} "
            "duplicate record_id values."
        )


def load_datasets(data_dir=DATA_DIR):
    """
    Load and validate train, later, and holdout datasets.

    Returns:
        A dictionary containing:
        - train
        - later
        - holdout
    """
    data_dir = Path(data_dir)

    train = load_csv(data_dir / "train.csv")
    later = load_csv(data_dir / "later.csv")
    holdout = load_csv(data_dir / "holdout_features.csv")

    # Validate expected columns.
    validate_columns(train, TRAIN_COLUMNS, "Train dataset")
    validate_columns(later, TRAIN_COLUMNS, "Later dataset")
    validate_columns(holdout, HOLDOUT_COLUMNS, "Holdout dataset")

    # Validate record IDs.
    validate_record_ids(train, "Train dataset")
    validate_record_ids(later, "Later dataset")
    validate_record_ids(holdout, "Holdout dataset")

    # The target must be available in train and later,
    # but must not be present in holdout.
    if train[TARGET_COLUMN].isna().any():
        raise ValueError("Train dataset contains missing target values.")

    if later[TARGET_COLUMN].isna().any():
        raise ValueError("Later dataset contains missing target values.")

    if TARGET_COLUMN in holdout.columns:
        raise ValueError(
            "Holdout dataset should not contain the target column."
        )

    return {
        "train": train,
        "later": later,
        "holdout": holdout,
    }


def print_dataset_summary(datasets):
    """Print a short summary of the loaded datasets."""
    for name, df in datasets.items():
        print(f"\n{name.upper()} DATASET")
        print("-" * 35)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Target available: {TARGET_COLUMN in df.columns}")


if __name__ == "__main__":
    loaded_datasets = load_datasets()
    print_dataset_summary(loaded_datasets)
    print("\nAll datasets loaded and validated successfully.")
