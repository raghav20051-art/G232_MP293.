
"""Preprocessing utilities for ATM cash demand prediction."""

from pathlib import Path

import pandas as pd


TARGET = "withdrawal_thousand_rs"

NUMERIC_FEATURES = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
]

CATEGORICAL_FEATURES = ["atm_type", "area"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ATM_TYPE_MAP = {
    "bank_branch": "bank_branch",
    "bank branch": "bank_branch",
    "ban.": "bank_branch",
    "mall": "mall",
    "mal.": "mall",
    "standalone": "standalone",
    "sta.": "standalone",
}


def normalize_atm_type(value):
    """Normalize ATM type spelling and capitalization."""

    if pd.isna(value):
        return pd.NA

    cleaned = str(value).strip().lower()

    if cleaned not in ATM_TYPE_MAP:
        raise ValueError(f"Unknown ATM type: {value!r}")

    return ATM_TYPE_MAP[cleaned]


def clean_dataset(df, dataset_name="dataset"):
    """
    Clean a dataset without changing the original DataFrame.

    The training dataset is expected to contain Highway distances
    in metres. Later and Holdout distances are already in kilometres.
    """

    cleaned = df.copy()

    required_columns = FEATURES.copy()

    if dataset_name.lower() != "holdout":
        required_columns.append(TARGET)

    missing_columns = [
        column
        for column in required_columns
        if column not in cleaned.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    # Convert invalid sentinel values into missing values.
    cleaned["avg_ticket_rs"] = cleaned["avg_ticket_rs"].replace(
        -999, float("nan")
    )

    # Normalize ATM type values.
    cleaned["atm_type"] = cleaned["atm_type"].apply(
        normalize_atm_type
    )

    # Normalize area text.
    cleaned["area"] = cleaned["area"].astype("string").str.strip().str.lower()

    # Correct the known training-only Highway distance unit issue.
    if dataset_name.lower() == "train":
        highway_mask = cleaned["area"] == "highway"

        cleaned.loc[
            highway_mask, "distance_to_market_km"
        ] = (
            cleaned.loc[
                highway_mask, "distance_to_market_km"
            ] / 1000
        )

    # Convert numeric features to numeric types.
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        )

    # Validate target values when target is present.
    if TARGET in cleaned.columns:
        cleaned[TARGET] = pd.to_numeric(
            cleaned[TARGET], errors="coerce"
        )

        if cleaned[TARGET].isna().any():
            raise ValueError(
                f"{dataset_name} contains missing or invalid target values."
            )

    return cleaned


def fit_preprocessor(train_df):
    """
    Learn numeric imputation values and categorical categories
    from training data only.
    """

    numeric_medians = train_df[NUMERIC_FEATURES].median()

    if numeric_medians.isna().any():
        invalid = numeric_medians[numeric_medians.isna()].index.tolist()
        raise ValueError(
            f"Cannot calculate training medians for: {invalid}"
        )

    categorical_categories = {}

    for column in CATEGORICAL_FEATURES:
        categories = sorted(
            train_df[column].dropna().unique().tolist()
        )

        if not categories:
            raise ValueError(
                f"No categories found for {column} in training data."
            )

        categorical_categories[column] = categories

    return {
        "numeric_medians": numeric_medians.to_dict(),
        "categorical_categories": categorical_categories,
    }


def transform_dataset(df, preprocessor):
    """Apply training-fitted preprocessing to another dataset."""

    transformed = df[FEATURES].copy()

    # Numeric imputation using training medians.
    for column, median in preprocessor["numeric_medians"].items():
        transformed[column] = transformed[column].fillna(median)

    # One-hot encoding using categories learned from training.
    for column, categories in preprocessor[
        "categorical_categories"
    ].items():

        values = transformed[column].fillna("unknown")

        for category in categories:
            transformed[f"{column}_{category}"] = (
                values == category
            ).astype(int)

        # Unseen or missing categories are represented by all-zero
        # encoded columns for that feature.
        transformed = transformed.drop(columns=[column])

    return transformed


def preprocess_all(train_df, later_df, holdout_df):
    """Clean, fit and transform all datasets consistently."""

    train_clean = clean_dataset(train_df, "train")
    later_clean = clean_dataset(later_df, "later")
    holdout_clean = clean_dataset(holdout_df, "holdout")

    preprocessor = fit_preprocessor(train_clean)

    X_train = transform_dataset(train_clean, preprocessor)
    X_later = transform_dataset(later_clean, preprocessor)
    X_holdout = transform_dataset(holdout_clean, preprocessor)

    y_train = train_clean[TARGET].copy()
    y_later = later_clean[TARGET].copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_later": X_later,
        "y_later": y_later,
        "X_holdout": X_holdout,
        "preprocessor": preprocessor,
        "train_clean": train_clean,
        "later_clean": later_clean,
        "holdout_clean": holdout_clean,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]

    train = pd.read_csv(base_dir / "data" / "train.csv")
    later = pd.read_csv(base_dir / "data" / "later.csv")
    holdout = pd.read_csv(
        base_dir / "data" / "holdout_features.csv"
    )

    result = preprocess_all(train, later, holdout)

    print("PREPROCESSING COMPLETED")
    print("-" * 45)

    for name in ["X_train", "X_later", "X_holdout"]:
        print(f"{name}: {result[name].shape}")

    print("\nMissing values after preprocessing:")
    for name in ["X_train", "X_later", "X_holdout"]:
        print(name, result[name].isna().sum().sum())

    print("\nTraining target shape:", result["y_train"].shape)
    print("Later target shape:", result["y_later"].shape)

    print("\nLearned training medians:")
    print(result["preprocessor"]["numeric_medians"])
