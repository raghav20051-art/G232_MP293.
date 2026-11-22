
"""Preprocessing utilities for ATM cash demand prediction."""

import pandas as pd

from src.mp293.config import load_config


CONFIG = load_config()
PREPROCESSING_CONFIG = CONFIG["preprocessing"]

TARGET = PREPROCESSING_CONFIG["target"]

NUMERIC_FEATURES = PREPROCESSING_CONFIG["numeric_features"]
CATEGORICAL_FEATURES = PREPROCESSING_CONFIG["categorical_features"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SENTINEL_VALUE = PREPROCESSING_CONFIG["sentinel_value"]
HIGHWAY_DISTANCE_DIVISOR = PREPROCESSING_CONFIG[
    "highway_distance_divisor"
]

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
    """Clean a dataset without changing the original DataFrame."""

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

    # Replace invalid sentinel values.
    cleaned["avg_ticket_rs"] = cleaned["avg_ticket_rs"].replace(
        SENTINEL_VALUE, float("nan")
    )

    # Normalize ATM type values.
    cleaned["atm_type"] = cleaned["atm_type"].apply(
        normalize_atm_type
    )

    # Normalize area text.
    cleaned["area"] = (
        cleaned["area"].astype("string").str.strip().str.lower()
    )

    # Correct the training-only Highway distance unit issue.
    if dataset_name.lower() == "train":
        highway_mask = cleaned["area"] == "highway"

        cleaned.loc[
            highway_mask, "distance_to_market_km"
        ] = (
            cleaned.loc[
                highway_mask, "distance_to_market_km"
            ] / HIGHWAY_DISTANCE_DIVISOR
        )

    # Convert numeric features.
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        )

    # Validate target values.
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
    """Learn imputation values and categories from training data only."""

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

    # One-hot encoding using training categories.
    for column, categories in preprocessor[
        "categorical_categories"
    ].items():

        values = transformed[column].fillna("unknown")

        for category in categories:
            transformed[f"{column}_{category}"] = (
                values == category
            ).astype(int)

        transformed = transformed.drop(columns=[column])

    return transformed


def preprocess_all(train_df, later_df, holdout_df):
    """Clean and transform all datasets consistently."""

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
