
import pandas as pd
import pytest

from src.mp293.preprocessing import (
    clean_dataset,
    fit_preprocessor,
    transform_dataset,
    preprocess_all,
)


def sample_data(include_target=True):
    """Create a small sample dataset for testing."""

    data = {
        "record_id": [1, 2, 3],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "nearby_shops": [10, 20, 30],
        "days_to_salary": [5, 10, 22],
        "footfall_index": [40, 50, 60],
        "distance_to_market_km": [2000, 3.0, 4.0],
        "avg_ticket_rs": [1000, -999, 3000],
        "other_atms_1km": [2, 3, 4],
        "atm_type": ["Bank_Branch", "mall", "sta."],
        "area": ["Highway", "Market", "Residential"],
    }

    if include_target:
        data["withdrawal_thousand_rs"] = [100, 200, 300]

    return pd.DataFrame(data)


def test_highway_distance_conversion():
    df = sample_data()

    cleaned = clean_dataset(df, "train")

    assert cleaned.loc[0, "distance_to_market_km"] == 2.0


def test_sentinel_value_becomes_missing():
    df = sample_data()

    cleaned = clean_dataset(df, "train")

    assert pd.isna(cleaned.loc[1, "avg_ticket_rs"])


def test_atm_type_normalization():
    df = sample_data()

    cleaned = clean_dataset(df, "train")

    assert cleaned["atm_type"].tolist() == [
        "bank_branch",
        "mall",
        "standalone",
    ]


def test_invalid_atm_type_raises_error():
    df = sample_data()
    df.loc[0, "atm_type"] = "unknown_type"

    with pytest.raises(ValueError):
        clean_dataset(df, "train")


def test_original_data_is_not_modified():
    df = sample_data()
    original = df.copy(deep=True)

    clean_dataset(df, "train")

    pd.testing.assert_frame_equal(df, original)


def test_training_medians_are_used_for_imputation():
    df = sample_data()
    cleaned = clean_dataset(df, "train")

    preprocessor = fit_preprocessor(cleaned)

    transformed = transform_dataset(cleaned, preprocessor)

    assert transformed["avg_ticket_rs"].isna().sum() == 0
    assert transformed.loc[1, "avg_ticket_rs"] == 2000


def test_all_datasets_have_same_feature_columns():
    train = sample_data()
    later = sample_data()
    holdout = sample_data(include_target=False)

    result = preprocess_all(train, later, holdout)

    assert list(result["X_train"].columns) == list(
        result["X_later"].columns
    )

    assert list(result["X_train"].columns) == list(
        result["X_holdout"].columns
    )


def test_preprocessed_datasets_have_no_missing_values():
    train = sample_data()
    later = sample_data()
    holdout = sample_data(include_target=False)

    result = preprocess_all(train, later, holdout)

    assert result["X_train"].isna().sum().sum() == 0
    assert result["X_later"].isna().sum().sum() == 0
    assert result["X_holdout"].isna().sum().sum() == 0
