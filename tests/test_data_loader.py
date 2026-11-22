
from pathlib import Path

import pandas as pd
import pytest

from src.mp293.data_loader import (
    load_csv,
    validate_columns,
    validate_record_ids,
    load_datasets,
    TARGET_COLUMN,
)


def create_valid_datasets(directory):
    """Create small valid datasets for testing."""
    directory.mkdir(parents=True, exist_ok=True)

    common_data = {
        "record_id": ["ATM001", "ATM002"],
        "date": ["2026-01-01", "2026-01-02"],
        "nearby_shops": [10, 20],
        "days_to_salary": [5, 10],
        "footfall_index": [50, 60],
        "distance_to_market_km": [2.0, 3.0],
        "avg_ticket_rs": [2000, 2500],
        "other_atms_1km": [3, 4],
        "atm_type": ["mall", "standalone"],
        "area": ["Market", "Highway"],
    }

    train = pd.DataFrame(common_data)
    train[TARGET_COLUMN] = [100, 150]

    later = pd.DataFrame(common_data)
    later[TARGET_COLUMN] = [120, 160]

    holdout = pd.DataFrame(common_data)

    train.to_csv(directory / "train.csv", index=False)
    later.to_csv(directory / "later.csv", index=False)
    holdout.to_csv(directory / "holdout_features.csv", index=False)


def test_load_datasets_successfully(tmp_path):
    create_valid_datasets(tmp_path)

    datasets = load_datasets(tmp_path)

    assert set(datasets.keys()) == {"train", "later", "holdout"}
    assert len(datasets["train"]) == 2
    assert len(datasets["later"]) == 2
    assert len(datasets["holdout"]) == 2
    assert TARGET_COLUMN in datasets["train"].columns
    assert TARGET_COLUMN not in datasets["holdout"].columns


def test_missing_dataset_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_missing_required_columns_raises_error():
    df = pd.DataFrame({"record_id": ["ATM001"]})

    with pytest.raises(ValueError, match="missing columns"):
        validate_columns(
            df,
            ["record_id", "date", "area"],
            "Test dataset",
        )


def test_duplicate_record_ids_raise_error():
    df = pd.DataFrame({
        "record_id": ["ATM001", "ATM001"]
    })

    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_record_ids(df, "Test dataset")


def test_missing_record_id_raises_error():
    df = pd.DataFrame({
        "record_id": ["ATM001", None]
    })

    with pytest.raises(ValueError, match="missing record_id"):
        validate_record_ids(df, "Test dataset")


def test_missing_target_values_raise_error(tmp_path):
    create_valid_datasets(tmp_path)

    train = pd.read_csv(tmp_path / "train.csv")
    train.loc[0, TARGET_COLUMN] = None
    train.to_csv(tmp_path / "train.csv", index=False)

    with pytest.raises(ValueError, match="missing target"):
        load_datasets(tmp_path)


def test_holdout_target_column_is_rejected(tmp_path):
    create_valid_datasets(tmp_path)

    holdout = pd.read_csv(tmp_path / "holdout_features.csv")
    holdout[TARGET_COLUMN] = [100, 150]
    holdout.to_csv(tmp_path / "holdout_features.csv", index=False)

    with pytest.raises(ValueError, match="should not contain"):
        load_datasets(tmp_path)
