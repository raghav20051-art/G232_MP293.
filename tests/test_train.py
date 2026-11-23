"""Tests for Stage 2 model training and MLflow integration."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.mp293.train import (
    calculate_sha256,
    evaluate_model,
    get_model_configs,
    load_training_data,
    predict_with_model,
    register_best_model,
    run_experiments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_training_data(tmp_path):
    """Create a small processed training CSV for testing."""
    np.random.seed(42)
    n = 30

    data = {
        "record_id": [f"ATM{i:04d}" for i in range(1, n + 1)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "nearby_shops": np.random.randint(1, 25, n),
        "days_to_salary": np.random.randint(0, 30, n),
        "footfall_index": np.random.uniform(10, 100, n).round(1),
        "distance_to_market_km": np.random.uniform(0.5, 15, n).round(2),
        "avg_ticket_rs": np.random.uniform(300, 5000, n).round(0),
        "other_atms_1km": np.random.randint(0, 20, n),
        "atm_type_bank_branch": np.random.randint(0, 2, n),
        "atm_type_mall": np.random.randint(0, 2, n),
        "atm_type_standalone": np.random.randint(0, 2, n),
        "area_highway": np.random.randint(0, 2, n),
        "area_market": np.random.randint(0, 2, n),
        "area_residential": np.random.randint(0, 2, n),
    }
    data["withdrawal_thousand_rs"] = (
        50 + data["nearby_shops"] * 2 + np.random.normal(0, 10, n)
    ).round(1)

    df = pd.DataFrame(data)
    path = tmp_path / "train_processed.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_raw_train(tmp_path):
    """Create a small raw train CSV for SHA-256 testing."""
    path = tmp_path / "train.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    return path


@pytest.fixture
def mlflow_tmp_db(tmp_path):
    """Return a temporary MLflow SQLite tracking URI."""
    db_path = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def fast_model_configs():
    """Ultra-fast model configs for testing."""
    return [
        (
            "Fast_Tree_1",
            DecisionTreeRegressor(max_depth=3, random_state=42),
            {"model_type": "DecisionTreeRegressor", "max_depth": 3},
        ),
        (
            "Fast_Tree_2",
            DecisionTreeRegressor(max_depth=5, random_state=42),
            {"model_type": "DecisionTreeRegressor", "max_depth": 5},
        ),
        (
            "Fast_Forest_1",
            RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42),
            {"model_type": "RandomForestRegressor", "n_estimators": 3, "max_depth": 3},
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSHA256:
    def test_sha256_produces_hex_string(self, sample_raw_train):
        h = calculate_sha256(sample_raw_train)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_sha256_is_deterministic(self, sample_raw_train):
        h1 = calculate_sha256(sample_raw_train)
        h2 = calculate_sha256(sample_raw_train)
        assert h1 == h2

    def test_sha256_matches_hashlib(self, sample_raw_train):
        expected = hashlib.sha256(sample_raw_train.read_bytes()).hexdigest()
        actual = calculate_sha256(sample_raw_train)
        assert actual == expected


class TestDataLoading:
    def test_load_training_data_shapes(self, sample_training_data):
        X, y, cols = load_training_data(sample_training_data)
        assert len(X) == 30
        assert len(y) == 30
        assert "record_id" not in cols
        assert "date" not in cols
        assert "withdrawal_thousand_rs" not in cols

    def test_target_not_in_features(self, sample_training_data):
        X, y, cols = load_training_data(sample_training_data)
        assert "withdrawal_thousand_rs" not in X.columns


class TestModelEvaluation:
    def test_evaluate_returns_metrics(self, sample_training_data):
        X, y, _ = load_training_data(sample_training_data)
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        metrics = evaluate_model(model, X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


class TestModelConfigs:
    def test_at_least_three_configs(self):
        configs = get_model_configs()
        assert len(configs) >= 3

    def test_configs_have_expected_structure(self):
        configs = get_model_configs()
        for name, model, params in configs:
            assert isinstance(name, str)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert isinstance(params, dict)
            assert "model_type" in params


class TestMLflowExperiments:
    def test_run_experiments_logs_runs(
        self, sample_training_data, sample_raw_train, mlflow_tmp_db
    ):
        X, y, _ = load_training_data(sample_training_data)
        data_hash = calculate_sha256(sample_raw_train)

        with patch("src.mp293.train.get_model_configs", side_effect=fast_model_configs):
            results = run_experiments(
                X=X,
                y=y,
                data_hash=data_hash,
                tracking_uri=mlflow_tmp_db,
                experiment_name="test_experiment",
                raw_train_path=sample_raw_train,
            )

        assert len(results) >= 3

        # Verify runs in MLflow
        mlflow.set_tracking_uri(mlflow_tmp_db)
        client = MlflowClient()
        experiment = client.get_experiment_by_name("test_experiment")
        runs = client.search_runs([experiment.experiment_id])

        assert len(runs) >= 3

        for run in runs:
            # Check data_sha256 is logged
            assert run.data.params.get("data_sha256") == data_hash
            # Check MAE metric is logged
            assert "mae" in run.data.metrics

    def test_register_best_model_creates_version(
        self, sample_training_data, sample_raw_train, mlflow_tmp_db
    ):
        X, y, _ = load_training_data(sample_training_data)
        data_hash = calculate_sha256(sample_raw_train)

        with patch("src.mp293.train.get_model_configs", side_effect=fast_model_configs):
            results = run_experiments(
                X=X,
                y=y,
                data_hash=data_hash,
                tracking_uri=mlflow_tmp_db,
                experiment_name="test_register",
                raw_train_path=sample_raw_train,
            )

        model_name, version, best_run_id = register_best_model(
            results=results,
            tracking_uri=mlflow_tmp_db,
            model_name="test_model",
        )

        assert str(version) == "1"

        # Verify champion alias
        client = MlflowClient()
        mv = client.get_model_version_by_alias("test_model", "champion")
        assert str(mv.version) == "1"


class TestPrediction:
    def test_predict_with_registered_model(
        self, sample_training_data, sample_raw_train, mlflow_tmp_db
    ):
        X, y, _ = load_training_data(sample_training_data)
        data_hash = calculate_sha256(sample_raw_train)

        with patch("src.mp293.train.get_model_configs", side_effect=fast_model_configs):
            results = run_experiments(
                X=X,
                y=y,
                data_hash=data_hash,
                tracking_uri=mlflow_tmp_db,
                experiment_name="test_predict",
                raw_train_path=sample_raw_train,
            )

        register_best_model(
            results=results,
            tracking_uri=mlflow_tmp_db,
            model_name="test_pred_model",
        )

        preds = predict_with_model(
            X,
            model_name="test_pred_model",
            tracking_uri=mlflow_tmp_db,
        )

        assert len(preds) == len(X)
        assert all(isinstance(p, (int, float, np.floating)) for p in preds)
