"""Stage 2: Model training with MLflow experiment tracking.

Trains multiple regression models on preprocessed ATM cash demand data,
logs experiments to MLflow (SQLite backend), and registers the best model.
"""

import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "ATM_Cash_Demand_Prediction"
REGISTERED_MODEL_NAME = "atm_cash_demand_model"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
TARGET = "withdrawal_thousand_rs"
METADATA_COLS = ["record_id", "date"]


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_training_data(processed_path: Path | None = None):
    """Load processed training data and split into features / target."""
    if processed_path is None:
        processed_path = PROCESSED_DIR / "train_processed.csv"

    df = pd.read_csv(processed_path)

    # Drop metadata columns
    feature_cols = [c for c in df.columns if c not in METADATA_COLS + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET]

    return X, y, feature_cols


def evaluate_model(model, X, y):
    """Compute regression metrics using 5-fold cross-validation."""
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error"))
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Also fit on full training set for final metrics
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        "mae": float(np.mean(cv_mae)),
        "mae_std": float(np.std(cv_mae)),
        "rmse": float(np.mean(cv_rmse)),
        "rmse_std": float(np.std(cv_rmse)),
        "r2": float(np.mean(cv_r2)),
        "r2_std": float(np.std(cv_r2)),
        "train_mae": float(mean_absolute_error(y, y_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "train_r2": float(r2_score(y, y_pred)),
    }

    return metrics


def get_model_configs():
    """Return a list of (name, model, params_dict) tuples to try."""
    configs = [
        (
            "RandomForest_v1",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            ),
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        ),
        (
            "GradientBoosting_v1",
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
            ),
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        ),
        (
            "RandomForest_v2",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
            ),
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        ),
    ]

    # Try to include XGBoost if available
    try:
        from xgboost import XGBRegressor

        configs.append(
            (
                "XGBoost_v1",
                XGBRegressor(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
                {
                    "model_type": "XGBRegressor",
                    "n_estimators": 150,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
            )
        )
    except ImportError:
        print("XGBoost not available, skipping.")

    return configs


def run_experiments(
    X=None,
    y=None,
    data_hash=None,
    tracking_uri=None,
    experiment_name=None,
    raw_train_path=None,
):
    """Run all model experiments and log to MLflow.

    Returns:
        List of (run_id, name, mae) tuples for all runs.
    """
    if tracking_uri is None:
        tracking_uri = TRACKING_URI
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
    if raw_train_path is None:
        raw_train_path = RAW_TRAIN_PATH

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if X is None or y is None:
        X, y, _ = load_training_data()

    if data_hash is None:
        data_hash = calculate_sha256(raw_train_path)

    print(f"Training data shape: {X.shape}")
    print(f"Data SHA-256: {data_hash}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print("-" * 60)

    configs = get_model_configs()
    results = []

    for name, model, params in configs:
        print(f"\n--- Training: {name} ---")

        with mlflow.start_run(run_name=name) as run:
            # Log parameters
            mlflow.log_param("data_sha256", data_hash)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Evaluate
            metrics = evaluate_model(model, X, y)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model artifact
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_id = run.info.run_id
            mae = metrics["mae"]
            results.append((run_id, name, mae))

            print(f"  Run ID: {run_id}")
            print(f"  CV MAE: {mae:.4f} Â± {metrics['mae_std']:.4f}")
            print(f"  CV RMSE: {metrics['rmse']:.4f}")
            print(f"  CV RÂ²: {metrics['r2']:.4f}")

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    for run_id, name, mae in sorted(results, key=lambda x: x[2]):
        print(f"  {name:30s}  MAE={mae:.4f}  run={run_id[:8]}")

    return results


def register_best_model(
    results=None,
    tracking_uri=None,
    model_name=None,
):
    """Register the best model (lowest MAE) and assign 'champion' alias.

    Returns:
        (model_name, version, run_id)
    """
    if tracking_uri is None:
        tracking_uri = TRACKING_URI
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Find the best run
    best_run_id, best_name, best_mae = min(results, key=lambda x: x[2])
    print(f"\nBest model: {best_name} (MAE={best_mae:.4f}, run={best_run_id[:8]})")

    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    print(f"Registered: {model_name} v{version}")

    # Assign champion alias
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"Alias 'champion' assigned to v{version}")

    return model_name, version, best_run_id


def predict_with_model(X, model_name=None, tracking_uri=None):
    """Load the champion model and predict."""
    if tracking_uri is None:
        tracking_uri = TRACKING_URI
    if model_name is None:
        model_name = REGISTERED_MODEL_NAME

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    return model.predict(X)


def main():
    """Full Stage 2 pipeline: train, evaluate, register."""
    print("=" * 60)
    print("STAGE 2: MODEL TRAINING AND REGISTRATION")
    print("=" * 60)

    # Load data
    X, y, feature_cols = load_training_data()
    data_hash = calculate_sha256(RAW_TRAIN_PATH)

    # Run experiments
    results = run_experiments(X=X, y=y, data_hash=data_hash)

    # Register best model
    model_name, version, best_run_id = register_best_model(results)

    # Update findings.yaml with tracking info
    _update_findings(best_run_id, model_name, version)

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"  Experiment runs: {len(results)}")
    print(f"  Best run ID: {best_run_id}")
    print(f"  Registered model: {model_name} v{version}")
    print(f"  Champion alias: set")

    return results, model_name, version, best_run_id


def _update_findings(best_run_id, model_name, version):
    """Update the tracking section of findings.yaml."""
    import yaml

    findings_path = PROJECT_ROOT / "reports" / "findings.yaml"
    if not findings_path.exists():
        print("findings.yaml not found, skipping update.")
        return

    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    findings["tracking"] = {
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": best_run_id,
        "registered_model_name": model_name,
        "champion_version": int(version),
    }

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.dump(findings, f, default_flow_style=False, sort_keys=False)

    print(f"Updated findings.yaml tracking section.")


if __name__ == "__main__":
    main()

