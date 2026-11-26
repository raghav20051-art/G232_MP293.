"""Export the MLflow champion model to models/model.joblib.

Usage:
    python -m src.mp293.export_model

This script retrieves the model registered as 'atm_cash_demand_model' with
the 'champion' alias from the project's MLflow SQLite backend and serialises
it to models/model.joblib for use by the FastAPI service.
"""

import sys
from pathlib import Path

import joblib
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
REGISTERED_MODEL_NAME = "atm_cash_demand_model"
OUTPUT_PATH = PROJECT_ROOT / "models" / "model.joblib"


def export_champion_model(
    tracking_uri: str = TRACKING_URI,
    model_name: str = REGISTERED_MODEL_NAME,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Load champion model from MLflow and export to joblib."""
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}@champion"
    print(f"Loading model from: {model_uri}")
    print(f"Tracking URI: {tracking_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"ERROR: Failed to load model from MLflow: {exc}", file=sys.stderr)
        print(
            "Ensure that:\n"
            "  1. mlflow.db exists in the project root\n"
            "  2. Stage 2 training has been run\n"
            "  3. The 'champion' alias has been assigned",
            file=sys.stderr,
        )
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model exported to: {output_path}")

    # Verify round-trip
    loaded = joblib.load(output_path)
    assert hasattr(loaded, "predict"), "Round-trip verification failed"
    print("Round-trip verification: OK")

    return output_path


if __name__ == "__main__":
    export_champion_model()
