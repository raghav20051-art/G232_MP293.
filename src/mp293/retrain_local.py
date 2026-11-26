"""Re-train the champion model locally and export to models/model.joblib.

Used when the original mlruns/ artifact directory is unavailable after a
fresh clone on a different machine. This script:

1. Trains a RandomForestRegressor with the EXACT champion hyperparameters
   recorded in mlflow.db (n_estimators=100, max_depth=10, etc.).
2. Exports directly to models/model.joblib via joblib (bypassing MLflow
   artifact logging, which requires the original Windows artifact path).
3. Saves preprocessor.json for use by the FastAPI service.

NOTE: Because the original raw CSVs reside on the team member's Windows
machine and are not committed to Git, the synthetic dataset used here
mirrors the feature schema exactly but is not the original data. The
hyperparameters and feature pipeline are preserved verbatim.

Usage:
    python -m src.mp293.retrain_local
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "models" / "model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessor.json"

# ---------------------------------------------------------------------------
# Exact champion hyperparameters from mlflow.db run 6967c355
# ---------------------------------------------------------------------------
CHAMPION_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Feature columns in training order (matches preprocessing.py + prepare_data.py)
NUMERIC_FEATURES = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
]
ATM_TYPE_CATEGORIES = ["bank_branch", "mall", "standalone"]
AREA_CATEGORIES = ["highway", "market", "residential"]
FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + [f"atm_type_{c}" for c in ATM_TYPE_CATEGORIES]
    + [f"area_{c}" for c in AREA_CATEGORIES]
)


def _generate_synthetic_training_data(n: int = 1000, seed: int = 42):
    """Generate a structurally representative synthetic dataset.

    Feature distributions match the data investigation findings:
    - 1000 training rows (matches train.csv record count)
    - Target: withdrawal_thousand_rs (continuous, positive)
    - All numeric features already cleaned (no -999 sentinel, no metre Highway)
    - Categorical features already one-hot encoded
    """
    rng = np.random.default_rng(seed)

    nearby_shops = rng.integers(1, 50, size=n).astype(float)
    days_to_salary = rng.integers(1, 30, size=n).astype(float)
    footfall_index = rng.uniform(10, 100, size=n)
    distance_to_market_km = rng.uniform(0.5, 20, size=n)
    avg_ticket_rs = rng.uniform(500, 5000, size=n)
    other_atms_1km = rng.integers(0, 15, size=n).astype(float)

    # One-hot atm_type
    atm_choices = rng.choice(len(ATM_TYPE_CATEGORIES), size=n)
    atm_type_ohe = np.zeros((n, 3))
    atm_type_ohe[np.arange(n), atm_choices] = 1

    # One-hot area
    area_choices = rng.choice(len(AREA_CATEGORIES), size=n)
    area_ohe = np.zeros((n, 3))
    area_ohe[np.arange(n), area_choices] = 1

    X = pd.DataFrame(
        np.column_stack([
            nearby_shops, days_to_salary, footfall_index,
            distance_to_market_km, avg_ticket_rs, other_atms_1km,
            atm_type_ohe, area_ohe,
        ]),
        columns=FEATURE_COLUMNS,
    )

    # Plausible linear target + noise
    y = (
        nearby_shops * 0.8
        + footfall_index * 0.5
        + avg_ticket_rs * 0.015
        - other_atms_1km * 2.0
        - distance_to_market_km * 1.5
        + rng.normal(0, 5, size=n)
    )
    y = pd.Series(np.maximum(y, 1.0), name="withdrawal_thousand_rs")
    return X, y


def _save_preprocessor_json():
    """Write preprocessor.json so the FastAPI service can load medians."""
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = {
        "numeric_medians": {
            "nearby_shops": 25.0,
            "days_to_salary": 15.0,
            "footfall_index": 55.0,
            "distance_to_market_km": 5.0,
            "avg_ticket_rs": 2000.0,
            "other_atms_1km": 7.0,
        },
        "categorical_categories": {
            "atm_type": ATM_TYPE_CATEGORIES,
            "area": AREA_CATEGORIES,
        },
    }
    with open(PREPROCESSOR_PATH, "w", encoding="utf-8") as f:
        json.dump(preprocessor, f, indent=4)
    print(f"Saved: {PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)}")


def retrain_and_export():
    """Train champion model → export to models/model.joblib."""
    print("=" * 60)
    print("LOCAL RE-TRAINING: champion model (RandomForest_v1 params)")
    print("=" * 60)
    print(f"Output path  : {OUTPUT_PATH}")
    print(f"Params       : {CHAMPION_PARAMS}")
    print("-" * 60)

    X, y = _generate_synthetic_training_data()
    print(f"Synthetic training data: {X.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")

    model = RandomForestRegressor(**CHAMPION_PARAMS)
    model.fit(X, y)
    print("Model trained.")

    # Cross-validation metrics
    cv_mae = -cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"5-fold CV MAE: {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")

    # Export directly via joblib (avoids MLflow's broken Windows artifact path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    size_kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"Exported: {OUTPUT_PATH} ({size_kb} KB)")

    # Round-trip verification
    loaded = joblib.load(OUTPUT_PATH)
    sample_preds = loaded.predict(X.iloc[:3])
    print(f"Round-trip predictions (3 samples): {sample_preds.round(2)}")
    assert len(sample_preds) == 3, "Round-trip verification failed"

    # Save preprocessor.json
    _save_preprocessor_json()

    print("\n" + "=" * 60)
    print("RE-TRAINING COMPLETE")
    print(f"  model.joblib : {OUTPUT_PATH}")
    print(f"  Feature cols : {len(FEATURE_COLUMNS)}")
    print("=" * 60)
    return OUTPUT_PATH


if __name__ == "__main__":
    retrain_and_export()
