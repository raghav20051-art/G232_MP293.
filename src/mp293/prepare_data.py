
"""Run preprocessing and save reproducible processed datasets."""

import json
from pathlib import Path

import pandas as pd

from src.mp293.config import load_config
from src.mp293.preprocessing import preprocess_all


def main():
    # Load project configuration.
    config = load_config()
    data_config = config["data"]

    # Project root directory.
    project_root = Path(__file__).resolve().parents[2]

    # Resolve paths from project root.
    train_path = project_root / data_config["train_path"]
    later_path = project_root / data_config["later_path"]
    holdout_path = project_root / data_config["holdout_path"]
    output_dir = project_root / data_config["processed_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original datasets.
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    # Run preprocessing.
    result = preprocess_all(train, later, holdout)

    # Preserve record identifiers and dates.
    metadata_columns = ["record_id", "date"]

    datasets = {
        "train_processed.csv": (
            result["train_clean"][metadata_columns].reset_index(drop=True)
            .join(result["X_train"].reset_index(drop=True))
            .join(result["y_train"].reset_index(drop=True))
        ),
        "later_processed.csv": (
            result["later_clean"][metadata_columns].reset_index(drop=True)
            .join(result["X_later"].reset_index(drop=True))
            .join(result["y_later"].reset_index(drop=True))
        ),
        "holdout_processed.csv": (
            result["holdout_clean"][metadata_columns].reset_index(drop=True)
            .join(result["X_holdout"].reset_index(drop=True))
        ),
    }

    # Save processed datasets.
    for filename, dataframe in datasets.items():
        output_path = output_dir / filename
        dataframe.to_csv(output_path, index=False)

        print(f"Saved: {output_path.relative_to(project_root)}")
        print(f"Shape: {dataframe.shape}")

    # Save training-fitted preprocessing information.
    preprocessor_path = output_dir / "preprocessor.json"

    with open(preprocessor_path, "w", encoding="utf-8") as file:
        json.dump(result["preprocessor"], file, indent=4)

    print(f"Saved: {preprocessor_path.relative_to(project_root)}")

    print("\nPREPROCESSING PIPELINE COMPLETED")
    print("-" * 45)
    print("Training rows:", len(result["X_train"]))
    print("Later rows:", len(result["X_later"]))
    print("Holdout rows:", len(result["X_holdout"]))


if __name__ == "__main__":
    main()
