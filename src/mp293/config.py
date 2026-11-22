
"""Load project configuration from params.yaml."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "params.yaml"


def load_config(config_path=CONFIG_PATH):
    """Load and validate project configuration."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid params.yaml configuration.")

    if "data" not in config or "preprocessing" not in config:
        raise ValueError(
            "params.yaml must contain data and preprocessing sections."
        )

    return config


if __name__ == "__main__":
    settings = load_config()

    print("CONFIGURATION LOADED SUCCESSFULLY")
    print("-" * 40)
    print(settings)
