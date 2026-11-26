"""Load and validate the shared project configuration.

Run from the sentiment_analysis_lstm directory:
    from src.config import load_config
    config = load_config()
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = ("data", "model", "training", "runtime", "paths", "tensorboard")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    training = config["training"]
    for key in ("batch_size", "epochs", "learning_rate", "seed"):
        if key not in training:
            raise ConfigError(f"Missing training.{key}")

    if int(training["batch_size"]) <= 0:
        raise ConfigError("training.batch_size must be greater than 0")
    if int(training["epochs"]) <= 0:
        raise ConfigError("training.epochs must be greater than 0")
    if float(training["learning_rate"]) <= 0:
        raise ConfigError("training.learning_rate must be greater than 0")
    if int(training["seed"]) < 0:
        raise ConfigError("training.seed must be non-negative")

    if int(config["model"]["num_classes"]) != 2:
        raise ConfigError("IMDb sentiment classification requires model.num_classes = 2")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and validate shared project settings.

    Args:
        path: YAML path. When omitted, configs/default.yaml is used.

    Returns:
        A dictionary containing the configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping/object.")

    _validate_config(config)
    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_path"] = str(config_path.resolve())
    return config
