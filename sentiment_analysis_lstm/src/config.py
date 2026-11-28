"""Shared configuration for the IMDb sentiment-analysis project.

ProjectConfig is the typed interface. load_config preserves the nested mapping
interface used by the data-pipeline scripts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Typed project settings with paths resolved from the project root."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    dataset_dir: Path | None = None
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    tensorboard_dir: Path | None = None
    seed: int = 42
    validation_size: float = 0.2
    batch_size: int = 64
    max_vocab_size: int = 30_000
    max_sequence_length: int = 300
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    num_workers: int = 0
    min_freq: int = 2
    lowercase: bool = True
    device: str = "auto"
    amp: bool = False
    bidirectional: bool = False
    pretrained_embeddings: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.dataset_dir = self._resolve(self.dataset_dir or "data/raw/aclImdb")
        self.output_dir = self._resolve(self.output_dir or "reports")
        self.checkpoint_dir = self._resolve(self.checkpoint_dir or "models")
        self.tensorboard_dir = self._resolve(self.tensorboard_dir or "runs")
        self.validate()

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()

    def validate(self) -> None:
        positive_ints = {
            "batch_size": self.batch_size, "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length, "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "epochs": self.epochs, "min_freq": self.min_freq,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        """Load either the shared nested YAML schema or a flat mapping."""
        config_path = Path(path).expanduser().resolve()
        raw = _read_yaml(config_path)
        if not isinstance(raw, dict):
            raise ValueError("The YAML configuration must contain a mapping/object.")
        if any(key in raw for key in ("data", "training", "model", "paths", "runtime")):
            data, training = raw.get("data", {}), raw.get("training", {})
            model, paths, runtime = raw.get("model", {}), raw.get("paths", {}), raw.get("runtime", {})
            values = {
                "project_root": config_path.parent.parent,
                "dataset_dir": data.get("dataset_dir", str(Path(data.get("root_dir", "data")) / "raw" / "aclImdb")),
                "output_dir": paths.get("report_dir", "reports"),
                "checkpoint_dir": paths.get("checkpoint_dir", "models"),
                "tensorboard_dir": paths.get("tensorboard_dir", "runs"),
                "seed": training.get("seed", 42),
                "validation_size": data.get("validation_size", 0.2),
                "batch_size": training.get("batch_size", 64),
                "max_vocab_size": data.get("max_vocab_size", 30000),
                "max_sequence_length": data.get("max_sequence_length", 300),
                "embedding_dim": model.get("embedding_dim", 100),
                "hidden_dim": model.get("hidden_dim", 128),
                "num_layers": model.get("num_layers", 1),
                "dropout": model.get("dropout", 0.3),
                "learning_rate": training.get("learning_rate", 1e-3),
                "weight_decay": training.get("weight_decay", 0.0),
                "epochs": training.get("epochs", 10),
                "num_workers": runtime.get("num_workers", 0),
                "min_freq": data.get("min_freq", 2),
                "lowercase": data.get("lowercase", True),
                "device": runtime.get("device", "auto"),
                "amp": runtime.get("amp", False),
                "bidirectional": model.get("bidirectional", False),
                "pretrained_embeddings": model.get("pretrained_embeddings", False),
            }
        else:
            values = dict(raw)
            values.setdefault("project_root", config_path.parent.parent)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(sorted(unknown))}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Return the nested mapping interface expected by data scripts."""
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    if not isinstance(raw, dict):
        raise ValueError("The YAML configuration must contain a mapping/object.")
    defaults: dict[str, Any] = {
        "data": {"root_dir": "data", "max_vocab_size": 30000, "max_sequence_length": 300,
                 "validation_size": 0.2, "lowercase": True, "min_freq": 2},
        "training": {"seed": 42, "batch_size": 64, "epochs": 10,
                     "learning_rate": 0.001, "weight_decay": 0.0},
        "model": {"embedding_dim": 100, "hidden_dim": 128, "num_layers": 1,
                  "dropout": 0.3, "bidirectional": False, "pretrained_embeddings": False},
        "runtime": {"device": "auto", "num_workers": 0, "deterministic": True, "amp": False},
        "paths": {"report_dir": "reports", "checkpoint_dir": "models", "tensorboard_dir": "runs"},
    }
    for section, values in raw.items():
        if section not in defaults or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        unknown = set(values) - set(defaults[section])
        if unknown:
            raise ValueError(f"Unknown {section} option(s): {sorted(unknown)}")
        defaults[section].update(values)
    project_root = config_path.parent.parent
    for section, key in (("data", "root_dir"), ("paths", "report_dir"),
                         ("paths", "checkpoint_dir"), ("paths", "tensorboard_dir")):
        value = Path(defaults[section][key]).expanduser()
        if not value.is_absolute():
            value = project_root / value
        defaults[section][key] = str(value.resolve())
    ProjectConfig.from_yaml(config_path)
    return defaults
