"""Shared configuration for the IMDb sentiment-analysis project."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Default paths and hyperparameters for the main Embedding + LSTM model."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    dataset_dir: Path | None = None
    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    tensorboard_dir: Path | None = None

    seed: int = 42
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
    bidirectional: bool = False
    pretrained_embeddings: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.dataset_dir = self._resolve(self.dataset_dir or self.project_root / "data" / "aclImdb")
        self.output_dir = self._resolve(self.output_dir or self.project_root / "reports")
        self.checkpoint_dir = self._resolve(self.checkpoint_dir or self.project_root / "models")
        self.tensorboard_dir = self._resolve(self.tensorboard_dir or self.project_root / "runs")
        self.validate()

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()

    def validate(self) -> None:
        positive_ints = {
            "batch_size": self.batch_size,
            "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0, got {value}.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        """Load configuration overrides from a YAML file."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as file:
            values: Any = yaml.safe_load(file) or {}
        if not isinstance(values, dict):
            raise ValueError("The YAML configuration must contain a mapping/object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(sorted(unknown))}")
        values.setdefault("project_root", config_path.parent.parent)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation with paths converted to strings."""
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
