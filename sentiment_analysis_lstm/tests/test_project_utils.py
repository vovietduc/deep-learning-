"""Tests for shared configuration, seed and device utilities."""
from pathlib import Path

import pytest
import torch

from src.config import ConfigError, load_config
from src.utils.device import get_device, get_device_info
from src.utils.seed import seed_everything


def test_default_config_loads_and_has_binary_labels():
    config = load_config()
    assert config["model"]["num_classes"] == 2
    assert config["training"]["seed"] == 42
    assert config["training"]["batch_size"] > 0


def test_config_rejects_invalid_batch_size(tmp_path: Path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
data: {}
model:
  num_classes: 2
training:
  batch_size: 0
  epochs: 2
  learning_rate: 0.001
  seed: 42
runtime: {}
paths: {}
tensorboard: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="batch_size"):
        load_config(invalid)


def test_seed_rejects_negative_value():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_reproduces_torch_random_values():
    seed_everything(123)
    first = torch.rand(3)
    seed_everything(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_device_helper_returns_supported_device():
    device = get_device()
    assert device.type in {"cpu", "cuda"}
    info = get_device_info(device)
    assert info["device"] == str(device)
    assert isinstance(info["cuda_available"], bool)
