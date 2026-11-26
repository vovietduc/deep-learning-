from pathlib import Path

import pytest
import torch

from src.config import ProjectConfig
from src.utils.device import describe_device, get_device
from src.utils.seed import set_seed


def test_default_config_has_project_paths():
    config = ProjectConfig()
    assert config.dataset_dir == config.project_root / "data" / "aclImdb"
    assert config.tensorboard_dir == config.project_root / "runs"
    assert config.batch_size > 0


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ProjectConfig(dropout=1.0)


def test_config_loads_yaml(tmp_path: Path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text("seed: 123\nbatch_size: 16\n", encoding="utf-8")
    config = ProjectConfig.from_yaml(config_file)
    assert config.seed == 123
    assert config.batch_size == 16


def test_config_rejects_unknown_yaml_keys(tmp_path: Path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("unknown_option: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        ProjectConfig.from_yaml(config_file)


def test_seed_reproduces_torch_random_values():
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_device_helper_returns_valid_device():
    device = get_device()
    assert device.type in {"cpu", "cuda"}
    assert isinstance(describe_device(device), str)
