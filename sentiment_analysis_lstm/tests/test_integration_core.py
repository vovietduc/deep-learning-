from pathlib import Path

import pytest
import torch

from src.config import ProjectConfig, load_config
from src.utils.device import describe_device, get_device, resolve_device
from src.utils.seed import seed_everything, set_seed
from src.utils.tensorboard import close_writer, create_summary_writer, log_epoch_metrics


def test_default_config_has_project_paths():
    config = ProjectConfig()
    assert config.dataset_dir == config.project_root / "data" / "raw" / "aclImdb"
    assert config.tensorboard_dir == config.project_root / "runs"
    assert config.batch_size > 0


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ProjectConfig(dropout=1.0)


def test_config_loads_nested_yaml(tmp_path: Path):
    config_file = tmp_path / "configs" / "default.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        "data:\n  validation_size: 0.2\n"
        "training:\n  seed: 123\n  batch_size: 16\n"
        "runtime:\n  device: cpu\n",
        encoding="utf-8",
    )
    typed = ProjectConfig.from_yaml(config_file)
    legacy = load_config(config_file)
    assert typed.seed == 123
    assert typed.batch_size == 16
    assert typed.validation_size == 0.2
    assert legacy["training"]["seed"] == 123
    assert legacy["runtime"]["device"] == "cpu"
    assert Path(legacy["data"]["root_dir"]).is_absolute()


def test_config_rejects_unknown_yaml_keys(tmp_path: Path):
    config_file = tmp_path / "configs" / "bad.yaml"
    config_file.parent.mkdir()
    config_file.write_text("unknown_option: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config_file)


def test_seed_reproduces_torch_random_values():
    set_seed(7)
    first = torch.rand(3)
    seed_everything(7)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_device_helper_returns_valid_device():
    device = get_device()
    assert device.type in {"cpu", "cuda"}
    assert isinstance(describe_device(device), str)
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")


def test_tensorboard_writes_event_file(tmp_path: Path):
    writer = create_summary_writer(tmp_path, "test-run")
    log_epoch_metrics(writer, 1, {"loss/train": 0.5, "accuracy/val": 0.8})
    close_writer(writer)
    assert list((tmp_path / "test-run").glob("events.out.tfevents.*"))
