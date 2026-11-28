"""Shared TensorBoard helpers; the model owner calls them from the epoch loop."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
from torch.utils.tensorboard import SummaryWriter


def create_summary_writer(log_dir: str | Path, run_name: str, comment: str = "") -> SummaryWriter:
    """Create a TensorBoard writer under log_dir/run_name."""
    run_path = Path(run_name)
    if not run_name or run_path.is_absolute() or ".." in run_path.parts:
        raise ValueError("run_name must be a safe relative directory name.")
    target = Path(log_dir) / run_path
    target.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(target), comment=comment)


def log_epoch_metrics(writer: SummaryWriter, epoch: int, metrics: Mapping[str, float],
                      prefix: str = "") -> None:
    """Write numeric epoch metrics with stable tag names."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric {name!r} must be numeric.")
        tag = f"{prefix}/{name}" if prefix else name
        writer.add_scalar(tag, float(value), epoch)


def close_writer(writer: SummaryWriter) -> None:
    """Flush pending events and close the writer."""
    writer.flush()
    writer.close()
