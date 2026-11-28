"""Project-level command runner.

Examples (from the sentiment_analysis_lstm directory):
  python -m scripts.run_project --stage check
  python -m scripts.run_project --stage data
  python -m scripts.run_project --stage train
  python -m scripts.run_project --stage evaluate
  python -m scripts.run_project --stage all

Each stage delegates to the module owner. Placeholder modules are not treated
as completed stages.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from src.config import load_config
from src.utils.device import describe_device, resolve_device
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_module(module: str, config_path: Path, extra_args: list[str] | None = None) -> None:
    module_file = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not module_file.is_file():
        raise FileNotFoundError(f"Stage {module!r} is not available: {module_file}")
    source = module_file.read_text(encoding="utf-8")
    if "TODO:" in source and "implement" in source.lower():
        raise RuntimeError(f"Stage {module!r} is still a placeholder; its owner must finish it first.")
    command = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        command.extend(extra_args)
    print(f"\n[run_project] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--stage", choices=("check", "data", "train", "evaluate", "all"), default="check")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config_path = config_path.resolve()
    cfg = load_config(config_path)
    seed = int(cfg["training"]["seed"])
    set_seed(seed, deterministic=bool(cfg["runtime"].get("deterministic", True)))
    device = resolve_device(str(cfg["runtime"].get("device", "auto")))

    print("=== IMDb Sentiment Analysis: Project Check ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {config_path}")
    print(f"Seed         : {seed}")
    print(f"Device       : {device}")
    print(f"Description  : {describe_device(device)}")
    print(f"Data root    : {cfg['data']['root_dir']}")
    print(f"Reports      : {cfg['paths']['report_dir']}")
    print(f"Checkpoints  : {cfg['paths']['checkpoint_dir']}")
    print(f"TensorBoard  : {cfg['paths']['tensorboard_dir']}")
    if args.stage == "check":
        print("[run_project] Configuration and runtime checks passed.")
        return 0

    skip = ["--skip-download"] if args.skip_download else []
    stages = {
        "data": [("scripts.run_data_pipeline", skip)],
        "train": [("src.train", [])],
        "evaluate": [("src.evaluate", [])],
        "all": [("scripts.run_data_pipeline", skip), ("src.train", []), ("src.evaluate", [])],
    }
    for module, extra in stages[args.stage]:
        _run_module(module, config_path, extra)
    print(f"[run_project] Stage {args.stage!r} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
