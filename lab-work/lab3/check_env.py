"""Kiểm tra đúng Python và import dependencies trước khi chạy bài."""
import sys
import struct
import importlib
import importlib.metadata
import json
from pathlib import Path

def check_environment():
    print("Python:", sys.version.split()[0], "|", sys.executable)
    if sys.version_info[:2] != (3, 14) or struct.calcsize("P") != 8:
        print("Can Python 3.14 64-bit. Chon .venv314/Scripts/python.exe trong VS Code.")
        return False
    modules = {"numpy":"numpy", "torch":"torch", "transformers":"transformers",
               "datasets":"datasets", "evaluate":"evaluate", "accelerate":"accelerate",
               "sklearn":"scikit-learn", "matplotlib":"matplotlib", "ipykernel":"ipykernel"}
    versions = {}
    ok = True
    for module, package in modules.items():
        try:
            importlib.import_module(module)
            versions[package] = importlib.metadata.version(package)
            print("OK", package, versions[package])
        except Exception as exc:
            ok = False
            print("LOI", package, type(exc).__name__, str(exc))
    if ok:
        try:
            from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
            import torch
            print("Trainer import OK | CUDA:", torch.cuda.is_available())
            data = {"python":sys.version, "executable":sys.executable,
                    "packages":versions, "cuda":torch.cuda.is_available()}
            Path(__file__).with_name("environment_check.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print("LOI Trainer:", repr(exc))
            ok = False
    if not ok:
        print("Hay chay 01_SETUP_WINDOWS.cmd, cho den khi hien SETUP OK.")
        print("Neu da cai: chon dung .venv314/Scripts/python.exe, khong dung Python global.")
    return ok

if __name__ == "__main__":
    raise SystemExit(0 if check_environment() else 1)
