"""Shared paths, reproducibility and JSON output."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def configure(seed=42):
    import torch
    from transformers import set_seed
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(seed)

def metrics(prediction):
    import numpy as np
    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, axis=-1)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": float((pred == labels).mean()), "precision": precision,
            "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
