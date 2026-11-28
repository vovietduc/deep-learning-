"""Preprocessing adapter that reuses Duy Can's canonical tokenizer/vocabulary.

This module keeps the function signatures expected by Quoc Anh's audit/EDA
pipeline while ensuring the entire project uses one tokenizer and vocabulary.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LABEL_MAPPING = {"neg": 0, "pos": 1}
from src.data.padding import effective_length, pad_or_truncate
from src.data.tokenizer import Tokenizer
from src.data.vocabulary import PAD_IDX, PAD_TOKEN, UNK_IDX, UNK_TOKEN, Vocabulary

_DEFAULT_TOKENIZER = Tokenizer(lowercase=False, keep_punctuation=True)


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalize raw review text with the shared project tokenizer."""
    return Tokenizer(lowercase=lowercase, keep_punctuation=True).normalize(text)


def tokenize(text: str) -> List[str]:
    """Tokenize already-normalized text without changing its configured casing."""
    return _DEFAULT_TOKENIZER.tokenize(text)


def encode_and_pad(tokens: Sequence[str], vocab: Vocabulary, max_length: int) -> Tuple[List[int], Dict[str, Any]]:
    """Encode tokens and pad/truncate to the fixed sequence length."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    raw_ids = vocab.encode(list(tokens))
    original_length = len(raw_ids)
    ids = pad_or_truncate(raw_ids, max_length, pad_id=PAD_IDX, padding="post", truncating="post")
    effective = effective_length(original_length, max_length)
    stats = {
        "original_length": original_length,
        "effective_length": effective,
        "was_truncated": original_length > max_length,
        "pad_count": max_length - min(original_length, max_length),
        "oov_count": sum(token_id == UNK_IDX for token_id in raw_ids[:max_length]),
    }
    return ids, stats


def load_split_reviews(
    imdb_dir: str | Path,
    split: str,
    max_per_class: Optional[int] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """Load labeled reviews from train or test; unsup reviews are excluded."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported supervised split: {split}.")
    base = Path(imdb_dir) / split
    texts: List[str] = []
    labels: List[int] = []
    filenames: List[str] = []
    for label_name in ("neg", "pos"):
        folder = base / label_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected directory missing: {folder}")
        files = sorted(folder.glob("*.txt"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for file_path in files:
            texts.append(file_path.read_text(encoding="utf-8", errors="replace"))
            labels.append(int(LABEL_MAPPING[label_name]))
            filenames.append(f"{split}/{label_name}/{file_path.name}")
    return texts, labels, filenames


def stratified_train_val_split(
    texts: Sequence[str],
    labels: Sequence[int],
    filenames: Sequence[str],
    validation_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[int], List[str]]]:
    """Deterministically split only the official train partition."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1.")
    if not (len(texts) == len(labels) == len(filenames)):
        raise ValueError("texts, labels and filenames must have equal lengths.")
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        if int(label) not in by_class:
            raise ValueError(f"Invalid sentiment label: {label}")
        by_class[int(label)].append(index)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        n_val = int(round(len(indices) * validation_size))
        if indices and n_val == 0:
            n_val = 1
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    def gather(indices: List[int]):
        return (
            [texts[i] for i in indices],
            [int(labels[i]) for i in indices],
            [filenames[i] for i in indices],
        )
    return {"train": gather(train_indices), "val": gather(val_indices)}


def _encode_split(
    tokenized_texts: Sequence[Sequence[str]],
    labels: Sequence[int],
    vocab: Vocabulary,
    max_length: int,
):
    n = len(tokenized_texts)
    input_ids = np.zeros((n, max_length), dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    total_tokens = total_oov = truncated_docs = total_pad = 0
    for index, tokens in enumerate(tokenized_texts):
        ids, stats = encode_and_pad(tokens, vocab, max_length)
        input_ids[index] = ids
        lengths[index] = stats["effective_length"]
        total_tokens += stats["effective_length"]
        total_oov += stats["oov_count"]
        truncated_docs += int(stats["was_truncated"])
        total_pad += stats["pad_count"]
    summary = {
        "num_samples": n,
        "total_effective_tokens": total_tokens,
        "oov_tokens": total_oov,
        "oov_rate": round(total_oov / max(1, total_tokens), 6),
        "truncated_docs": truncated_docs,
        "truncation_rate": round(truncated_docs / max(1, n), 6),
        "padding_rate": round(total_pad / max(1, n * max_length), 6),
    }
    return input_ids, label_array, lengths, summary


def run_preprocessing_pipeline(
    imdb_dir: str | Path,
    processed_dir: str | Path = "data/processed",
    interim_dir: str | Path = "data/interim",
    max_vocab_size: int = 30000,
    max_sequence_length: int = 300,
    validation_size: float = 0.2,
    lowercase: bool = True,
    min_freq: int = 2,
    seed: int = 42,
    max_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """Load -> stratified split -> shared tokenize -> train-only vocabulary -> encode."""
    processed = Path(processed_dir)
    interim = Path(interim_dir)
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    raw_train_texts, raw_train_labels, raw_train_files = load_split_reviews(
        imdb_dir, "train", max_per_class=max_per_class
    )
    raw_test_texts, raw_test_labels, raw_test_files = load_split_reviews(
        imdb_dir, "test", max_per_class=max_per_class
    )
    split = stratified_train_val_split(
        raw_train_texts, raw_train_labels, raw_train_files,
        validation_size=validation_size, seed=seed,
    )
    train_texts, train_labels, train_files = split["train"]
    val_texts, val_labels, val_files = split["val"]

    tokenizer = Tokenizer(lowercase=lowercase, keep_punctuation=True)
    train_tokens = [tokenizer.tokenize(text) for text in train_texts]
    val_tokens = [tokenizer.tokenize(text) for text in val_texts]
    test_tokens = [tokenizer.tokenize(text) for text in raw_test_texts]

    interim_summary = {
        "seed": seed,
        "lowercase": lowercase,
        "tokenizer": tokenizer.get_config(),
        "splits": {"train": len(train_texts), "val": len(val_texts), "test": len(raw_test_texts)},
    }
    (interim / "interim_summary.json").write_text(
        json.dumps(interim_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    vocab = Vocabulary.build_from_tokens(
        train_tokens, max_vocab_size=max_vocab_size, min_freq=min_freq
    )
    vocab_path = vocab.save(processed / "vocabulary.json")

    split_stats: Dict[str, Any] = {}
    for name, tokens, labels in (
        ("train", train_tokens, train_labels),
        ("val", val_tokens, val_labels),
        ("test", test_tokens, raw_test_labels),
    ):
        x, y, lengths, stats = _encode_split(tokens, labels, vocab, max_sequence_length)
        np.savez_compressed(processed / f"{name}.npz", input_ids=x, labels=y, lengths=lengths)
        split_stats[name] = stats

    manifest = {
        "seed": seed,
        "validation_size": validation_size,
        "label_mapping": LABEL_MAPPING,
        "pad_token": PAD_TOKEN,
        "pad_idx": PAD_IDX,
        "unk_token": UNK_TOKEN,
        "unk_idx": UNK_IDX,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": raw_test_files,
    }
    (processed / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "seed": seed,
        "lowercase": lowercase,
        "tokenizer": tokenizer.get_config(),
        "min_freq": min_freq,
        "max_vocab_size": max_vocab_size,
        "vocab_size": len(vocab),
        "max_sequence_length": max_sequence_length,
        "validation_size": validation_size,
        "label_mapping": LABEL_MAPPING,
        "special_tokens": {"pad_token": PAD_TOKEN, "pad_idx": PAD_IDX,
                           "unk_token": UNK_TOKEN, "unk_idx": UNK_IDX},
        "vocab_path": str(vocab_path),
        "splits": split_stats,
    }
    (processed / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
