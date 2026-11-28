from pathlib import Path
import json
import numpy as np

from src.data.preprocessing import clean_text, tokenize, encode_and_pad, run_preprocessing_pipeline
from src.data.tokenizer import Tokenizer
from src.data.vocabulary import PAD_IDX, UNK_IDX, Vocabulary


def _make_imdb(root: Path, per_class: int = 10) -> Path:
    imdb = root / "aclImdb"
    for split in ("train", "test"):
        for label_name, phrase in (("neg", "awful boring"), ("pos", "wonderful enjoyable")):
            folder = imdb / split / label_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                (folder / f"{index}_8.txt").write_text(
                    f"{phrase} movie number {index}! <br /> nice acting",
                    encoding="utf-8",
                )
    return imdb


def test_legacy_preprocessing_api_uses_duy_can_contract():
    raw = "Great movie!! <br /> I wasn't bored &amp; loved it."
    cleaned = clean_text(raw, lowercase=True)
    tokens = tokenize(cleaned)
    canonical = Tokenizer(lowercase=True, keep_punctuation=True).tokenize(raw)
    assert tokens == canonical
    assert "!" in tokens and "wasn't" in tokens and "&" in cleaned


def test_vocabulary_supports_both_member_apis():
    vocab = Vocabulary.build_from_tokens([["good", "movie", "good"]], max_vocab_size=10, min_freq=1)
    assert vocab.token_to_idx is vocab.token_to_id
    assert vocab.idx_to_token is vocab.id_to_token
    assert vocab.token_to_idx["<PAD>"] == PAD_IDX == 0
    assert vocab.token_to_idx["<UNK>"] == UNK_IDX == 1
    assert vocab.encode(["good", "unknown"])[1] == UNK_IDX


def test_legacy_encode_and_pad_uses_shared_vocabulary():
    vocab = Vocabulary.build_from_tokens([["good", "movie"]], max_vocab_size=10, min_freq=1)
    encoded, stats = encode_and_pad(["good", "unknown"], vocab, max_length=4)
    assert encoded[0] == vocab.token_to_id["good"]
    assert encoded[1] == UNK_IDX
    assert encoded[2:] == [PAD_IDX, PAD_IDX]
    assert stats["effective_length"] == 2
    assert stats["pad_count"] == 2


def test_preprocessing_pipeline_writes_common_artifacts(tmp_path: Path):
    imdb = _make_imdb(tmp_path, per_class=10)
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    result = run_preprocessing_pipeline(
        imdb_dir=imdb,
        processed_dir=processed,
        interim_dir=interim,
        max_vocab_size=100,
        max_sequence_length=20,
        validation_size=0.2,
        min_freq=1,
        seed=42,
    )
    assert result["splits"]["train"]["num_samples"] == 16
    assert result["splits"]["val"]["num_samples"] == 4
    assert result["splits"]["test"]["num_samples"] == 20
    train = np.load(processed / "train.npz")
    assert train["input_ids"].shape == (16, 20)
    assert train["labels"].shape == (16,)
    assert train["lengths"].shape == (16,)
    manifest = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["train_files"]) == 16
    assert len(manifest["val_files"]) == 4
    assert len(manifest["test_files"]) == 20
    assert not set(manifest["train_files"]) & set(manifest["val_files"])
    assert (processed / "vocabulary.json").is_file()
