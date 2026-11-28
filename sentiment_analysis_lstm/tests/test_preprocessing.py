"""Unit test cho pipeline tiền xử lý: tokenizer, vocabulary, padding, dataset.

Các test dataset cần PyTorch; nếu chưa cài torch chúng sẽ được SKIP (không FAIL).
"""

import json

import pytest

from src.data.padding import effective_length, pad_batch, pad_or_truncate
from src.data.tokenizer import Tokenizer
from src.data.vocabulary import Vocabulary


# ============================================================ Tokenizer
class TestTokenizer:
    def test_empty_text(self):
        assert Tokenizer().tokenize("") == []

    def test_none_and_whitespace_only(self):
        tok = Tokenizer()
        assert tok.tokenize(None) == []
        assert tok.tokenize("   \n\t  ") == []

    def test_html_removed_and_lowercase(self):
        result = Tokenizer(keep_punctuation=False).tokenize("I LOVE this movie! <br />")
        assert result == ["i", "love", "this", "movie"]

    def test_html_tag_splits_words(self):
        assert Tokenizer().tokenize("good<br /><br />movie") == ["good", "movie"]

    def test_html_entities(self):
        assert Tokenizer().tokenize("Tom &amp; Jerry") == ["tom", "jerry"]
        assert Tokenizer().tokenize("&lt;br&gt;great") == ["great"]

    def test_comparison_symbols_not_treated_as_html(self):
        # "< b and c >" không phải thẻ HTML nên không được xoá
        assert Tokenizer().tokenize("a < b and c > d") == ["a", "b", "and", "c", "d"]

    def test_contractions_kept_as_one_token(self):
        assert Tokenizer().tokenize("It wasn't bad, didn't hate it") == [
            "it", "wasn't", "bad", "didn't", "hate", "it",
        ]

    def test_curly_apostrophe_normalized(self):
        assert Tokenizer().tokenize("don’t") == ["don't"]

    def test_punctuation_option(self):
        assert Tokenizer(keep_punctuation=True).tokenize("Great!!") == ["great", "!", "!"]
        assert Tokenizer(keep_punctuation=False).tokenize("Great!!") == ["great"]

    def test_lowercase_false_keeps_uppercase_tokens(self):
        # Lỗi cũ: regex chỉ có [a-z] nên lowercase=False làm mất chữ hoa
        assert Tokenizer(lowercase=False, keep_punctuation=False).tokenize("Great Movie") == [
            "Great", "Movie",
        ]

    def test_unicode_letters_kept(self):
        assert Tokenizer().tokenize("Amélie was great") == ["amélie", "was", "great"]

    def test_deterministic_and_callable(self):
        tok = Tokenizer()
        assert tok("Nice film") == tok.tokenize("Nice film")

    def test_config(self):
        assert Tokenizer(lowercase=False).get_config()["lowercase"] is False


# ============================================================ Vocabulary
class TestVocabulary:
    @staticmethod
    def _vocab():
        v = Vocabulary()
        v.build([["i", "love", "movie"], ["i", "love", "film"]])
        return v

    def test_special_tokens(self):
        v = Vocabulary()
        assert v.token_to_id["<PAD>"] == 0
        assert v.token_to_id["<UNK>"] == 1
        assert len(v) == 2

    def test_unknown_token_maps_to_unk(self):
        v = self._vocab()
        encoded = v.encode(["i", "unknown", "movie"])
        assert encoded[0] == v.token_to_id["i"]
        assert encoded[1] == v.UNK_ID
        assert encoded[2] == v.token_to_id["movie"]

    def test_ids_start_at_two_and_frequent_first(self):
        v = self._vocab()
        # i, love xuất hiện 2 lần -> ID nhỏ hơn film, movie (1 lần)
        assert v.token_to_id["i"] == 2
        assert v.token_to_id["love"] == 3
        assert min(v.token_to_id["film"], v.token_to_id["movie"]) == 4

    def test_build_independent_of_input_order(self):
        a, b = Vocabulary(), Vocabulary()
        a.build([["x", "y"], ["z"]])
        b.build([["z"], ["y", "x"]])
        assert a.token_to_id == b.token_to_id

    def test_min_freq(self):
        v = Vocabulary(min_freq=2)
        v.build([["a", "a", "b"]])
        assert "a" in v and "b" not in v

    def test_max_size_includes_special_tokens(self):
        v = Vocabulary(max_size=4)
        v.build([["a", "a", "a", "b", "b", "c"]])
        assert len(v) == 4
        assert "a" in v and "b" in v and "c" not in v

    def test_rebuild_does_not_accumulate(self):
        v = Vocabulary()
        v.build([["old"]])
        v.build([["new"]])
        assert "old" not in v and "new" in v

    def test_decode_roundtrip_and_unknown_id(self):
        v = self._vocab()
        ids = v.encode(["i", "love", "film"])
        assert v.decode(ids) == ["i", "love", "film"]
        assert v.decode([999]) == ["<UNK>"]
        assert v.decode([0, 2], skip_pad=True) == ["i"]

    def test_empty_build_only_special_tokens(self):
        v = Vocabulary()
        v.build([])
        assert len(v) == 2
        assert v.encode(["anything"]) == [v.UNK_ID]

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            Vocabulary(min_freq=0)
        with pytest.raises(ValueError):
            Vocabulary(max_size=1)

    def test_save_and_load(self, tmp_path):
        v = self._vocab()
        v.save(tmp_path / "vocab.json")
        loaded = Vocabulary.load(tmp_path / "vocab.json")
        assert loaded.token_to_id == v.token_to_id
        assert loaded.id_to_token == v.id_to_token
        assert loaded.encode(["i", "zzz"]) == v.encode(["i", "zzz"])


# ============================================================ Padding
class TestPadding:
    def test_short_sequence_padding(self):
        assert pad_or_truncate([2, 3, 4], max_length=5, pad_id=0) == [2, 3, 4, 0, 0]

    def test_long_sequence_truncation(self):
        assert pad_or_truncate([1, 2, 3, 4, 5, 6], max_length=4) == [1, 2, 3, 4]

    def test_exact_length(self):
        assert pad_or_truncate([1, 2, 3], max_length=3) == [1, 2, 3]

    def test_empty_sequence_is_all_padding(self):
        assert pad_or_truncate([], max_length=3) == [0, 0, 0]

    def test_pre_padding_and_pre_truncating(self):
        assert pad_or_truncate([2, 3], 4, padding="pre") == [0, 0, 2, 3]
        assert pad_or_truncate([1, 2, 3, 4, 5], 3, truncating="pre") == [3, 4, 5]

    def test_returns_new_list(self):
        seq = [1, 2, 3]
        out = pad_or_truncate(seq, 3)
        assert out == seq and out is not seq

    def test_custom_pad_id(self):
        assert pad_or_truncate([7], 3, pad_id=9) == [7, 9, 9]

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            pad_or_truncate([1], 0)
        with pytest.raises(ValueError):
            pad_or_truncate([1], 3, padding="left")

    def test_pad_batch(self):
        assert pad_batch([[1], [1, 2, 3]], max_length=2) == [[1, 0], [1, 2]]

    def test_effective_length(self):
        assert effective_length(3, 5) == 3
        assert effective_length(9, 5) == 5
        assert effective_length(0, 5) == 1  # pack_padded_sequence không nhận độ dài 0


# ============================================================ Dataset (cần torch)
@pytest.fixture
def torch_mod():
    return pytest.importorskip("torch")


@pytest.fixture
def fake_imdb(tmp_path):
    """IMDb giả nhỏ: 6 review mỗi lớp, trong thư mục train/{pos,neg}."""
    train = tmp_path / "train"
    (train / "pos").mkdir(parents=True)
    (train / "neg").mkdir(parents=True)
    for i in range(6):
        (train / "pos" / f"{i}_9.txt").write_text(f"I love this movie! great film {i}", encoding="utf-8")
        (train / "neg" / f"{i}_2.txt").write_text(f"I hate this movie! awful film {i}", encoding="utf-8")
    return train


def _make_vocab(samples, tokenizer, **kw):
    from src.data.dataset import build_vocabulary

    return build_vocabulary(samples, tokenizer, min_freq=1, **kw)


def test_dataset_output_shape_and_dtype(torch_mod, fake_imdb):
    from src.data.dataset import IMDBDataset, collect_samples

    tok = Tokenizer()
    vocab = _make_vocab(collect_samples(fake_imdb), tok)
    ds = IMDBDataset(tok, vocab, max_length=10, split_dir=fake_imdb)

    assert len(ds) == 12
    input_ids, label = ds[0]
    assert input_ids.shape == (10,)
    assert input_ids.dtype == torch_mod.long
    assert label.dtype == torch_mod.long
    assert label.item() == 0  # neg đứng trước pos
    assert ds[len(ds) - 1][1].item() == 1


def test_dataset_values_match_manual_pipeline(torch_mod, fake_imdb):
    """Review -> tokens -> IDs -> padded sequence -> tensor, đối chiếu từng bước."""
    from src.data.dataset import IMDBDataset, collect_samples

    tok = Tokenizer()
    samples = collect_samples(fake_imdb)
    vocab = _make_vocab(samples, tok)
    ds = IMDBDataset(tok, vocab, max_length=12, samples=samples)

    path, _ = samples[0]
    text = open(path, encoding="utf-8").read()
    expected = pad_or_truncate(vocab.encode(tok.tokenize(text)), 12, pad_id=vocab.PAD_ID)

    assert ds[0][0].tolist() == expected
    assert expected[-1] == vocab.PAD_ID  # review ngắn hơn 12 token -> có padding


def test_dataset_return_lengths(torch_mod, fake_imdb):
    from src.data.dataset import IMDBDataset, collect_samples

    tok = Tokenizer()
    vocab = _make_vocab(collect_samples(fake_imdb), tok)
    ds = IMDBDataset(tok, vocab, max_length=4, split_dir=fake_imdb, return_lengths=True)
    input_ids, length, label = ds[0]
    assert input_ids.shape == (4,)
    assert length.item() == 4  # review dài hơn 4 token -> bị cắt, length = max_length


def test_empty_review_gives_all_pad_and_length_one(torch_mod, tmp_path):
    from src.data.dataset import IMDBDataset

    d = tmp_path / "train"
    (d / "pos").mkdir(parents=True)
    (d / "neg").mkdir(parents=True)
    (d / "pos" / "0_9.txt").write_text("", encoding="utf-8")
    (d / "neg" / "0_1.txt").write_text("<br /><br />", encoding="utf-8")

    tok = Tokenizer()
    vocab = Vocabulary()
    vocab.build([["x"]])
    ds = IMDBDataset(tok, vocab, max_length=5, split_dir=d, return_lengths=True)
    for i in range(len(ds)):
        ids, length, _ = ds[i]
        assert ids.tolist() == [0] * 5
        assert length.item() == 1


def test_oov_token_becomes_unk_in_dataset(torch_mod, fake_imdb):
    from src.data.dataset import IMDBDataset

    tok = Tokenizer()
    vocab = Vocabulary()
    vocab.build([["movie"]])  # chỉ biết "movie"
    ds = IMDBDataset(tok, vocab, max_length=10, split_dir=fake_imdb)
    ids = ds[0][0].tolist()
    assert vocab.UNK_ID in ids and vocab.token_to_id["movie"] in ids


def test_cache_gives_same_result(torch_mod, fake_imdb):
    from src.data.dataset import IMDBDataset, collect_samples

    tok = Tokenizer()
    vocab = _make_vocab(collect_samples(fake_imdb), tok)
    cached = IMDBDataset(tok, vocab, 10, split_dir=fake_imdb, cache=True)
    plain = IMDBDataset(tok, vocab, 10, split_dir=fake_imdb, cache=False)
    for i in range(len(cached)):
        first = cached[i][0].tolist()
        assert first == cached[i][0].tolist()  # lần 2 đọc từ cache
        assert first == plain[i][0].tolist()  # không cache cho cùng kết quả


def test_dataset_argument_errors(torch_mod, fake_imdb, tmp_path):
    from src.data.dataset import IMDBDataset

    tok, vocab = Tokenizer(), Vocabulary()
    with pytest.raises(FileNotFoundError):
        IMDBDataset(tok, vocab, 10, split_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        IMDBDataset(tok, vocab, 10, split_dir=tmp_path)  # thiếu pos/neg
    with pytest.raises(ValueError):
        IMDBDataset(tok, vocab, 10)  # thiếu cả split_dir lẫn samples
    with pytest.raises(ValueError):
        IMDBDataset(tok, vocab, 10, split_dir=fake_imdb, samples=[])
    with pytest.raises(ValueError):
        IMDBDataset(tok, vocab, 0, split_dir=fake_imdb)


def test_dataloader_batch_shapes(torch_mod, fake_imdb):
    from src.data.dataset import IMDBDataset, collect_samples, create_dataloader

    tok = Tokenizer()
    vocab = _make_vocab(collect_samples(fake_imdb), tok)
    ds = IMDBDataset(tok, vocab, max_length=8, split_dir=fake_imdb)
    x, y = next(iter(create_dataloader(ds, batch_size=4, shuffle=True, seed=0)))
    assert x.shape == (4, 8)
    assert y.shape == (4,)


def test_encode_text_matches_dataset(torch_mod, fake_imdb):
    """Demo/inference (encode_text) phải cho đúng kết quả như lúc train."""
    from src.data.dataset import IMDBDataset, collect_samples, encode_text, read_text

    tok = Tokenizer()
    samples = collect_samples(fake_imdb)
    vocab = _make_vocab(samples, tok)
    ds = IMDBDataset(tok, vocab, 10, samples=samples)
    ids, _ = encode_text(read_text(samples[3][0]), tok, vocab, 10)
    assert ids == ds[3][0].tolist()

    empty_ids, length = encode_text("", tok, vocab, 10)
    assert empty_ids == [0] * 10 and length == 1


def test_split_train_val_is_stratified_disjoint_and_reproducible(torch_mod, fake_imdb):
    from src.data.dataset import collect_samples, split_train_val

    samples = collect_samples(fake_imdb)
    train, val = split_train_val(samples, val_ratio=0.5, seed=1)

    assert len(train) + len(val) == len(samples)
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(samples)
    assert {y for _, y in val} == {0, 1}
    assert sum(y for _, y in val) == len(val) // 2  # cân bằng nhãn
    assert (train, val) == split_train_val(samples, val_ratio=0.5, seed=1)
    assert val != split_train_val(samples, val_ratio=0.5, seed=2)[1]

    with pytest.raises(ValueError):
        split_train_val(samples, val_ratio=1.0)


def test_vocabulary_built_from_train_only(torch_mod, fake_imdb):
    """Từ chỉ có trong validation phải thành UNK (không rò rỉ vào vocabulary)."""
    from src.data.dataset import build_vocabulary, read_text

    tok = Tokenizer()
    (fake_imdb / "pos" / "99_9.txt").write_text("zebra", encoding="utf-8")
    train = [(str(fake_imdb / "pos" / "0_9.txt"), 1)]
    val = [(str(fake_imdb / "pos" / "99_9.txt"), 1)]

    vocab = build_vocabulary(train, tok, min_freq=1)
    assert "zebra" not in vocab
    assert vocab.encode(tok.tokenize(read_text(val[0][0]))) == [vocab.UNK_ID]


def test_preprocessing_stats_and_config(torch_mod, fake_imdb, tmp_path):
    from src.data.dataset import collect_samples, preprocessing_stats, save_preprocessing_config

    tok = Tokenizer()
    samples = collect_samples(fake_imdb)
    vocab = _make_vocab(samples, tok)

    stats = preprocessing_stats(samples, tok, vocab, max_length=4)
    assert stats["num_reviews"] == 12
    assert stats["oov_rate"] == 0.0  # vocab xây từ chính các review này
    assert stats["truncated_rate"] == 1.0  # mọi review > 4 token
    assert stats["padded_rate"] == 0.0

    stats_small = preprocessing_stats(samples, tok, Vocabulary(), max_length=50)
    assert stats_small["oov_rate"] == 1.0 and stats_small["padded_rate"] == 1.0
    assert preprocessing_stats([], tok, vocab, 4) == {"num_reviews": 0}

    save_preprocessing_config(tmp_path / "cfg.json", tok, vocab, max_length=200, seed=42)
    cfg = json.loads((tmp_path / "cfg.json").read_text(encoding="utf-8"))
    assert cfg["max_length"] == 200 and cfg["seed"] == 42
    assert cfg["vocab"]["pad_id"] == 0 and cfg["label_map"] == {"neg": 0, "pos": 1}


def test_split_manifest(torch_mod, fake_imdb, tmp_path):
    from src.data.dataset import collect_samples, save_split_manifest, split_train_val

    train, val = split_train_val(collect_samples(fake_imdb), 0.25, seed=7)
    save_split_manifest(train, val, tmp_path / "m.json", seed=7, val_ratio=0.25, root=fake_imdb.parent)
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["n_train"] == len(train) and m["n_val"] == len(val) and m["seed"] == 7
    assert m["train"][0][0].startswith("train/")


def test_end_to_end_review_to_tensor(torch_mod):
    """Ví dụ yêu cầu của phân công: review -> tokens -> IDs -> padded -> tensor."""
    tok = Tokenizer()
    review = "I really loved this movie.<br />The actors were amazing!"
    tokens = tok.tokenize(review)
    vocab = Vocabulary()
    vocab.build([tokens])
    ids = vocab.encode(tokens)
    padded = pad_or_truncate(ids, max_length=20, pad_id=vocab.PAD_ID)
    tensor = torch_mod.tensor(padded, dtype=torch_mod.long)

    assert tokens[:3] == ["i", "really", "loved"]
    assert len(ids) == len(tokens) == 10  # 9 từ + dấu '!'
    assert tensor.shape == (20,)
    assert tensor[len(ids):].sum().item() == 0
