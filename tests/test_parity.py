from __future__ import annotations

import random

import numpy as np
import pytest
import tiktoken

import mojo_tiktoken as mtk
from mojo_tiktoken import core as mojo_core
from mojo_tiktoken import model as mojo_model
from tiktoken import model as upstream_model


@pytest.fixture(scope="module")
def cl100k():
    return mtk.get_encoding("cl100k_base"), tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def o200k():
    return mtk.get_encoding("o200k_base"), tiktoken.get_encoding("o200k_base")


def test_registered_encoding_names_match_upstream():
    assert mtk.list_encoding_names() == tiktoken.list_encoding_names()


def test_model_registry_matches_upstream():
    assert mojo_model.MODEL_TO_ENCODING == upstream_model.MODEL_TO_ENCODING
    assert mojo_model.MODEL_PREFIX_TO_ENCODING == upstream_model.MODEL_PREFIX_TO_ENCODING


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4", "cl100k_base"),
        ("gpt-4-0613", "cl100k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-5-2026-01-01", "o200k_base"),
        ("gpt-oss-120b", "o200k_harmony"),
        ("text-davinci-003", "p50k_base"),
        ("gpt2", "gpt2"),
    ],
)
def test_model_mapping_matches_upstream(model, expected):
    assert mtk.encoding_name_for_model(model) == expected
    assert mtk.encoding_name_for_model(model) == tiktoken.encoding_name_for_model(model)


def test_unknown_model_matches_upstream_error():
    with pytest.raises(KeyError):
        mtk.encoding_name_for_model("definitely-not-a-model")


def test_registry_caches_encoding_instances(cl100k):
    ours, _ = cl100k
    assert mtk.get_encoding("cl100k_base") is ours
    assert mtk.encoding_for_model("gpt-4") is ours


def test_encoding_metadata_matches_upstream(cl100k):
    ours, theirs = cl100k
    assert repr(ours) == repr(theirs)
    assert ours.name == theirs.name
    assert ours.n_vocab == theirs.n_vocab
    assert ours.max_token_value == theirs.max_token_value
    assert ours.eot_token == theirs.eot_token
    assert ours.special_tokens_set == theirs.special_tokens_set


@pytest.mark.parametrize("name", mtk.list_encoding_names())
def test_all_standard_encodings_match_upstream(name):
    ours = mtk.get_encoding(name)
    theirs = tiktoken.get_encoding(name)
    text = "A multilingual café: Ελληνικά, русский, العربية, 漢字, 𐐷.\n"
    tokens = ours.encode_ordinary(text)
    assert tokens == theirs.encode_ordinary(text)
    assert ours.decode(tokens) == theirs.decode(tokens) == text
    assert ours.special_tokens_set == theirs.special_tokens_set
    assert ours.n_vocab == theirs.n_vocab


def test_harmony_duplicate_special_value_matches_upstream():
    ours = mtk.get_encoding("o200k_harmony")
    theirs = tiktoken.get_encoding("o200k_harmony")
    for token in ["<|endofprompt|>", "<|reserved_200018|>"]:
        assert ours.encode_single_token(token) == theirs.encode_single_token(token) == 200018
    assert ours.decode_single_token_bytes(200018) == theirs.decode_single_token_bytes(200018)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "The quick brown fox jumps over 13 lazy dogs.",
        "if value >= 10:\n    print(f'{value=}')\n",
        "こんにちは世界",
        "Привет, мир",
        "مرحبا بالعالم",
        "café naïve résumé",
        "𐐷 mixed Unicode 漢字",
        "12345678901234567890",
        " \t\n\r\n   ",
        "a" * 4096,
        " " * 4096,
    ],
)
def test_cl100k_encode_ordinary_parity(cl100k, text):
    ours, theirs = cl100k
    assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)


@pytest.mark.parametrize(
    "text",
    [
        "English, Ελληνικά, русский, العربية, हिन्दी, 日本語",
        "CamelCase lower UPPER 1234567",
        "𐐷 computes 2**64 values\n",
        "\n\nmultiple\r\nline endings\r",
    ],
)
def test_o200k_encode_ordinary_parity(o200k, text):
    ours, theirs = o200k
    assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)


def test_random_raw_byte_piece_parity(cl100k):
    ours, theirs = cl100k
    random_source = random.Random(8128)
    for length in range(1, 96):
        data = bytes(random_source.randrange(256) for _ in range(length))
        assert ours._encode_single_piece(data) == theirs._encode_single_piece(data)


def test_random_unicode_parity(o200k):
    ours, theirs = o200k
    random_source = random.Random(42)
    alphabet = "abcXYZ 019!?\n\té漢𐐷Приветمرحبا"
    for _ in range(100):
        text = "".join(random_source.choice(alphabet) for _ in range(random_source.randrange(200)))
        assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)


def test_allowed_special_token_parity(cl100k):
    ours, theirs = cl100k
    text = "before<|endoftext|>after<|fim_prefix|>code"
    allowed = {"<|endoftext|>", "<|fim_prefix|>"}
    assert ours.encode(text, allowed_special=allowed) == theirs.encode(text, allowed_special=allowed)
    assert ours.encode(text, allowed_special="all") == theirs.encode(text, allowed_special="all")


def test_disallowed_special_token_parity(cl100k):
    ours, theirs = cl100k
    text = "unsafe <|endoftext|> input"
    with pytest.raises(ValueError):
        ours.encode(text)
    assert ours.encode(text, disallowed_special=()) == theirs.encode(text, disallowed_special=())


def test_unknown_allowed_special_is_ignored(cl100k):
    ours, theirs = cl100k
    assert ours.encode("ordinary text", allowed_special={"not-a-token"}, disallowed_special=()) == (
        theirs.encode("ordinary text", allowed_special={"not-a-token"}, disallowed_special=())
    )


def test_overlapping_custom_special_tokens_use_longest_match():
    ranks = {bytes([byte]): byte for byte in range(256)}
    encoding = mtk.Encoding(
        "overlapping-specials",
        pat_str=r".+",
        mergeable_ranks=ranks,
        special_tokens={"<x>": 256, "<x>long": 257},
    )
    assert encoding.encode("<x>long", allowed_special="all") == [257]


def test_encode_and_decode_batch_parity(cl100k):
    ours, theirs = cl100k
    texts = ["hello", "γειά σου", "line one\nline two", "𐐷" * 20]
    ours_tokens = ours.encode_batch(texts, num_threads=3)
    theirs_tokens = theirs.encode_batch(texts, num_threads=3)
    assert ours_tokens == theirs_tokens
    assert ours.decode_batch(ours_tokens, num_threads=2) == texts
    assert ours.decode_bytes_batch(ours_tokens, num_threads=2) == [
        text.encode() for text in texts
    ]


def test_encode_ordinary_batch_parity(o200k):
    ours, theirs = o200k
    texts = ["alpha beta", "日本語", "123 456", ""]
    assert ours.encode_ordinary_batch(texts, num_threads=2) == theirs.encode_ordinary_batch(
        texts, num_threads=2
    )


def test_encode_ordinary_batch_parallel_path(cl100k, monkeypatch):
    ours, theirs = cl100k
    texts = ["parallel document " * 100, "another independent document " * 100]
    monkeypatch.setattr(mojo_core, "_BATCH_PARALLEL_MIN_BYTES", 1)
    assert ours.encode_ordinary_batch(texts, num_threads=2) == theirs.encode_ordinary_batch(
        texts, num_threads=2
    )


def test_encode_to_numpy_parity(cl100k):
    ours, theirs = cl100k
    text = "numpy output with 漢字 and 123"
    actual = ours.encode_to_numpy(text)
    expected = theirs.encode_to_numpy(text)
    assert actual.dtype == expected.dtype == np.uint32
    assert np.array_equal(actual, expected)
    assert actual.base is not None


def test_decode_bytes_and_errors_parity(cl100k):
    ours, theirs = cl100k
    tokens = ours.encode_ordinary("A𐐷B")
    assert ours.decode_bytes(tokens) == theirs.decode_bytes(tokens)
    assert ours.decode(tokens) == theirs.decode(tokens)
    partial = [ours.encode_single_token(bytes([0xFF]))]
    assert ours.decode(partial) == theirs.decode(partial)
    with pytest.raises(UnicodeDecodeError):
        ours.decode(partial, errors="strict")
    with pytest.raises(TypeError):
        ours.decode_bytes([1.5])
    with pytest.raises(OverflowError):
        ours.decode_bytes([2**64])


def test_decode_simd_tail(cl100k):
    ours, _ = cl100k
    tokens = [
        ours.encode_single_token(b"hello"),
        ours.encode_single_token(b" world"),
        ours.encode_single_token(b"!"),
    ]
    assert ours.decode_bytes(tokens) == b"hello world!"


def test_decode_parallel_threshold(cl100k):
    ours, _ = cl100k
    token = ours.encode_single_token(b"a")
    count = mojo_core._DECODE_PARALLEL_MIN_TOKENS + 17
    assert ours.decode_bytes([token] * count) == b"a" * count


def test_decode_with_offsets_parity(cl100k):
    ours, theirs = cl100k
    tokens = ours.encode_ordinary("é𐐷 hello 漢字")
    assert ours.decode_with_offsets(tokens) == theirs.decode_with_offsets(tokens)


def test_single_token_operations_parity(cl100k):
    ours, theirs = cl100k
    for value in [b"hello", " world", "<|endoftext|>"]:
        token = ours.encode_single_token(value)
        assert token == theirs.encode_single_token(value)
        assert ours.decode_single_token_bytes(token) == theirs.decode_single_token_bytes(token)
    with pytest.raises(KeyError):
        ours.encode_single_token(b"not one vocabulary token probably")
    with pytest.raises(KeyError):
        ours.decode_single_token_bytes(2**31)


def test_decode_tokens_bytes_parity(o200k):
    ours, theirs = o200k
    tokens = ours.encode_ordinary("token boundaries: 你好𐐷")
    assert ours.decode_tokens_bytes(tokens) == theirs.decode_tokens_bytes(tokens)


def test_token_byte_values_parity(cl100k):
    ours, theirs = cl100k
    assert ours.token_byte_values() == theirs.token_byte_values()


def test_special_token_predicate_parity(cl100k):
    ours, theirs = cl100k
    for token in [ours.eot_token, ours.encode_single_token(b"a"), -1]:
        assert ours.is_special_token(token) == theirs.is_special_token(token)


def test_surrogate_replacement_parity(cl100k):
    ours, theirs = cl100k
    text = "before\ud800after"
    assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)


def test_custom_encoding_matches_upstream_and_supports_large_token_ids():
    ranks = {bytes([byte]): byte for byte in range(256)}
    ranks.update({b"ab": 256, b"bc": 257, b"abc": 3_000_000_000})
    ours = mtk.Encoding(
        "custom",
        pat_str=r"\p{L}+|\p{N}+|[^\p{L}\p{N}]+",
        mergeable_ranks=ranks,
        special_tokens={"<special>": 3_000_000_001},
    )
    theirs = tiktoken.Encoding(
        "custom",
        pat_str=r"\p{L}+|\p{N}+|[^\p{L}\p{N}]+",
        mergeable_ranks=ranks,
        special_tokens={"<special>": 3_000_000_001},
    )
    for text in ["abc", "zabc9", "abc<special>abc"]:
        allowed = "all" if "<special>" in text else set()
        assert ours.encode(text, allowed_special=allowed) == theirs.encode(
            text, allowed_special=allowed
        )


def test_constructor_rejects_incomplete_byte_vocabulary():
    with pytest.raises(ValueError, match="every single byte"):
        mtk.Encoding("bad", pat_str=r".+", mergeable_ranks={b"a": 0}, special_tokens={})


def test_constructor_enforces_explicit_vocabulary_size():
    ranks = {bytes([byte]): byte for byte in range(256)}
    with pytest.raises(AssertionError, match="explicit_n_vocab"):
        mtk.Encoding(
            "bad-size",
            pat_str=r".+",
            mergeable_ranks=ranks,
            special_tokens={},
            explicit_n_vocab=0,
        )


@pytest.mark.parametrize("rank", [-1, 2**32, 1.5, True])
def test_constructor_rejects_token_ids_native_output_cannot_represent(rank):
    ranks = {bytes([byte]): byte for byte in range(256)}
    ranks[b"ab"] = rank
    with pytest.raises((TypeError, ValueError)):
        mtk.Encoding("bad-rank", pat_str=r".+", mergeable_ranks=ranks, special_tokens={})


def test_native_encode_rejects_malformed_offsets_before_ffi(cl100k):
    ours, _ = cl100k
    valid_documents = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="nondecreasing"):
        ours._encode_packed_documents(
            b"ab",
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
        )
    with pytest.raises(TypeError, match="int64"):
        ours._encode_packed_documents(
            b"a",
            np.array([0, 1], dtype=np.int32),
            valid_documents,
        )
    with pytest.raises(ValueError, match="input byte length"):
        ours._encode_packed_documents(
            b"a",
            np.array([0, 0], dtype=np.int64),
            valid_documents,
        )
    with pytest.raises(ValueError, match="piece count"):
        ours._encode_packed_documents(
            b"a",
            np.array([0, 1], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        )


def test_batch_rejects_invalid_thread_count(cl100k):
    ours, _ = cl100k
    with pytest.raises(ValueError, match="at least 1"):
        ours.encode_ordinary_batch(["text"], num_threads=0)


def test_get_encoding_input_errors():
    with pytest.raises(ValueError):
        mtk.get_encoding("unknown")
    with pytest.raises(ValueError):
        mtk.get_encoding(123)
