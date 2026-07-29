from __future__ import annotations

from .load import load_tiktoken_bpe

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

R50K_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


def _r50k_ranks() -> dict[bytes, int]:
    return load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    )


def gpt2() -> dict:
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        "pat_str": R50K_PATTERN,
        "mergeable_ranks": _r50k_ranks(),
        "special_tokens": {ENDOFTEXT: 50256},
    }


def r50k_base() -> dict:
    return {
        "name": "r50k_base",
        "explicit_n_vocab": 50257,
        "pat_str": R50K_PATTERN,
        "mergeable_ranks": _r50k_ranks(),
        "special_tokens": {ENDOFTEXT: 50256},
    }


def _p50k_ranks() -> dict[bytes, int]:
    return load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    )


def p50k_base() -> dict:
    return {
        "name": "p50k_base",
        "explicit_n_vocab": 50281,
        "pat_str": R50K_PATTERN,
        "mergeable_ranks": _p50k_ranks(),
        "special_tokens": {ENDOFTEXT: 50256},
    }


def p50k_edit() -> dict:
    return {
        "name": "p50k_edit",
        "pat_str": R50K_PATTERN,
        "mergeable_ranks": _p50k_ranks(),
        "special_tokens": {
            ENDOFTEXT: 50256,
            FIM_PREFIX: 50281,
            FIM_MIDDLE: 50282,
            FIM_SUFFIX: 50283,
        },
    }


def cl100k_base() -> dict:
    return {
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": load_tiktoken_bpe(
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        ),
        "special_tokens": {
            ENDOFTEXT: 100257,
            FIM_PREFIX: 100258,
            FIM_MIDDLE: 100259,
            FIM_SUFFIX: 100260,
            ENDOFPROMPT: 100276,
        },
    }


def _o200k_pattern() -> str:
    return "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )


def _o200k_ranks() -> dict[bytes, int]:
    return load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    )


def o200k_base() -> dict:
    return {
        "name": "o200k_base",
        "pat_str": _o200k_pattern(),
        "mergeable_ranks": _o200k_ranks(),
        "special_tokens": {ENDOFTEXT: 199999, ENDOFPROMPT: 200018},
    }


def o200k_harmony() -> dict:
    specials = {
        ENDOFTEXT: 199999,
        ENDOFPROMPT: 200018,
        "<|startoftext|>": 199998,
        "<|reserved_200000|>": 200000,
        "<|reserved_200001|>": 200001,
        "<|return|>": 200002,
        "<|constrain|>": 200003,
        "<|reserved_200004|>": 200004,
        "<|channel|>": 200005,
        "<|start|>": 200006,
        "<|end|>": 200007,
        "<|message|>": 200008,
        "<|reserved_200009|>": 200009,
        "<|reserved_200010|>": 200010,
        "<|reserved_200011|>": 200011,
        "<|call|>": 200012,
    }
    specials.update({f"<|reserved_{value}|>": value for value in range(200013, 201088)})
    return {
        "name": "o200k_harmony",
        "pat_str": _o200k_pattern(),
        "mergeable_ranks": _o200k_ranks(),
        "special_tokens": specials,
    }


ENCODING_CONSTRUCTORS = {
    "gpt2": gpt2,
    "r50k_base": r50k_base,
    "p50k_base": p50k_base,
    "p50k_edit": p50k_edit,
    "cl100k_base": cl100k_base,
    "o200k_base": o200k_base,
    "o200k_harmony": o200k_harmony,
}
