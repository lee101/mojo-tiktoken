from __future__ import annotations

from .core import Encoding
from .registry import get_encoding

MODEL_PREFIX_TO_ENCODING = {
    "o1-": "o200k_base",
    "o3-": "o200k_base",
    "o4-mini-": "o200k_base",
    "gpt-5-": "o200k_base",
    "gpt-4.5-": "o200k_base",
    "gpt-4.1-": "o200k_base",
    "chatgpt-4o-": "o200k_base",
    "gpt-4o-": "o200k_base",
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "gpt-35-turbo-": "cl100k_base",
    "gpt-oss-": "o200k_harmony",
    "ft:gpt-4o": "o200k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
    "ft:davinci-002": "cl100k_base",
    "ft:babbage-002": "cl100k_base",
}

MODEL_TO_ENCODING = {
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-5": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    "gpt2": "gpt2",
    "gpt-2": "gpt2",
}


def encoding_name_for_model(model_name: str) -> str:
    if model_name in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model_name]
    for prefix, encoding_name in MODEL_PREFIX_TO_ENCODING.items():
        if model_name.startswith(prefix):
            return encoding_name
    raise KeyError(
        f"Could not automatically map {model_name} to a tokeniser. "
        "Please use `mojo_tiktoken.get_encoding` to explicitly get the tokeniser you expect."
    ) from None


def encoding_for_model(model_name: str) -> Encoding:
    return get_encoding(encoding_name_for_model(model_name))
