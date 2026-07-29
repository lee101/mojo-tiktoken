from __future__ import annotations

import threading

from .core import Encoding
from .openai_public import ENCODING_CONSTRUCTORS

ENCODINGS: dict[str, Encoding] = {}
_LOCK = threading.RLock()


def get_encoding(encoding_name: str) -> Encoding:
    if not isinstance(encoding_name, str):
        raise ValueError(f"Expected a string in get_encoding, got {type(encoding_name)}")
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]
    with _LOCK:
        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(f"Unknown encoding {encoding_name}.")
        encoding = Encoding(**ENCODING_CONSTRUCTORS[encoding_name]())
        ENCODINGS[encoding_name] = encoding
        return encoding


def list_encoding_names() -> list[str]:
    return list(ENCODING_CONSTRUCTORS)
