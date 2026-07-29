from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import urllib.request
import uuid


def _read_file(path: str) -> bytes:
    if "://" not in path:
        with open(path, "rb", buffering=0) as file:
            return file.read()
    with urllib.request.urlopen(path) as response:
        return response.read()


def read_file_cached(path: str, expected_hash: str | None = None) -> bytes:
    cache_dir = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get("DATA_GYM_CACHE_DIR", os.path.join(tempfile.gettempdir(), "data-gym-cache")),
    )
    if not cache_dir:
        data = _read_file(path)
    else:
        cache_path = os.path.join(cache_dir, hashlib.sha1(path.encode()).hexdigest())
        if os.path.exists(cache_path):
            with open(cache_path, "rb", buffering=0) as file:
                data = file.read()
            if expected_hash is None or hashlib.sha256(data).hexdigest() == expected_hash:
                return data
        data = _read_file(path)
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Hash mismatch for data downloaded from {path}")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = f"{cache_path}.{uuid.uuid4()}.tmp"
        with open(temporary, "wb") as file:
            file.write(data)
        os.replace(temporary, cache_path)
    return data


def load_tiktoken_bpe(path: str, expected_hash: str | None = None) -> dict[bytes, int]:
    ranks = {}
    for line in read_file_cached(path, expected_hash).splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks
