from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "tiktoken.mojo")
LIBRARY = os.path.join(ROOT, "dist", "libmojo-tiktoken.so")

I = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_TIKTOKEN_MOJO")
    if override:
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    raise BuildError("mojo not found; run through pixi or set MOJO_TIKTOKEN_MOJO")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIBRARY)
        and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE)
    ):
        return LIBRARY
    os.makedirs(os.path.dirname(LIBRARY), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        SOURCE,
        "-o",
        LIBRARY,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIBRARY):
        raise BuildError((result.stderr or result.stdout).strip()[:8000])
    return LIBRARY


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        encode = _library.mtk_encode_pieces
        encode.argtypes = [I] * 21
        encode.restype = I
        decode_offsets = _library.mtk_decode_offsets
        decode_offsets.argtypes = [I] * 6
        decode_offsets.restype = I
        decode_into = _library.mtk_decode_into
        decode_into.argtypes = [I] * 7
        decode_into.restype = None
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


if __name__ == "__main__":
    print(build(force="--force" in sys.argv))
