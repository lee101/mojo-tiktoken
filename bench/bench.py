from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_tiktoken as mtk  # noqa: E402
import tiktoken  # noqa: E402


def best_time(function, repeats: int = 5) -> float:
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def compare(name, ours, theirs, repeats: int = 5) -> tuple[str, float, float, float]:
    expected = theirs()
    actual = ours()
    if actual != expected:
        raise AssertionError(f"benchmark outputs differ for {name}")
    ours_time = best_time(ours, repeats)
    upstream_time = best_time(theirs, repeats)
    return name, ours_time, upstream_time, upstream_time / ours_time


def main() -> None:
    ours_cl = mtk.get_encoding("cl100k_base")
    upstream_cl = tiktoken.get_encoding("cl100k_base")
    ours_o2 = mtk.get_encoding("o200k_base")
    upstream_o2 = tiktoken.get_encoding("o200k_base")

    english = (
        "Byte-pair encoding repeatedly merges the highest-priority adjacent byte pair. "
        "Regex pre-tokenization keeps contractions, numbers, punctuation, and whitespace "
        "in compatible pieces before those merges are applied.\n"
    ) * 4000
    source = (
        "def encode_piece(data: bytes, ranks: dict[bytes, int]) -> list[int]:\n"
        "    while len(parts) > 1:\n"
        "        index = min(range(len(parts) - 1), key=lambda i: ranks.get(parts[i] + parts[i + 1], INF))\n"
        "        parts[index:index + 2] = [parts[index] + parts[index + 1]]\n"
        "    return [ranks[part] for part in parts]\n"
    ) * 2500
    multilingual = (
        "English Ελληνικά русский العربية हिन्दी বাংলা 日本語 한국어 中文 "
        "café naïve 𐐷 123456789.\n"
    ) * 5000
    documents = [
        f"document {index}: " + english[index * 97 : index * 97 + 1600]
        for index in range(256)
    ]
    raw_piece = (b"antidisestablishmentarianism_and_tokenization_12345!" * 1200)

    rows = [
        compare(
            f"cl100k English encode ({len(english) / 1e6:.2f} MB)",
            lambda: ours_cl.encode_ordinary(english),
            lambda: upstream_cl.encode_ordinary(english),
        ),
        compare(
            f"cl100k source encode ({len(source) / 1e6:.2f} MB)",
            lambda: ours_cl.encode_ordinary(source),
            lambda: upstream_cl.encode_ordinary(source),
        ),
        compare(
            f"o200k multilingual encode ({len(multilingual.encode()) / 1e6:.2f} MB)",
            lambda: ours_o2.encode_ordinary(multilingual),
            lambda: upstream_o2.encode_ordinary(multilingual),
        ),
        compare(
            "cl100k batch encode (256 documents)",
            lambda: ours_cl.encode_ordinary_batch(documents, num_threads=8),
            lambda: upstream_cl.encode_ordinary_batch(documents, num_threads=8),
            repeats=3,
        ),
        compare(
            f"cl100k single-piece BPE ({len(raw_piece) / 1000:.0f} KB)",
            lambda: ours_cl._encode_single_piece(raw_piece),
            lambda: upstream_cl._encode_single_piece(raw_piece),
        ),
    ]

    tokens = upstream_cl.encode_ordinary(english)
    rows.append(
        compare(
            f"cl100k decode ({len(tokens) / 1000:.0f}k tokens)",
            lambda: ours_cl.decode(tokens),
            lambda: upstream_cl.decode(tokens),
        )
    )

    print(f"Machine: {platform.processor() or platform.machine()}, {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}, upstream tiktoken: {tiktoken.__version__}")
    print()
    print("| case | mojo-tiktoken | tiktoken | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_time, upstream_time, ratio in rows:
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{upstream_time * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
