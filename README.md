# mojo-tiktoken

`tiktoken`-compatible byte-pair encoding with the merge engine implemented in
[Mojo](https://www.modular.com/mojo).

The package exposes the familiar Python API, uses the same Unicode regex
pre-tokenization patterns and standard OpenAI vocabularies, and produces the
same token IDs as upstream `tiktoken` for the covered API. It is independent at
runtime: upstream `tiktoken` is installed by the development environment for
parity tests and benchmarks, but `mojo_tiktoken` does not import it.

```python
import mojo_tiktoken as tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode("hello world")

assert tokens == [15339, 1917]
assert encoding.decode(tokens) == "hello world"
```

## Coverage

All seven standard encoding names from `tiktoken` 0.13 are available:
`gpt2`, `r50k_base`, `p50k_base`, `p50k_edit`, `cl100k_base`,
`o200k_base`, and `o200k_harmony`. The first use downloads the corresponding
public vocabulary, verifies its SHA-256 hash, and uses the same on-disk cache
location as upstream.

The covered public API is:

- `Encoding` with custom regex patterns, mergeable ranks, special tokens, and
  `explicit_n_vocab`
- `get_encoding`, `list_encoding_names`, `encoding_for_model`, and
  `encoding_name_for_model`
- `encode`, `encode_ordinary`, `encode_to_numpy`, `encode_batch`, and
  `encode_ordinary_batch`, including upstream-compatible allowed and
  disallowed special-token behavior
- `decode`, `decode_bytes`, their batch forms, `decode_with_offsets`,
  `decode_tokens_bytes`, `encode_single_token`, and
  `decode_single_token_bytes`
- metadata including `name`, `n_vocab`, `max_token_value`, `eot_token`,
  `special_tokens_set`, `is_special_token`, and `token_byte_values`

Not covered are the explicitly unstable `encode_with_unstable` completion API,
third-party `tiktoken_ext` plugin discovery, and upstream's pickle-by-registry
optimization. Overlapping custom special-token strings are resolved
longest-first here; upstream's hash-map-dependent ordering can choose a shorter
match. Standard vocabularies have no overlapping special strings. Regex
pre-tokenization runs in Python's `regex` package rather than in Mojo; the BPE
merge and vocabulary lookup are the compute kernels ported here.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-tiktoken.so`. The Python wrapper also
rebuilds it on import when the Mojo source is newer than the shared library.

To run the example directly:

```bash
pixi run python -c 'import mojo_tiktoken as t; e=t.get_encoding("cl100k_base"); print(e.encode("hello world"))'
```

## Performance

Measured with `pixi run bench` on this x86_64 machine, Linux
6.8.0-136-generic, Python 3.13.14, upstream `tiktoken` 0.13.0. Times are the
best of five runs, except the batch case,
which uses three. The last column is upstream time divided by Mojo time, so a
value below 1 means mojo-tiktoken is slower.

| case | mojo-tiktoken | tiktoken | upstream / Mojo |
| --- | ---: | ---: | ---: |
| cl100k English encode (0.85 MB) | 113.84 ms | 76.64 ms | 0.67x |
| cl100k source encode (0.76 MB) | 233.77 ms | 147.08 ms | 0.63x |
| o200k multilingual encode (0.73 MB) | 121.86 ms | 41.10 ms | 0.34x |
| cl100k batch encode (256 documents) | 93.75 ms | 18.10 ms | 0.19x |
| cl100k single-piece BPE (62 KB) | 16.93 ms | 17.82 ms | 1.05x |
| cl100k decode (148k tokens) | 19.08 ms | 3.58 ms | 0.19x |

The isolated BPE workload remains ahead. End-to-end encode, batch, and decode
remain slower because upstream combines regex, BPE, and result construction in
Rust. Here regex matching and Python-facing result construction remain
significant costs even though the native buffers cross FFI without copies.

These are deliberately end-to-end numbers: regex splitting, allocations, FFI,
and Python result construction are included.

No GPU path is provided. BPE hashing has a serial dependency per byte and then
performs branch-heavy table and heap operations, while decoding is a
memory-bandwidth copy. Neither has the arithmetic intensity needed to amortize
device transfers, so CPU remains the only execution path.

## How it works

Python's `regex` package splits Unicode text with the exact pattern attached to
the encoding. A large ordinary string is UTF-8 encoded once, and NumPy offset
arrays describe piece and document boundaries. Batch inputs are packed into
one contiguous byte buffer, so an ordinary batch crosses the FFI once.

The vocabulary is prepared once per `Encoding` as:

- one contiguous token-byte blob plus an `int64` offset array;
- a parallel `int64` token-rank array;
- a power-of-two open-addressed hash table containing entry indices.

The Mojo kernel hashes candidate byte ranges with FNV-1a and verifies matching
bytes before accepting a rank. BPE state is a linked list over byte positions.
A binary min-heap ordered by `(merge_rank, byte_position)` selects the next
pair, while generation counters discard stale heap entries after neighboring
merges. This gives the usual priority-ordered BPE result without repeatedly
rescanning every pair.

There is one compilation unit with C-ABI encode and decode entry points.
ctypes passes all buffers as integer addresses; the exported Mojo wrappers
reconstruct `UnsafePointer[..., AnyOrigin[mut=True]]` values internally. Python
owns the input, output, and scratch arrays, and Mojo performs no allocation
across the ABI. Token output is written as `uint32`, allowing
`encode_to_numpy` to return the native output view without conversion.

Decode uses a dense token-to-byte table when token IDs are reasonably dense.
Its byte-copy loop uses SIMD with a scalar remainder, and large token streams
are divided into independent chunks with a serial threshold to avoid thread
launch overhead. Large multi-document encodes use the same thresholded
parallel strategy with per-document scratch regions.

## Verification

The pytest suite checks exact behavioral parity against the real upstream
package across standard vocabularies, multilingual and random inputs, raw byte
pieces, special tokens, batches, decoding offsets, NumPy output, surrogates,
and custom encodings. It also covers token IDs above signed 32-bit range.

## License

MIT. The standard encoding definitions and regex patterns track the
MIT-licensed upstream `tiktoken` project.
