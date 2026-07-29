from __future__ import annotations

import functools
import operator
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, Sequence

import numpy as np
import regex

from ._lib import addr, lib

SpecialArgument = Literal["all"] | AbstractSet[str]
DisallowedArgument = Literal["all"] | Collection[str]
_BATCH_PARALLEL_MIN_BYTES = 1_048_576
_DECODE_PARALLEL_MIN_TOKENS = 16_384
_MAX_NATIVE_TOKEN = np.iinfo(np.uint32).max


def _fnv1a(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


@functools.lru_cache(maxsize=128)
def _special_pattern(tokens: frozenset[str]) -> regex.Pattern:
    return regex.compile("|".join(regex.escape(token) for token in sorted(tokens, key=len, reverse=True)))


class Encoding:
    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: int | None = None,
    ):
        self.name = name
        self._pat_str = pat_str
        self._mergeable_ranks = dict(mergeable_ranks)
        self._special_tokens = dict(special_tokens)
        self._special_token_values = set(special_tokens.values())
        self._pattern = regex.compile(pat_str)

        if not self._mergeable_ranks:
            raise ValueError("mergeable_ranks must not be empty")
        if len(set(self._mergeable_ranks.values())) != len(self._mergeable_ranks):
            raise ValueError("mergeable token ranks must be unique")
        all_ranks = [*self._mergeable_ranks.values(), *self._special_tokens.values()]
        if any(
            not isinstance(rank, int) or isinstance(rank, bool)
            for rank in all_ranks
        ):
            raise TypeError("token ranks must be integers")
        if any(rank < 0 or rank > _MAX_NATIVE_TOKEN for rank in all_ranks):
            raise ValueError("token ranks must fit in an unsigned 32-bit integer")
        missing = [bytes([byte]) for byte in range(256) if bytes([byte]) not in self._mergeable_ranks]
        if missing:
            raise ValueError("mergeable_ranks must contain every single byte")

        self.max_token_value = max(
            max(self._mergeable_ranks.values()), max(self._special_tokens.values(), default=0)
        )
        if explicit_n_vocab is not None:
            if len(self._mergeable_ranks) + len(self._special_tokens) != explicit_n_vocab:
                raise AssertionError("vocabulary size does not match explicit_n_vocab")
            if self.max_token_value != explicit_n_vocab - 1:
                raise AssertionError("maximum token value does not match explicit_n_vocab")

        self._decoder = {rank: token for token, rank in self._mergeable_ranks.items()}
        overlap = self._decoder.keys() & self._special_token_values
        if overlap:
            raise ValueError(f"special and mergeable token values overlap: {sorted(overlap)[:3]}")
        for token, rank in self._special_tokens.items():
            self._decoder.setdefault(rank, token.encode("utf-8"))
        self._prepare_native_table()
        self._prepare_native_decoder()

    def _prepare_native_table(self) -> None:
        items = list(self._mergeable_ranks.items())
        offsets = np.empty(len(items) + 1, dtype=np.int64)
        offsets[0] = 0
        chunks = []
        for index, (token, _) in enumerate(items):
            chunks.append(token)
            offsets[index + 1] = offsets[index] + len(token)
        key_data = b"".join(chunks)
        self._key_bytes = np.frombuffer(key_data, dtype=np.uint8)
        self._key_offsets = offsets
        self._key_ranks = np.fromiter((rank for _, rank in items), dtype=np.int64)

        table_size = 1
        while table_size < len(items) * 2:
            table_size <<= 1
        slots = np.full(table_size, -1, dtype=np.int32)
        mask = table_size - 1
        for entry, (token, _) in enumerate(items):
            slot = _fnv1a(token) & mask
            while slots[slot] >= 0:
                slot = (slot + 1) & mask
            slots[slot] = entry
        self._slots = slots
        self._slot_mask = mask

    def _prepare_native_decoder(self) -> None:
        if self.max_token_value > max(1_000_000, len(self._decoder) * 4):
            self._decode_bytes = None
            self._decode_starts = None
            self._decode_lengths = None
            return
        starts = np.full(self.max_token_value + 1, -1, dtype=np.int64)
        lengths = np.empty(self.max_token_value + 1, dtype=np.int32)
        chunks = []
        offset = 0
        for token, token_bytes in self._decoder.items():
            starts[token] = offset
            lengths[token] = len(token_bytes)
            chunks.append(token_bytes)
            offset += len(token_bytes)
        self._decode_bytes = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        self._decode_starts = starts
        self._decode_lengths = lengths

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r}>"

    def _fix_surrogates(self, text: str) -> str:
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")

    def _encode_ordinary_native(self, text: str) -> list[int]:
        if not text:
            return []
        data_bytes, piece_offsets = self._pretokenize(text)
        result = self._encode_packed_documents(
            data_bytes,
            piece_offsets,
            np.array([0, len(piece_offsets) - 1], dtype=np.int64),
        )
        return result[0]

    def _encode_ordinary_native_numpy(self, text: str) -> np.ndarray:
        if not text:
            return np.empty(0, dtype=np.uint32)
        data_bytes, piece_offsets = self._pretokenize(text)
        result = self._encode_packed_documents(
            data_bytes,
            piece_offsets,
            np.array([0, len(piece_offsets) - 1], dtype=np.int64),
            return_numpy=True,
        )
        assert isinstance(result, np.ndarray)
        return result

    def _pretokenize(self, text: str) -> tuple[bytes, np.ndarray]:
        text = self._fix_surrogates(text)
        if self._pattern.groups:
            matched = [match.group() for match in self._pattern.finditer(text)]
        else:
            matched = self._pattern.findall(text)
        if sum(map(len, matched)) != len(text):
            raise ValueError("pre-tokenization regex did not cover the input")
        data_bytes = text.encode("utf-8")
        piece_offsets = np.empty(len(matched) + 1, dtype=np.int64)
        piece_offsets[0] = 0
        if text.isascii():
            lengths = np.fromiter(map(len, matched), dtype=np.int64, count=len(matched))
        else:
            lengths = np.fromiter(
                (len(piece.encode("utf-8")) for piece in matched),
                dtype=np.int64,
                count=len(matched),
            )
        np.cumsum(lengths, out=piece_offsets[1:])
        if piece_offsets[-1] != len(data_bytes):
            raise ValueError("pre-tokenization offsets did not cover the UTF-8 input")
        return data_bytes, piece_offsets

    def _pretokenize_pieces(self, text: str) -> list[bytes]:
        text = self._fix_surrogates(text)
        if self._pattern.groups:
            matched = [match.group() for match in self._pattern.finditer(text)]
        else:
            matched = self._pattern.findall(text)
        if sum(map(len, matched)) != len(text):
            raise ValueError("pre-tokenization regex did not cover the input")
        return [piece.encode("utf-8") for piece in matched]

    def _encode_pieces_native(self, pieces: list[bytes]) -> list[int]:
        batches = self._encode_piece_documents(pieces, [0, len(pieces)])
        return batches[0]

    def _encode_piece_documents(
        self,
        pieces: list[bytes],
        document_piece_offsets: list[int],
        *,
        num_threads: int = 1,
    ) -> list[list[int]]:
        data_bytes = b"".join(pieces)
        piece_offsets = np.empty(len(pieces) + 1, dtype=np.int64)
        piece_offsets[0] = 0
        for index, piece in enumerate(pieces):
            piece_offsets[index + 1] = piece_offsets[index] + len(piece)
        return self._encode_packed_documents(
            data_bytes,
            piece_offsets,
            np.asarray(document_piece_offsets, dtype=np.int64),
            num_threads=num_threads,
        )

    def _encode_packed_documents(
        self,
        data_bytes: bytes,
        piece_offsets: np.ndarray,
        document_pieces: np.ndarray,
        *,
        num_threads: int = 1,
        return_numpy: bool = False,
    ) -> list[list[int]] | np.ndarray:
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if piece_offsets.dtype != np.int64 or piece_offsets.ndim != 1:
            raise TypeError("piece_offsets must be a one-dimensional int64 array")
        if document_pieces.dtype != np.int64 or document_pieces.ndim != 1:
            raise TypeError("document_pieces must be a one-dimensional int64 array")
        if not piece_offsets.flags.c_contiguous or not document_pieces.flags.c_contiguous:
            raise ValueError("offset arrays must be C-contiguous")
        if len(piece_offsets) == 0 or len(document_pieces) == 0:
            raise ValueError("offset arrays must contain an initial zero")
        if piece_offsets[0] != 0 or document_pieces[0] != 0:
            raise ValueError("offset arrays must start at zero")
        if np.any(piece_offsets[1:] < piece_offsets[:-1]):
            raise ValueError("piece offsets must be nondecreasing")
        if np.any(document_pieces[1:] < document_pieces[:-1]):
            raise ValueError("document piece offsets must be nondecreasing")
        piece_count = len(piece_offsets) - 1
        if int(piece_offsets[-1]) != len(data_bytes):
            raise ValueError("piece offsets must end at the input byte length")
        if int(document_pieces[-1]) != piece_count:
            raise ValueError("document piece offsets must end at the piece count")
        document_count = len(document_pieces) - 1
        if return_numpy and document_count != 1:
            raise ValueError("NumPy output requires exactly one document")
        if not data_bytes:
            if return_numpy:
                return np.empty(0, dtype=np.uint32)
            return [[] for _ in range(document_count)]
        data = np.frombuffer(data_bytes, dtype=np.uint8)
        tokens = np.empty(len(data), dtype=np.uint32)
        scratch_offsets = np.empty(document_count + 1, dtype=np.int64)
        scratch_offsets[0] = 0
        for document in range(document_count):
            first_piece = int(document_pieces[document])
            last_piece = int(document_pieces[document + 1])
            if first_piece == last_piece:
                scratch_size = 1
            else:
                scratch_size = (
                    int(
                        np.max(
                            piece_offsets[first_piece + 1 : last_piece + 1]
                            - piece_offsets[first_piece:last_piece]
                        )
                    )
                    + 1
                )
            scratch_offsets[document + 1] = scratch_offsets[document] + scratch_size
        total_scratch = int(scratch_offsets[-1])
        next_node = np.empty(total_scratch, dtype=np.int32)
        previous_node = np.empty(total_scratch, dtype=np.int32)
        generation = np.empty(total_scratch, dtype=np.int32)
        heap_size = 3 * total_scratch
        heap_left = np.empty(heap_size, dtype=np.int32)
        heap_generation = np.empty(heap_size, dtype=np.int32)
        heap_rank = np.empty(heap_size, dtype=np.int64)
        document_token_starts = np.empty(document_count, dtype=np.int64)
        document_token_counts = np.empty(document_count, dtype=np.int64)

        count = lib().mtk_encode_pieces(
            addr(data),
            addr(piece_offsets),
            len(piece_offsets) - 1,
            addr(document_pieces),
            document_count,
            addr(document_token_starts),
            addr(document_token_counts),
            addr(scratch_offsets),
            addr(self._key_bytes),
            addr(self._key_offsets),
            addr(self._key_ranks),
            addr(self._slots),
            self._slot_mask,
            addr(tokens),
            addr(next_node),
            addr(previous_node),
            addr(generation),
            addr(heap_left),
            addr(heap_generation),
            addr(heap_rank),
            num_threads,
            _BATCH_PARALLEL_MIN_BYTES,
        )
        if count < 0 or count > len(tokens):
            raise RuntimeError(f"Mojo encoder returned invalid token count {count}")
        if np.any(document_token_starts < 0) or np.any(document_token_counts < 0):
            raise RuntimeError("Mojo encoder returned negative output bounds")
        if np.any(document_token_starts + document_token_counts > len(tokens)):
            raise RuntimeError("Mojo encoder returned out-of-range output bounds")
        if int(document_token_counts.sum()) != count:
            raise RuntimeError("Mojo encoder returned inconsistent token counts")
        if return_numpy:
            start = int(document_token_starts[0])
            end = start + int(document_token_counts[0])
            return tokens[start:end]
        return [
            tokens[
                document_token_starts[index] : (
                    document_token_starts[index] + document_token_counts[index]
                )
            ].tolist()
            for index in range(document_count)
        ]

    def encode_ordinary(self, text: str) -> list[int]:
        return self._encode_ordinary_native(text)

    def encode(
        self,
        text: str,
        *,
        allowed_special: SpecialArgument = set(),
        disallowed_special: DisallowedArgument = "all",
    ) -> list[int]:
        allowed = self.special_tokens_set if allowed_special == "all" else set(allowed_special)
        allowed &= self.special_tokens_set
        disallowed = self.special_tokens_set - allowed if disallowed_special == "all" else set(disallowed_special)
        if disallowed:
            match = _special_pattern(frozenset(disallowed)).search(text)
            if match:
                raise ValueError(
                    f"Encountered text corresponding to disallowed special token {match.group()!r}."
                )
        if not allowed:
            return self._encode_ordinary_native(text)

        result: list[int] = []
        position = 0
        for match in _special_pattern(frozenset(allowed)).finditer(text):
            result.extend(self._encode_ordinary_native(text[position : match.start()]))
            result.append(self._special_tokens[match.group()])
            position = match.end()
        result.extend(self._encode_ordinary_native(text[position:]))
        return result

    def encode_to_numpy(
        self,
        text: str,
        *,
        allowed_special: SpecialArgument = set(),
        disallowed_special: DisallowedArgument = "all",
    ) -> np.ndarray:
        allowed = self.special_tokens_set if allowed_special == "all" else set(allowed_special)
        allowed &= self.special_tokens_set
        disallowed = (
            self.special_tokens_set - allowed
            if disallowed_special == "all"
            else set(disallowed_special)
        )
        if disallowed:
            match = _special_pattern(frozenset(disallowed)).search(text)
            if match:
                raise ValueError(
                    f"Encountered text corresponding to disallowed special token {match.group()!r}."
                )
        if not allowed:
            return self._encode_ordinary_native_numpy(text)
        return np.asarray(
            self.encode(
                text,
                allowed_special=allowed,
                disallowed_special=(),
            ),
            dtype=np.uint32,
        )

    def encode_ordinary_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        pieces: list[bytes] = []
        document_piece_offsets = [0]
        for document in text:
            if document:
                pieces.extend(self._pretokenize_pieces(document))
            document_piece_offsets.append(len(pieces))
        return self._encode_piece_documents(
            pieces,
            document_piece_offsets,
            num_threads=num_threads,
        )

    def encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: SpecialArgument = set(),
        disallowed_special: DisallowedArgument = "all",
    ) -> list[list[int]]:
        encoder = functools.partial(
            self.encode,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(encoder, text))

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        if isinstance(text_or_bytes, str):
            if text_or_bytes in self._special_tokens:
                return self._special_tokens[text_or_bytes]
            text_or_bytes = text_or_bytes.encode("utf-8")
        try:
            return self._mergeable_ranks[text_or_bytes]
        except KeyError:
            for token, value in self._special_tokens.items():
                if token.encode("utf-8") == text_or_bytes:
                    return value
            raise

    def _encode_single_piece(self, text_or_bytes: str | bytes) -> list[int]:
        if isinstance(text_or_bytes, str):
            text_or_bytes = self._fix_surrogates(text_or_bytes).encode("utf-8")
        return self._encode_pieces_native([text_or_bytes])

    def _encode_bytes(self, text: bytes) -> list[int]:
        return self._encode_pieces_native([text])

    def decode_single_token_bytes(self, token: int) -> bytes:
        try:
            return self._decoder[token]
        except KeyError:
            raise KeyError(token) from None

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:
        return [self.decode_single_token_bytes(token) for token in tokens]

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        if not tokens:
            return b""
        if self._decode_bytes is None:
            return b"".join(map(self._decoder.__getitem__, tokens))
        token_array = np.fromiter(
            (operator.index(token) for token in tokens),
            dtype=np.int64,
            count=len(tokens),
        )
        output_offsets = np.empty(len(token_array) + 1, dtype=np.int64)
        decoded_length = lib().mtk_decode_offsets(
            addr(token_array),
            len(token_array),
            addr(self._decode_starts),
            addr(self._decode_lengths),
            len(self._decode_starts),
            addr(output_offsets),
        )
        if decoded_length < 0:
            for token in tokens:
                if token not in self._decoder:
                    raise KeyError(token)
            raise RuntimeError("Mojo decoder rejected valid tokens")
        decoded = np.empty(decoded_length, dtype=np.uint8)
        if decoded_length:
            lib().mtk_decode_into(
                addr(token_array),
                len(token_array),
                addr(self._decode_bytes),
                addr(self._decode_starts),
                addr(self._decode_lengths),
                addr(output_offsets),
                addr(decoded),
                _DECODE_PARALLEL_MIN_TOKENS,
            )
        return decoded.tobytes()

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_with_offsets(self, tokens: Sequence[int]) -> tuple[str, list[int]]:
        token_bytes = self.decode_tokens_bytes(tokens)
        text_length = 0
        offsets = []
        for token in token_bytes:
            offsets.append(max(0, text_length - (0x80 <= token[0] < 0xC0)))
            text_length += sum(not 0x80 <= byte < 0xC0 for byte in token)
        return b"".join(token_bytes).decode("utf-8", errors="strict"), offsets

    def decode_batch(
        self,
        batch: Sequence[Sequence[int]],
        *,
        errors: str = "replace",
        num_threads: int = 8,
    ) -> list[str]:
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(decoder, batch))

    def decode_bytes_batch(
        self, batch: Sequence[Sequence[int]], *, num_threads: int = 8
    ) -> list[bytes]:
        with ThreadPoolExecutor(num_threads) as executor:
            return list(executor.map(self.decode_bytes, batch))

    def token_byte_values(self) -> list[bytes]:
        return sorted(self._mergeable_ranks)

    @property
    def special_tokens_set(self) -> set[str]:
        return set(self._special_tokens)

    @property
    def eot_token(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @property
    def n_vocab(self) -> int:
        return self.max_token_value + 1

    def is_special_token(self, token: int) -> bool:
        assert isinstance(token, int)
        return token in self._special_token_values
