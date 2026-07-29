from std.algorithm import parallelize
from std.sys.info import simd_width_of as simdwidthof


comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def lookup_rank(
    data: BPtr,
    start: Int,
    end: Int,
    keys: BPtr,
    key_offsets: I64Ptr,
    key_ranks: I64Ptr,
    slots: I32Ptr,
    mask: Int,
) -> Int:
    var hash_value = UInt64(14695981039346656037)
    for i in range(start, end):
        hash_value = (hash_value ^ UInt64(data[i])) * UInt64(1099511628211)
    var slot = Int(hash_value & UInt64(mask))
    while True:
        var entry = Int(slots[slot])
        if entry < 0:
            return -1
        var key_start = Int(key_offsets[entry])
        var key_end = Int(key_offsets[entry + 1])
        if key_end - key_start == end - start:
            var equal = True
            for i in range(end - start):
                if keys[key_start + i] != data[start + i]:
                    equal = False
                    break
            if equal:
                return Int(key_ranks[entry])
        slot = (slot + 1) & mask


def heap_less(rank_a: Int, left_a: Int, rank_b: Int, left_b: Int) -> Bool:
    return rank_a < rank_b or (rank_a == rank_b and left_a < left_b)


def heap_push(
    left: Int,
    generation: Int,
    rank: Int,
    heap_left: I32Ptr,
    heap_generation: I32Ptr,
    heap_rank: I64Ptr,
    size: Int,
) -> Int:
    var pos = size
    while pos > 0:
        var parent = (pos - 1) // 2
        var parent_rank = Int(heap_rank[parent])
        var parent_left = Int(heap_left[parent])
        if not heap_less(rank, left, parent_rank, parent_left):
            break
        heap_left[pos] = heap_left[parent]
        heap_generation[pos] = heap_generation[parent]
        heap_rank[pos] = heap_rank[parent]
        pos = parent
    heap_left[pos] = Int32(left)
    heap_generation[pos] = Int32(generation)
    heap_rank[pos] = Int64(rank)
    return size + 1


def heap_pop(heap_left: I32Ptr, heap_generation: I32Ptr, heap_rank: I64Ptr, size: Int) -> Int:
    var new_size = size - 1
    if new_size == 0:
        return 0
    var last_left = Int(heap_left[new_size])
    var last_generation = heap_generation[new_size]
    var last_rank = Int(heap_rank[new_size])
    var pos = 0
    while True:
        var child = pos * 2 + 1
        if child >= new_size:
            break
        if child + 1 < new_size:
            var right_rank = Int(heap_rank[child + 1])
            var right_left = Int(heap_left[child + 1])
            if heap_less(right_rank, right_left, Int(heap_rank[child]), Int(heap_left[child])):
                child += 1
        if not heap_less(Int(heap_rank[child]), Int(heap_left[child]), last_rank, last_left):
            break
        heap_left[pos] = heap_left[child]
        heap_generation[pos] = heap_generation[child]
        heap_rank[pos] = heap_rank[child]
        pos = child
    heap_left[pos] = Int32(last_left)
    heap_generation[pos] = last_generation
    heap_rank[pos] = Int64(last_rank)
    return new_size


def push_pair(
    left: Int,
    data: BPtr,
    keys: BPtr,
    key_offsets: I64Ptr,
    key_ranks: I64Ptr,
    slots: I32Ptr,
    mask: Int,
    next_node: I32Ptr,
    generation: I32Ptr,
    heap_left: I32Ptr,
    heap_generation: I32Ptr,
    heap_rank: I64Ptr,
    heap_size: Int,
    piece_start: Int,
    piece_length: Int,
) -> Int:
    var right = Int(next_node[left])
    if right < 0:
        return heap_size
    var after = Int(next_node[right])
    var end = after if after >= 0 else piece_length
    var rank = lookup_rank(
        data, piece_start + left, piece_start + end, keys, key_offsets, key_ranks, slots, mask
    )
    if rank < 0:
        return heap_size
    return heap_push(
        left,
        Int(generation[left]),
        rank,
        heap_left,
        heap_generation,
        heap_rank,
        heap_size,
    )


def encode_piece(
    data: BPtr,
    piece_start: Int,
    piece_end: Int,
    keys: BPtr,
    key_offsets: I64Ptr,
    key_ranks: I64Ptr,
    slots: I32Ptr,
    mask: Int,
    tokens: U32Ptr,
    token_count: Int,
    next_node: I32Ptr,
    previous_node: I32Ptr,
    generation: I32Ptr,
    heap_left: I32Ptr,
    heap_generation: I32Ptr,
    heap_rank: I64Ptr,
) -> Int:
    var count = token_count
    var direct = lookup_rank(
        data, piece_start, piece_end, keys, key_offsets, key_ranks, slots, mask
    )
    if direct >= 0:
        tokens[count] = UInt32(direct)
        return count + 1

    var length = piece_end - piece_start
    for local in range(length):
        next_node[local] = Int32(local + 1 if local + 1 < length else -1)
        previous_node[local] = Int32(local - 1 if local > 0 else -1)
        generation[local] = 0

    var heap_size = 0
    for local in range(length - 1):
        heap_size = push_pair(
            local,
            data,
            keys,
            key_offsets,
            key_ranks,
            slots,
            mask,
            next_node,
            generation,
            heap_left,
            heap_generation,
            heap_rank,
            heap_size,
            piece_start,
            length,
        )

    while heap_size > 0:
        var left = Int(heap_left[0])
        var version = Int(heap_generation[0])
        heap_size = heap_pop(heap_left, heap_generation, heap_rank, heap_size)
        if version != Int(generation[left]):
            continue
        var right = Int(next_node[left])
        if right < 0:
            continue
        var after = Int(next_node[right])
        next_node[left] = Int32(after)
        generation[left] += 1
        generation[right] += 1
        if after >= 0:
            previous_node[after] = Int32(left)
        var before = Int(previous_node[left])
        if before >= 0:
            generation[before] += 1
            heap_size = push_pair(
                before,
                data,
                keys,
                key_offsets,
                key_ranks,
                slots,
                mask,
                next_node,
                generation,
                heap_left,
                heap_generation,
                heap_rank,
                heap_size,
                piece_start,
                length,
            )
        if after >= 0:
            heap_size = push_pair(
                left,
                data,
                keys,
                key_offsets,
                key_ranks,
                slots,
                mask,
                next_node,
                generation,
                heap_left,
                heap_generation,
                heap_rank,
                heap_size,
                piece_start,
                length,
            )

    var node = 0
    while node >= 0:
        var after = Int(next_node[node])
        var end = after if after >= 0 else length
        var rank = lookup_rank(
            data, piece_start + node, piece_start + end, keys, key_offsets, key_ranks, slots, mask
        )
        tokens[count] = UInt32(rank)
        count += 1
        node = after
    return count


def encode_document(
    document: Int,
    data: BPtr,
    piece_offsets: I64Ptr,
    document_piece_offsets: I64Ptr,
    keys: BPtr,
    key_offsets: I64Ptr,
    key_ranks: I64Ptr,
    slots: I32Ptr,
    mask: Int,
    tokens: U32Ptr,
    next_node: I32Ptr,
    previous_node: I32Ptr,
    generation: I32Ptr,
    heap_left: I32Ptr,
    heap_generation: I32Ptr,
    heap_rank: I64Ptr,
    scratch_offsets: I64Ptr,
) -> Int:
    var first_piece = Int(document_piece_offsets[document])
    var last_piece = Int(document_piece_offsets[document + 1])
    var token_start = Int(piece_offsets[first_piece])
    var token_count = token_start
    var scratch_start = Int(scratch_offsets[document])
    for piece in range(first_piece, last_piece):
        var start = Int(piece_offsets[piece])
        var end = Int(piece_offsets[piece + 1])
        if end > start:
            token_count = encode_piece(
                data,
                start,
                end,
                keys,
                key_offsets,
                key_ranks,
                slots,
                mask,
                tokens,
                token_count,
                next_node + scratch_start,
                previous_node + scratch_start,
                generation + scratch_start,
                heap_left + 3 * scratch_start,
                heap_generation + 3 * scratch_start,
                heap_rank + 3 * scratch_start,
            )
    return token_count - token_start


@export("mtk_encode_pieces")
def mtk_encode_pieces(
    data_addr: Int,
    piece_offsets_addr: Int,
    piece_count: Int,
    document_piece_offsets_addr: Int,
    document_count: Int,
    document_token_starts_addr: Int,
    document_token_counts_addr: Int,
    scratch_offsets_addr: Int,
    keys_addr: Int,
    key_offsets_addr: Int,
    key_ranks_addr: Int,
    slots_addr: Int,
    mask: Int,
    tokens_addr: Int,
    next_addr: Int,
    previous_addr: Int,
    generation_addr: Int,
    heap_left_addr: Int,
    heap_generation_addr: Int,
    heap_rank_addr: Int,
    num_threads: Int,
    parallel_threshold: Int,
) abi("C") -> Int:
    var data = BPtr(unsafe_from_address=data_addr)
    var piece_offsets = I64Ptr(unsafe_from_address=piece_offsets_addr)
    var document_piece_offsets = I64Ptr(unsafe_from_address=document_piece_offsets_addr)
    var document_token_starts = I64Ptr(unsafe_from_address=document_token_starts_addr)
    var document_token_counts = I64Ptr(unsafe_from_address=document_token_counts_addr)
    var scratch_offsets = I64Ptr(unsafe_from_address=scratch_offsets_addr)
    var keys = BPtr(unsafe_from_address=keys_addr)
    var key_offsets = I64Ptr(unsafe_from_address=key_offsets_addr)
    var key_ranks = I64Ptr(unsafe_from_address=key_ranks_addr)
    var slots = I32Ptr(unsafe_from_address=slots_addr)
    var tokens = U32Ptr(unsafe_from_address=tokens_addr)
    var next_node = I32Ptr(unsafe_from_address=next_addr)
    var previous_node = I32Ptr(unsafe_from_address=previous_addr)
    var generation = I32Ptr(unsafe_from_address=generation_addr)
    var heap_left = I32Ptr(unsafe_from_address=heap_left_addr)
    var heap_generation = I32Ptr(unsafe_from_address=heap_generation_addr)
    var heap_rank = I64Ptr(unsafe_from_address=heap_rank_addr)

    @parameter
    def encode_one(document: Int):
        var first_piece = Int(document_piece_offsets[document])
        document_token_starts[document] = piece_offsets[first_piece]
        document_token_counts[document] = Int64(
            encode_document(
                document,
                data,
                piece_offsets,
                document_piece_offsets,
                keys,
                key_offsets,
                key_ranks,
                slots,
                mask,
                tokens,
                next_node,
                previous_node,
                generation,
                heap_left,
                heap_generation,
                heap_rank,
                scratch_offsets,
            )
        )

    var data_length = Int(piece_offsets[piece_count])
    if num_threads > 1 and document_count > 1 and data_length >= parallel_threshold:
        var worker_count = min(num_threads, document_count)
        var documents_per_worker = (document_count + worker_count - 1) // worker_count

        @parameter
        def encode_chunk(worker: Int):
            var first = worker * documents_per_worker
            var last = min(first + documents_per_worker, document_count)
            for document in range(first, last):
                encode_one(document)

        parallelize[encode_chunk](worker_count, worker_count)
    else:
        for document in range(document_count):
            encode_one(document)

    var total = 0
    for document in range(document_count):
        total += Int(document_token_counts[document])
    return total


@export("mtk_decode_offsets")
def mtk_decode_offsets(
    tokens_addr: Int,
    token_count: Int,
    decode_starts_addr: Int,
    decode_lengths_addr: Int,
    decode_table_size: Int,
    output_offsets_addr: Int,
) abi("C") -> Int:
    var tokens = I64Ptr(unsafe_from_address=tokens_addr)
    var decode_starts = I64Ptr(unsafe_from_address=decode_starts_addr)
    var decode_lengths = I32Ptr(unsafe_from_address=decode_lengths_addr)
    var output_offsets = I64Ptr(unsafe_from_address=output_offsets_addr)
    var total = 0
    output_offsets[0] = 0
    for i in range(token_count):
        var token = Int(tokens[i])
        if token < 0 or token >= decode_table_size or decode_starts[token] < 0:
            return -1
        total += Int(decode_lengths[token])
        output_offsets[i + 1] = Int64(total)
    return total


def copy_decoded_tokens(
    first: Int,
    last: Int,
    tokens: I64Ptr,
    decode_data: BPtr,
    decode_starts: I64Ptr,
    decode_lengths: I32Ptr,
    output_offsets: I64Ptr,
    decoded: BPtr,
):
    comptime W = simdwidthof[DType.float64]()
    for i in range(first, last):
        var token = Int(tokens[i])
        var source_start = Int(decode_starts[token])
        var length = Int(decode_lengths[token])
        var destination_start = Int(output_offsets[i])
        var offset = 0
        while offset + W <= length:
            var value = decode_data.load[width=W](source_start + offset)
            decoded.store(destination_start + offset, value)
            offset += W
        while offset < length:
            decoded[destination_start + offset] = decode_data[source_start + offset]
            offset += 1


@export("mtk_decode_into")
def mtk_decode_into(
    tokens_addr: Int,
    token_count: Int,
    decode_data_addr: Int,
    decode_starts_addr: Int,
    decode_lengths_addr: Int,
    output_offsets_addr: Int,
    decoded_addr: Int,
    parallel_threshold: Int,
) abi("C"):
    var tokens = I64Ptr(unsafe_from_address=tokens_addr)
    var decode_data = BPtr(unsafe_from_address=decode_data_addr)
    var decode_starts = I64Ptr(unsafe_from_address=decode_starts_addr)
    var decode_lengths = I32Ptr(unsafe_from_address=decode_lengths_addr)
    var output_offsets = I64Ptr(unsafe_from_address=output_offsets_addr)
    var decoded = BPtr(unsafe_from_address=decoded_addr)
    if token_count < parallel_threshold:
        copy_decoded_tokens(
            0,
            token_count,
            tokens,
            decode_data,
            decode_starts,
            decode_lengths,
            output_offsets,
            decoded,
        )
        return

    var chunk_size = 4096
    var chunk_count = (token_count + chunk_size - 1) // chunk_size

    @parameter
    def copy_chunk(chunk: Int):
        var first = chunk * chunk_size
        var last = min(first + chunk_size, token_count)
        copy_decoded_tokens(
            first,
            last,
            tokens,
            decode_data,
            decode_starts,
            decode_lengths,
            output_offsets,
            decoded,
        )

    parallelize[copy_chunk](chunk_count, min(chunk_count, 8))
