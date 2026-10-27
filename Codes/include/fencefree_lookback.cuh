#pragma once

#include "warp_prefetch.cuh"             // lookback::Flag
#include "warp_reduction_lookback.cuh"   // p2::Descriptors, serial_lookback,
                                         // publish_aggregate, publish_prefix

#include <cstdint>

namespace p2 {

// Fence-free single-word descriptors (paper §4.4, "Fence-free descriptor
// updates").
//
// The status flag and its payload are packed into one 64-bit word:
//   bits [0..1] : status (X=0, A=1, P=2)
//   bits [2..]  : payload (tile aggregate when A, inclusive prefix when P)
//
// A producer publishes with a single naturally-aligned 64-bit store, and a
// consumer reads with a single 64-bit load, so the flag and its payload always
// arrive together. This replaces the "write payload -> fence -> write flag"
// sequence of the two-array path, so no fence is needed for a consistent view.
//
// The packing is valid only when sizeof(T) <= 4 (value << 2 must fit in 64
// bits). For u64 the caller must fall back to the two-array Descriptors path.
__device__ __forceinline__ std::uint64_t pack_word(
    std::uint64_t value, lookback::Flag flag) {
    return (value << 2) | static_cast<std::uint64_t>(flag);
}

// Publish the tile aggregate with a single-word store (flag set to A).
template<class T>
__device__ __forceinline__ void publish_aggregate_fencefree(
    std::uint64_t* packed, std::uint64_t tile_id, T aggregate) {
    auto* word = reinterpret_cast<volatile std::uint64_t*>(packed + tile_id);
    *word = pack_word(static_cast<std::uint64_t>(aggregate), lookback::Flag::A);
}

// Publish the inclusive prefix with a single-word store (flag set to P).
template<class T>
__device__ __forceinline__ void publish_prefix_fencefree(
    std::uint64_t* packed, std::uint64_t tile_id, T prefix) {
    auto* word = reinterpret_cast<volatile std::uint64_t*>(packed + tile_id);
    *word = pack_word(static_cast<std::uint64_t>(prefix), lookback::Flag::P);
}

// Serial decoupled look-back over single-word descriptors. Call from exactly
// one thread. Mirrors serial_lookback: walk from tile-1 back toward 0;
// A accumulates, P terminates, X busy-waits. Returns the exclusive prefix of
// tile_id.
template<class T>
__device__ __forceinline__ T fencefree_lookback(
    const std::uint64_t* packed, std::uint64_t tile_id) {
    T exclusive = T{0};
    if (tile_id != 0) {
        std::uint64_t index = tile_id - 1;
        while (true) {
            const std::uint64_t word =
                reinterpret_cast<const volatile std::uint64_t*>(packed)[index];
            const unsigned flag = static_cast<unsigned>(word & 3u);
            const T value = static_cast<T>(word >> 2);
            if (flag == static_cast<unsigned>(lookback::Flag::P)) {
                exclusive = static_cast<T>(exclusive + value);
                break;
            }
            if (flag == static_cast<unsigned>(lookback::Flag::A)) {
                exclusive = static_cast<T>(exclusive + value);
                if (index == 0) break;
                --index;
                continue;
            }
            // X: poll until the producer publishes a valid status.
        }
    }
    return exclusive;
}

} // namespace p2
