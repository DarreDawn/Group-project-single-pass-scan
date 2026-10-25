#pragma once

#include "warp_prefetch.cuh"

#include <cuda/atomic>
#include <cstdint>
#include <type_traits>

namespace p2 {

// Device-side state shared by every CTA in one scan launch.
//
// CALLING CONTRACT:
// - Each descriptor array has at least num_tiles entries.
// - flags is initialized to Flag::X before the scan begins.
// - next_tile and error are initialized to zero before the scan begins.
// - A scan launch owns these arrays until all of its CTAs have completed.
template<class T>
struct Descriptors {
    lookback::Flag* flags;
    T* aggregates;
    T* prefixes;
    unsigned* next_tile;
    unsigned* error;
};

// Shared scratch for the first warp's look-back loop. Only lane 0 updates
// prefix, end_exclusive, and state; the warp synchronizes before consuming
// those updates.
template<class T>
struct WarpReductionScratch {
    T prefix;
    uint64_t end_exclusive;
    unsigned state;
};

enum : unsigned {
    LookbackDone = 0,
    LookbackRetry = 1,
    LookbackInvalid = 2
};

// Inclusive warp reduction. Every lane in a full warp must call this function
// with the same active mask. The result is useful in lane 0; additions wrap
// modulo the width of T, matching the scan operator.
template<class T>
__device__ __forceinline__ T warp_sum(T value) {
    using Word = std::conditional_t<(sizeof(T) <= 4),
                                    unsigned int, unsigned long long>;
    constexpr unsigned mask = 0xffffffffu;
    #pragma unroll
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        const Word other = __shfl_down_sync(
            mask, static_cast<Word>(value), offset);
        value = static_cast<T>(value + static_cast<T>(other));
    }
    return value;
}

// Read the descriptor status with acquire semantics. A producer publishes its
// payload before its matching release store to this flag.
__device__ __forceinline__ lookback::Flag load_flag(lookback::Flag* flags,
                                                     uint64_t index) {
    cuda::atomic_ref<lookback::Flag, cuda::thread_scope_device> flag(flags[index]);
    return flag.load(cuda::memory_order_acquire);
}

// Publish A only after the aggregate has been written. The aggregate must not
// be modified after this release store.
template<class T>
__device__ __forceinline__ void publish_aggregate(
    Descriptors<T> descriptors, uint64_t tile_id, T aggregate) {
    descriptors.aggregates[tile_id] = aggregate;
    cuda::atomic_ref<lookback::Flag, cuda::thread_scope_device> flag(
        descriptors.flags[tile_id]);
    flag.store(lookback::Flag::A, cuda::memory_order_release);
}

// Publish P only after the inclusive prefix has been written. The prefix must
// not be modified after this release store.
template<class T>
__device__ __forceinline__ void publish_prefix(
    Descriptors<T> descriptors, uint64_t tile_id, T prefix) {
    descriptors.prefixes[tile_id] = prefix;
    cuda::atomic_ref<lookback::Flag, cuda::thread_scope_device> flag(
        descriptors.flags[tile_id]);
    flag.store(lookback::Flag::P, cuda::memory_order_release);
}

// Acquire one predecessor's status and read only the payload selected by it.
// An X state deliberately reads no payload; its caller must retry rather than
// treating the default zero as a contribution.
template<class T>
__device__ __forceinline__ void acquire_predecessor(
    Descriptors<T> descriptors,
    uint64_t index,
    lookback::Flag& flag,
    T& value) {
    flag = load_flag(descriptors.flags, index);
    value = T{0};
    if (flag == lookback::Flag::A) {
        value = descriptors.aggregates[index];
    } else if (flag == lookback::Flag::P) {
        value = descriptors.prefixes[index];
    }
}

// Single-thread reference path. Call from one thread per CTA after the CTA's
// aggregate has been published as A. Returns the exclusive prefix and
// publishes no descriptor itself. Returns false only for a descriptor value
// outside X/A/P.
//
// A required X is polled until its producer publishes. The caller must use a
// tile-assignment scheme that lets all lower logical tile IDs make progress.
template<class T>
__device__ __forceinline__ bool serial_lookback(
    Descriptors<T> descriptors, uint64_t tile_id, T& exclusive_prefix) {
    T sum = T{0};
    if (tile_id != 0) {
        uint64_t index = tile_id - 1;
        while (true) {
            lookback::Flag flag;
            T value;
            acquire_predecessor(descriptors, index, flag, value);
            if (flag == lookback::Flag::P) {
                sum = static_cast<T>(sum + value);
                break;
            }
            if (flag == lookback::Flag::A) {
                sum = static_cast<T>(sum + value);
                if (index == 0) break;
                --index;
                continue;
            }
            if (flag != lookback::Flag::X) return false;
        }
    }
    exclusive_prefix = sum;
    return true;
}

/**
 * Compute the exclusive prefix of tile_id using a warp-wide look-back.
 *
 * CALLING CONTRACT:
 * - Called collectively by exactly the first warp of a one-dimensional CTA.
 * - All 32 lanes must pass identical descriptors, tile_id, and scratch.
 * - The CTA must publish its aggregate as A before calling this function.
 * - Descriptor arrays must remain valid and published payloads immutable
 *   until every CTA in the scan has completed.
 * - A required X is retried until its producer publishes; it is not a
 *   zero-valued contribution.
 *
 * ON RETURN:
 * - lane 0 owns the final prefix and state.
 * - LookbackDone means prefix is valid.
 * - LookbackInvalid means an invalid descriptor was observed and error was
 *   set; the caller must not accept prefix as a successful result.
 *
 * Window layout is nearest predecessor in lane 0, then progressively older
 * predecessors through lane 31. A P terminates at the nearest available
 * inclusive prefix; older entries are not part of the result.
 */
template<class T>
__device__ __forceinline__ void warp_reduction_lookback(
    Descriptors<T> descriptors,
    uint64_t tile_id,
    WarpReductionScratch<T>& scratch) {
    constexpr unsigned mask = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;

    if (lane == 0) {
        scratch.prefix = T{0};
        scratch.end_exclusive = tile_id;
        scratch.state = LookbackRetry;
    }
    __syncwarp(mask);

    while (scratch.state == LookbackRetry) {
        const uint64_t end = scratch.end_exclusive;
        const unsigned valid_count = lookback::window_size(end);
        const bool valid = lane < valid_count;

        lookback::Flag observed = lookback::Flag::X;
        T value = T{0};
        if (valid) {
            const uint64_t index = end - 1 - lane;
            acquire_predecessor(descriptors, index, observed, value);
        }

        const unsigned unresolved = __ballot_sync(
            mask, valid && observed != lookback::Flag::A);

        if (unresolved == 0) {
            const T window_sum = warp_sum(value);
            if (lane == 0) {
                scratch.prefix = static_cast<T>(scratch.prefix + window_sum);
                scratch.end_exclusive -= valid_count;
                if (scratch.end_exclusive == 0) {
                    scratch.state = LookbackDone;
                }
            }
        } else {
            const unsigned terminal_lane = static_cast<unsigned>(__ffs(unresolved) - 1);
            const unsigned terminal_flag = __shfl_sync(
                mask, static_cast<unsigned>(observed), terminal_lane);

            if (terminal_flag == static_cast<unsigned>(lookback::Flag::P)) {
                const T contribution = valid && lane < terminal_lane
                    ? value
                    : (lane == terminal_lane ? value : T{0});
                const T window_sum = warp_sum(contribution);
                if (lane == 0) {
                    scratch.prefix = static_cast<T>(scratch.prefix + window_sum);
                    scratch.state = LookbackDone;
                }
            } else if (terminal_flag != static_cast<unsigned>(lookback::Flag::X)) {
                if (lane == 0) {
                    scratch.state = LookbackInvalid;
                    cuda::atomic_ref<unsigned, cuda::thread_scope_device> error(
                        *descriptors.error);
                    error.store(1u, cuda::memory_order_relaxed);
                }
            }
        }

        __syncwarp(mask);
    }
}

} // namespace p2
