#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// P1 / Step1 only: inclusive addition inside one logical tile.
// Arithmetic is modulo 2^(sizeof(T)*8), with identical input/output/accumulator
// types. Supported T: uint8_t, uint16_t, uint32_t, uint64_t.
// This file deliberately contains no dynamic counter and no look-back protocol.
namespace step1 {

template<class T, int B, int Q>
struct BlockScanScratch {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T> &&
                  !std::is_same_v<T, bool> && sizeof(T) <= 8,
                  "Use an unsigned integer type of at most 64 bits");
    static_assert(B >= 32 && B <= 1024 && B % 32 == 0,
                  "B must be a multiple of 32 in [32, 1024]");
    static_assert(Q > 0 && Q <= 1024, "Q must be in [1, 1024]");

    static constexpr int tile_size = B * Q;
    static constexpr int warp_count = B / 32;

    T tile[tile_size];                    // Global/shared/register transposition.
    T warp_prefixes[warp_count];          // Inclusive sums of complete warps.
};

namespace detail {

template<class T>
__device__ __forceinline__ T add(T left, T right) {
    // The cast also makes uint8_t/uint16_t arithmetic explicitly modular.
    return static_cast<T>(left + right);
}

// Collective: all 32 lanes of this warp must execute every shuffle.
// Narrow types are promoted because CUDA shuffle moves 32- or 64-bit words.
template<class T>
__device__ __forceinline__ T warp_inclusive_sum(T value) {
    using Word = std::conditional_t<(sizeof(T) <= 4),
                                    unsigned int, unsigned long long>;
    const unsigned lane = threadIdx.x & 31u;
    #pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        const Word previous = __shfl_up_sync(
            0xffffffffu, static_cast<Word>(value), offset);
        if (lane >= offset) {
            value = add(static_cast<T>(previous), value);
        }
    }
    return value;
}

} // namespace detail

/**
 * Compute a block-local inclusive scan, retaining output in per-thread items[].
 *
 * CALLING CONTRACT FOR P2/P3:
 * - Launch a one-dimensional block with exactly B threads.
 * - Every thread must call this function, including threads without valid input.
 * - input is the base of the ENTIRE input array; n is its total element count.
 * - tile_id is a LOGICAL tile index supplied by the caller. P2 will supply the
 *   dynamic index; this function never reads blockIdx.x.
 * - The caller assigns every valid tile to exactly one block.
 * - items[] is private to each thread; scratch is shared by the whole block.
 * - Input indices must fit in uint64_t. For an actual scan, require
 *   tile_id < ceil(n/(B*Q)); the host skips the launch when n == 0.
 *
 * ON RETURN:
 * - items[q] is the inclusive prefix WITHIN THIS TILE, for the element at
 *   tile_id*(B*Q) + threadIdx.x*Q + q. Invalid padded positions must not be stored.
 * - The returned value is the tile's aggregate, valid in EVERY thread.
 * - No preceding tile's prefix has been added, and no output/status is published.
 * - All shared-memory reads are finished; scratch may be reused after return.
 *
 * INTEGRATION:
 * P3 publishes the returned aggregate and obtains an EXCLUSIVE block prefix.
 * Keep items[] alive during look-back, broadcast that prefix to the block, then
 * have EVERY thread call store_tile_with_prefix(). Do not write local output to
 * global memory and read it back just to add the prefix.
 */
template<class T, int B, int Q>
__device__ __forceinline__ T scan_local_tile(
    const T* input,
    uint64_t n,
    uint64_t tile_id,
    T (&items)[Q],
    BlockScanScratch<T, B, Q>& scratch
) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    constexpr unsigned warps = B / 32;
    const uint64_t base = tile_id * static_cast<uint64_t>(B) * Q;

    // 1. Striped, coalesced global reads. Shared memory retains input order.
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        const unsigned local = static_cast<unsigned>(q * B) + tid;
        const uint64_t global = base + local;
        scratch.tile[local] = global < n ? input[global] : T{0};
    }
    __syncthreads(); // All loads must complete before other threads consume them.

    // 2. Each thread scans its CONTIGUOUS Q-element range in register candidates.
    // Do not change this indexing to q*B+tid: it would change the scan order.
    T thread_total = 0;
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        thread_total = detail::add(thread_total, scratch.tile[tid * Q + q]);
        items[q] = thread_total;
    }

    // 3. Scan the thread totals: first within each warp, then across warps.
    const T within_warp = detail::warp_inclusive_sum(thread_total);
    if (lane == 31) {
        scratch.warp_prefixes[warp] = within_warp;
    }
    __syncthreads(); // Publishes warp totals; also ends all reads of scratch.tile.

    if (warp == 0) {
        // Every lane participates. Nonexistent warps contribute the identity.
        T warp_total = lane < warps ? scratch.warp_prefixes[lane] : T{0};
        warp_total = detail::warp_inclusive_sum(warp_total);
        if (lane < warps) {
            scratch.warp_prefixes[lane] = warp_total;
        }
    }
    __syncthreads(); // Every warp may now read the prefix of earlier warps.

    // Shift each warp's inclusive result by one lane to get a thread-exclusive
    // prefix; lane 0 uses zero. Then prepend the prefix of earlier warps.
    using Word = std::conditional_t<(sizeof(T) <= 4),
                                    unsigned int, unsigned long long>;
    const Word previous = __shfl_up_sync(
        0xffffffffu, static_cast<Word>(within_warp), 1);
    const T earlier_threads = lane == 0 ? T{0} : static_cast<T>(previous);
    const T earlier_warps = warp == 0 ? T{0} : scratch.warp_prefixes[warp - 1];
    const T thread_prefix = detail::add(earlier_warps, earlier_threads);
    const T aggregate = scratch.warp_prefixes[warps - 1];

    // 4. Convert thread-local scans into block-local scans.
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        items[q] = detail::add(thread_prefix, items[q]);
    }
    __syncthreads(); // No thread may reuse scratch before all readers finish.
    return aggregate;
}

/**
 * Apply the EXCLUSIVE prefix of earlier blocks and store final inclusive output.
 *
 * Collective: every thread in the B-thread block must call this function.
 * output is the base of the entire output array, NOT a pre-offset tile pointer.
 * tile_id/n must match scan_local_tile(). block_prefix must be equal in every
 * thread; the integrated caller is responsible for broadcasting it beforehand.
 * Pass zero in the standalone Step1 test to keep every tile independent.
 * scratch must be ready for reuse. Only valid indices are written.
 * Input/output overlap is outside this module's documented/tested contract.
 */
template<class T, int B, int Q>
__device__ __forceinline__ void store_tile_with_prefix(
    T* output,
    uint64_t n,
    uint64_t tile_id,
    T block_prefix,
    const T (&items)[Q],
    BlockScanScratch<T, B, Q>& scratch
) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = tile_id * static_cast<uint64_t>(B) * Q;

    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        scratch.tile[tid * Q + q] = detail::add(block_prefix, items[q]);
    }
    __syncthreads(); // Readers below may consume values written by other warps.

    // Transpose back to a coalesced global-store access pattern.
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        const unsigned local = static_cast<unsigned>(q * B) + tid;
        const uint64_t global = base + local;
        if (global < n) {
            output[global] = scratch.tile[local];
        }
    }
    __syncthreads(); // Safe for the caller to reuse scratch after this returns.
}

/**
 * Standalone Step1 kernel (no look-back). Launch with B threads and dynamic
 * shared bytes == sizeof(BlockScanScratch<T,B,Q>). Host skips n == 0.
 * block_totals may be nullptr; otherwise it has ceil(n/(B*Q)) elements.
 * These static block IDs belong to this testable wrapper, not the core module.
 */
template<class T, int B, int Q>
__global__ void block_local_scan_kernel(
    const T* input, T* output, T* block_totals, uint64_t n
) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& scratch = *reinterpret_cast<BlockScanScratch<T, B, Q>*>(storage);
    T items[Q];

    //P2: Ensure that all threads within the block are assigned the same tile_id
    const uint64_t tile_id = blockIdx.x;
    const T aggregate = scan_local_tile<T, B, Q>(input, n, tile_id, items, scratch); 
    if (threadIdx.x == 0 && block_totals != nullptr) {
        block_totals[tile_id] = aggregate;
    }

    // Standalone Step1 test: each tile is scanned independently,
    // so pass T{0} to write only the tile-local inclusive scan.
    //
    // When integrating into full_kernel, P2/P3 should pass the exclusive
    // block prefix computed by look-back at the corresponding call site.
    // All threads in the block must receive the same prefix before calling.
    // The addition and output-store logic inside store_tile_with_prefix
    // requires no changes.
    store_tile_with_prefix<T, B, Q>(output, n, tile_id, T{0}, items, scratch);
}

} // namespace step1
