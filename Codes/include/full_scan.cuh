#pragma once

#include "block_scan.cuh"
#include "warp_reduction_lookback.cuh"
#include "warp_prefetch_lookback.cuh"
#include "fencefree_lookback.cuh"

// Reuse the shared lookback::Flag enum from P1/P2 instead of defining a second
// copy inside lookback_publish.cuh (see README_P3 §5).
#define P3_USE_P1_FLAG
#include "lookback_baseline.cuh"

#include <cuda/atomic>
#include <cstdint>

namespace p2 {

/**
 * Which inter-block look-back strategy the full scan uses.
 *
 * Mapping to the assignment:
 *   Baseline      == Step 3. The "simplest version": a single thread per CTA
 *                     walks backwards through volatile Fs/As/Ps, busy-waits on
 *                     X, accumulates A, and terminates on P. Exactly
 *                     decoupled_lookback::baseline (self-publishes A and P).
 *   WarpPrefetch  == Step 4 Opt1. A warp reads up to 32 consecutive
 *                     predecessor descriptors into shared memory, then one
 *                     lane consumes the window sequentially (the global read is
 *                     the expensive part; one transaction per lane).
 *   WarpReduce    == Step 4 Opt2. All 32 lanes of the first warp poll their
 *                     own predecessor in parallel and reduce a window with
 *                     __shfl / __ballot instead of one thread reading
 *                     descriptors serially.
 *   FenceFree     == paper §4.4 "Fence-free descriptor updates": flag and
 *                     payload packed into one 64-bit word per tile, published
 *                     and read as a single store/load so no fence is needed
 *                     (for sizeof(T) <= 4; u64 falls back to the two-array
 *                     serial path).
 */
enum class ScanStrategy : int {
    Baseline      = 0,
    WarpPrefetch  = 1,
    WarpReduce    = 2,
    FenceFree     = 3
};

/**
 * Complete inclusive scan using P1's local tile scan and the selected P2/P3
 * look-back path.
 *
 * HOST CONTRACT:
 * - For nonempty input, launch one CTA per logical tile.
 * - Launch exactly B threads per CTA, with B a multiple of 32.
 * - Allocate num_tiles entries for each descriptor array.
 * - Initialize flags to X and next_tile/error to zero before launch.
 * - Keep input, output, descriptors, and counter alive through completion.
 *
 * KERNEL CONTRACT:
 * - Every thread in the CTA participates in the P1 scan and final store.
 * - The logical tile ID comes from next_tile, not blockIdx.x.
 * - Strategy selects the inter-block look-back implementation.
 * - Do not accept output if descriptors.error is nonzero.
 *
 * For n == 0, the host should skip the launch because there are no tiles.
 * Dynamic ticket assignment gives lower logical tile IDs to CTAs that start
 * first, so a look-back waiter does not depend on a later, unassigned tile.
 */
template<class T, int B, int Q, ScanStrategy Strategy>
__global__ void full_kernel(
    const T* input,
    T* output,
    uint64_t n,
    uint64_t num_tiles,
    Descriptors<T> descriptors) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& scan_scratch =
        *reinterpret_cast<step1::BlockScanScratch<T, B, Q>*>(storage);

    __shared__ uint64_t tile_id;
    __shared__ T block_prefix;
    __shared__ WarpReductionScratch<T> warp_reduce_scratch;
    __shared__ lookback::LookbackScratch<T> prefetch_scratch;

    // Only one thread requests a ticket; the barrier publishes it to the CTA.
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> next(
            *descriptors.next_tile);
        tile_id = next.fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();

    if (tile_id >= num_tiles) return;

    T items[Q];
    const T aggregate = step1::scan_local_tile<T, B, Q>(
        input, n, tile_id, items, scan_scratch);

    if constexpr (Strategy == ScanStrategy::Baseline) {
        // decoupled_lookback::baseline self-publishes A and P and returns the
        // EXCLUSIVE prefix. Call from exactly one thread (tid 0). Matches the
        // assignment's "volatile Fs/As/Ps" textbook version.
        if (threadIdx.x == 0) {
            decoupled_lookback::Descriptor<T> desc{
                descriptors.flags, descriptors.aggregates, descriptors.prefixes,
                static_cast<int>(num_tiles)};
            block_prefix = decoupled_lookback::baseline(
                desc, static_cast<int>(tile_id), aggregate);
        }
    } else if constexpr (Strategy == ScanStrategy::FenceFree) {
        if constexpr (sizeof(T) <= 4) {
            // Paper §4.4 single-word descriptors: one uint64_t holds {flag,
            // payload}. Publish and look-back each touch one word, so the flag
            // and value always arrive together and no fence is required.
            if (threadIdx.x == 0) {
                publish_aggregate_fencefree(
                    descriptors.packed, tile_id, aggregate);
                block_prefix = fencefree_lookback<T>(
                    descriptors.packed, tile_id);
                publish_prefix_fencefree(
                    descriptors.packed, tile_id,
                    static_cast<T>(block_prefix + aggregate));
            }
        } else {
            // u64: a 64-bit payload cannot share the word with the flag, so
            // fall back to the two-array serial path (identical to baseline).
            if (threadIdx.x == 0) {
                publish_aggregate(descriptors, tile_id, aggregate);
            }
            __syncwarp(0xffffffffu);
            if (threadIdx.x == 0) {
                T exclusive = T{0};
                const bool ok = serial_lookback(descriptors, tile_id, exclusive);
                block_prefix = ok ? exclusive : T{0};
                if (!ok) {
                    cuda::atomic_ref<unsigned, cuda::thread_scope_device> error(
                        *descriptors.error);
                    error.store(1u, cuda::memory_order_relaxed);
                }
                publish_prefix(descriptors, tile_id,
                               static_cast<T>(block_prefix + aggregate));
            }
        }
    } else {
        // The other strategies use the split publish protocol: caller
        // publishes A, computes the exclusive prefix, then publishes P.
        if (threadIdx.x == 0) {
            publish_aggregate(descriptors, tile_id, aggregate);
        }
        __syncwarp(0xffffffffu);

        if constexpr (Strategy == ScanStrategy::WarpPrefetch) {
            if (threadIdx.x < 32) {
                const auto result = lookback::warp_prefetch_lookback<T>(
                    descriptors.flags, descriptors.aggregates,
                    descriptors.prefixes, tile_id, prefetch_scratch);
                if (threadIdx.x == 0) {
                    block_prefix = result.success ? result.prefix : T{0};
                    if (!result.success) {
                        cuda::atomic_ref<unsigned, cuda::thread_scope_device>
                            error(*descriptors.error);
                        error.store(1u, cuda::memory_order_relaxed);
                    }
                }
            }
        } else if constexpr (Strategy == ScanStrategy::WarpReduce) {
            if (threadIdx.x < 32) {
                warp_reduction_lookback(descriptors, tile_id,
                                        warp_reduce_scratch);
            }
            if (threadIdx.x == 0) {
                block_prefix = warp_reduce_scratch.state == LookbackDone
                    ? warp_reduce_scratch.prefix : T{0};
            }
        }

        if (threadIdx.x == 0) {
            publish_prefix(descriptors, tile_id,
                           static_cast<T>(block_prefix + aggregate));
        }
    }

    // The selected look-back path writes block_prefix in thread 0 (or lane
    // 0); make it visible before every thread enters P1's collective store.
    __syncthreads();
    step1::store_tile_with_prefix<T, B, Q>(
        output, n, tile_id, block_prefix, items, scan_scratch);
}

} // namespace p2
