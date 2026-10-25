#pragma once

#include "block_scan.cuh"
#include "warp_reduction_lookback.cuh"

#include <cuda/atomic>
#include <cstdint>

namespace p2 {

/**
 * Complete inclusive scan using P1's local tile scan and a P2 look-back path.
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
 * - UseWarpReduction selects the first-warp Opt2 path when true and the
 *   single-thread serial reference when false.
 * - Do not accept output if descriptors.error is nonzero.
 *
 * For n == 0, the host should skip the launch because there are no tiles.
 * Dynamic ticket assignment gives lower logical tile IDs to CTAs that start
 * first, so a look-back waiter does not depend on a later, unassigned tile.
 */
template<class T, int B, int Q, bool UseWarpReduction>
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
    __shared__ WarpReductionScratch<T> lookback_scratch;

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

    if constexpr (UseWarpReduction) {
        // Publish the aggregate before any predecessor can consume this tile.
        if (threadIdx.x == 0) {
            publish_aggregate(descriptors, tile_id, aggregate);
        }
        __syncwarp(0xffffffffu);

        if (threadIdx.x < 32) {
            warp_reduction_lookback(
                descriptors, tile_id, lookback_scratch);
        }

        if (threadIdx.x == 0) {
            // A failed look-back publishes a fallback so successor CTAs can
            // finish; the device error makes the host reject this output.
            block_prefix = lookback_scratch.state == LookbackDone
                ? lookback_scratch.prefix : T{0};
            publish_prefix(
                descriptors, tile_id,
                static_cast<T>(block_prefix + aggregate));
        }
    } else {
        if (threadIdx.x == 0) {
            publish_aggregate(descriptors, tile_id, aggregate);
            const bool success = serial_lookback(
                descriptors, tile_id, block_prefix);
            if (!success) {
                block_prefix = T{0};
                cuda::atomic_ref<unsigned, cuda::thread_scope_device> error(
                    *descriptors.error);
                error.store(1u, cuda::memory_order_relaxed);
            }
            publish_prefix(
                descriptors, tile_id,
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
