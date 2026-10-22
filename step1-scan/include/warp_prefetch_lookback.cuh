#pragma once

#include "lookback_consume.cuh"
#include <cuda/atomic>

namespace lookback {

template<class T>
struct LookbackResult {
    T prefix;
    bool success; // On failure, prefix must NOT be used as a valid zero prefix.
};

template<class T>
struct LookbackScratch {
    PrefetchWindow<T> window;
    T prefix;
    uint64_t end_exclusive;
    WindowAction action;
};

/**
 * Opt1: cooperative warp prefetch followed by sequential lane-0 consumption.
 * Return the sum of all tile aggregates strictly BEFORE logical tile_id.
 *
 * CALLING CONTRACT FOR P1/P2/P3:
 * - Call collectively from all 32 lanes of the first warp of a 1D block.
 *   Arguments must be identical across those lanes. Never call only in lane 0.
 * - scratch must reside in shared memory and belong exclusively to this warp
 *   during the call. All participating lanes receive the same result.
 * - Fs/As/Ps point to whole descriptor arrays; tile_id must not exceed their
 *   allocated length. An empty predecessor range (tile_id == 0) returns zero.
 * - Fs accesses during concurrent execution must use device-scope atomics.
 *   Producers publish an immutable payload before a matching release flag
 *   store; prefetch_lookback_window performs the acquire flag load.
 * - Initialize descriptors before the scan. The complete scan must publish
 *   the current tile's aggregate before entering look-back. This helper does
 *   NOT publish A/P flags or compute the current tile's inclusive prefix.
 * - P2/P3 must ensure that required predecessor producers can make progress.
 *   A required X that is never published will cause indefinite retry.
 * - Before the whole block calls store_tile_with_prefix(), the caller must
 *   broadcast this result through shared memory with a BLOCK-wide barrier.
 *   Every thread in that block must participate in that barrier.
 *
 * STATE TRANSITIONS:
 * - Done: add this window's contribution and return.
 * - Continue: add contribution and move to the preceding window.
 * - Retry: preserve both prefix and end_exclusive; refetch the same window.
 *   Discarding its tentative sum prevents double-counting on retry.
 * - Invalid: return success == false; the caller must handle the failure.
 *
 * No output elements are written here. Only the exclusive block prefix is
 * computed. Reusing scratch in consecutive collective calls is supported.
 */
template<class T>
__device__ __forceinline__
LookbackResult<T> warp_prefetch_lookback(
    Flag* Fs,
    const volatile T* As,
    const volatile T* Ps,
    uint64_t tile_id,
    LookbackScratch<T>& scratch,
    unsigned* retry_signal = nullptr
) {
    const unsigned lane = threadIdx.x & 31u;
    constexpr unsigned mask = 0xffffffffu;

    __syncwarp(mask);
    if (lane == 0) {
        scratch.prefix = T{0};
        scratch.end_exclusive = tile_id;
        scratch.action = WindowAction::Continue;
    }
    __syncwarp(mask);

    while (true) {
        const uint64_t end = scratch.end_exclusive;
        prefetch_lookback_window<T>(Fs, As, Ps, end, scratch.window);

        if (lane == 0) {
            const auto result = consume_lookback_window<T>(scratch.window);
            scratch.action = result.action;
            switch (result.action) {
                case WindowAction::Done:
                    scratch.prefix = static_cast<T>(
                        scratch.prefix + result.contribution);
                    break;
                case WindowAction::Continue:
                    scratch.prefix = static_cast<T>(
                        scratch.prefix + result.contribution);
                    scratch.end_exclusive -= scratch.window.valid_count;
                    break;
                case WindowAction::Retry: {
                    // Optional test instrumentation.
                    // When supplied, retry_signal must point to a device unsigned initialized
                    // to zero before launch. All lanes must pass the same pointer.
                    // Only lane 0 writes it, using a device-scope atomic release store.
                    // It records that an internal Retry occurred; it is NOT a retry counter.
                    // Production callers should omit this argument.
                    if (retry_signal != nullptr) {
                        cuda::atomic_ref<unsigned, cuda::thread_scope_device>
                            signal(*retry_signal);
                        signal.store(1u, cuda::memory_order_release);
                    }

                    // Preserve the accumulated prefix AND the current window position.
                    // The next iteration must reload this window from global memory.
                    break;
                }
                case WindowAction::Invalid:
                    scratch.prefix = T{0};
                    break;
            }
        }

        // Publish lane 0's decision to the entire participating warp.
        __syncwarp(mask);
        const auto action = scratch.action;
        if (action == WindowAction::Done || action == WindowAction::Invalid) {
            const LookbackResult<T> result{
                scratch.prefix, action == WindowAction::Done};
            // Finish all reads before scratch can be reused by another call.
            __syncwarp(mask);
            return result;
        }
        // The prefetch helper's entry barrier protects the window on reuse.
    }
}

} // namespace lookback
