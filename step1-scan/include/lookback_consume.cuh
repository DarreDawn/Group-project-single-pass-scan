#pragma once

#include "warp_prefetch.cuh"
#include <type_traits>

namespace lookback {

// These actions describe what the caller should do next.
// They are distinct from the X/A/P flags stored in global descriptors.
enum class WindowAction : uint32_t {
    Done,       // Add contribution and finish the look-back.
    Retry,      // Keep the accumulator and window position unchanged; refresh.
    Continue,   // Add contribution and move to the preceding window.
    Invalid     // Malformed window; report an error rather than using its sum.
};

template<class T>
struct WindowResult {
    WindowAction action;
    T contribution; // Contribution of THIS window, not the full block prefix.
};

/**
 * Consume one already-prefetched window in nearest-to-farthest order.
 *
 * GPU use: call from one lane after prefetch_lookback_window() completes.
 * This function has no barriers and does not read or publish global state.
 * The window must not be overwritten while it is being consumed.
 * Host use: the same implementation is available for unit tests.
 *
 * CALLER CONTRACT FOR P1/P2/P3:
 * - Done: add contribution to the accumulated prefix and finish.
 * - Retry: add NOTHING and keep end_exclusive unchanged. Refresh the same
 *   window from global descriptors before calling again. Never poll only
 *   the cached shared-memory flag: this function cannot observe updates.
 * - Continue: add contribution, then subtract window.valid_count from
 *   end_exclusive before fetching the preceding window.
 * - Invalid: report a protocol/input error; do not consume the contribution.
 *
 * Retry intentionally discards the tentative A sums in this window so that
 * refreshing and replaying the window cannot double-count those aggregates.
 * A P flag terminates immediately; older entries, including X, are irrelevant.
 * An empty window represents no remaining predecessors and contributes zero.
 * Addition uses the same unsigned modular arithmetic as Step1.
 */
template<class T>
__host__ __device__ __forceinline__
WindowResult<T> consume_lookback_window(const PrefetchWindow<T>& window) {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T> &&
                  !std::is_same_v<T, bool> && sizeof(T) <= 8,
                  "Use the same unsigned integer type as Step1");

    if (window.valid_count > 32) {
        return {WindowAction::Invalid, T{0}};
    }
    if (window.valid_count == 0) {
        return {WindowAction::Done, T{0}};
    }

    T sum = T{0};
    for (unsigned slot = 0; slot < window.valid_count; ++slot) {
        switch (window.flags[slot]) {
            case Flag::A:
                sum = static_cast<T>(sum + window.aggregates[slot]);
                break;
            case Flag::P:
                return {WindowAction::Done,
                        static_cast<T>(sum + window.prefixes[slot])};
            case Flag::X:
                return {WindowAction::Retry, T{0}};
            default:
                return {WindowAction::Invalid, T{0}};
        }
    }

    return {WindowAction::Continue, sum};
}

} // namespace lookback
