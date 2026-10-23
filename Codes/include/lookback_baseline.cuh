#pragma once
// ---------------------------------------------------------------------------
// lookback_baseline.cuh
//
// Baseline decoupled look-back: a single thread per block walks backwards
// through predecessor descriptors, accumulating A entries and terminating on
// the first P entry. X entries are busy-waited.
//
// CALLING CONTRACT:
//   - Call with EXACTLY ONE thread per block (tid == 0).
//   - block_idx is the LOGICAL block index supplied by P2's dynamic numbering.
//     This function never reads blockIdx.x.
//   - block_aggregate is the tile aggregate returned by P1's
//     block_local_scan; it is valid in every thread.
//   - The returned exclusive prefix must be broadcast to the whole block
//     (e.g. via shared memory) before store_tile_with_prefix is called.
//   - All blocks must be simultaneously resident, or P2's dynamic numbering
//     must guarantee forward progress; otherwise the spin loop can deadlock.
//
// Block-0 edge case:
//   Block 0 has no predecessor. Its aggregate IS its inclusive prefix, so a
//   walker that observes Fs[0] == A can add As[0] and stop immediately; there
//   is no need to wait for Fs[0] to become P. The explicit `idx == 0` guard in
//   the loop prevents `--idx` from reaching -1 and reading Fs[-1].
// ---------------------------------------------------------------------------

#include "lookback_publish.cuh"

namespace decoupled_lookback {

// Returns the exclusive prefix of this block (sum over all preceding blocks).
template <typename T>
__device__ __forceinline__ T baseline(
    Descriptor<T> desc, int block_idx, T block_aggregate)
{
    // Publish this block's aggregate (A).
    publish_aggregate(desc, block_idx, block_aggregate);

    // Block 0: aggregate is already the inclusive prefix; nothing to look back.
    if (block_idx == 0) {
        publish_prefix(desc, block_idx, block_aggregate);
        return T{0};
    }

    // Serial look-back starting from the immediate predecessor.
    T exclusive_prefix = T{0};
    int idx = block_idx - 1;

    while (true) {
        // Busy-wait until the predecessor publishes anything (flag != X).
        // Do NOT place __syncthreads() or any block-wide barrier in here.
        while (desc.Fs[idx] == lookback::Flag::X) {
            // spin
        }
        __threadfence();                      // order flag read before As/Ps read

        lookback::Flag flag = desc.Fs[idx];
        if (flag == lookback::Flag::P) {
            exclusive_prefix = static_cast<T>(exclusive_prefix + desc.Ps[idx]);
            break;                            // hit a known prefix: terminate
        } else if (flag == lookback::Flag::A) {
            exclusive_prefix = static_cast<T>(exclusive_prefix + desc.As[idx]);
            // Block 0's aggregate IS its inclusive prefix: we are done.
            // Without this guard, --idx would reach -1 and read Fs[-1].
            if (idx == 0) {
                break;
            }
            --idx;                            // aggregate only: keep walking
        } else {
            break;                            // defensive: unexpected flag
        }
    }

    // Publish this block's inclusive prefix (P).
    T inclusive_prefix = static_cast<T>(exclusive_prefix + block_aggregate);
    publish_prefix(desc, block_idx, inclusive_prefix);

    return exclusive_prefix;
}

} // namespace decoupled_lookback