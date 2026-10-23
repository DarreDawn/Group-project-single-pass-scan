#pragma once
// ---------------------------------------------------------------------------
// lookback_pflag.cuh
//
// Decoupled look-back with P-flag propagation.
//
// Baseline publishes only its own P. Blocks that started look-back *before*
// that publication are still walking through A entries. P-flag propagation
// lets the completing block backfill those A entries as P, so subsequent
// (or in-flight) walkers can terminate earlier.
//
// Promotion formula:
//   E[b]          = exclusive prefix of the completing block b
//   running_at[i] = A[b-1] + A[b-2] + ... + A[i]
//   P[i]          = E[i+1] = E[b] - running_at[i] + A[i]
//
// Exact for integer types. For floating point the value may differ slightly
// from direct summation, matching the paper's nondeterminism caveat for
// pseudo-associative operators.
//
// CALLING CONTRACT (same as lookback_baseline.cuh):
//   - Call with EXACTLY ONE thread per block (tid == 0).
//   - block_idx is the LOGICAL block index supplied by P2.
//   - All blocks must be resident, or P2's dynamic numbering must guarantee
//     forward progress.
//
// Block-0 edge case:
//   A walker that observes Fs[0] == A can add As[0] and stop; there is no
//   predecessor beyond block 0. The explicit `idx == 0` guard prevents
//   `--idx` from reaching -1 and reading Fs[-1].
//
// Note on compute-sanitizer:
//   Promotion writes Ps[i]/Fs[i] concurrently from multiple blocks with the
//   same (deterministic) value. These are idempotent, natural-sized writes,
//   benign for CUDA but flagged as races by `racecheck`. Run memcheck and
//   synccheck on this test; skip racecheck, or accept the known-benign
//   reports.
// ---------------------------------------------------------------------------

#include "lookback_publish.cuh"

namespace decoupled_lookback {

// Returns the exclusive prefix of this block. Also promotes A entries it
// encountered to P.
template <typename T>
__device__ __forceinline__ T pflag(
    Descriptor<T> desc, int block_idx, T block_aggregate)
{
    publish_aggregate(desc, block_idx, block_aggregate);

    if (block_idx == 0) {
        publish_prefix(desc, block_idx, block_aggregate);
        return T{0};
    }

    // Track A entries seen during the walk so we can promote them later.
    constexpr int MAX_TRACK = 32;
    struct TrackedEntry {
        int idx;
        T   running_at;   // A[b-1] + ... + A[idx]
        T   a_value;      // A[idx]
    };
    TrackedEntry track[MAX_TRACK];
    int n_track = 0;

    T exclusive_prefix = T{0};
    int idx = block_idx - 1;

    while (true) {
        while (desc.Fs[idx] == lookback::Flag::X) {
            // spin
        }
        __threadfence();

        lookback::Flag flag = desc.Fs[idx];
        if (flag == lookback::Flag::P) {
            exclusive_prefix = static_cast<T>(exclusive_prefix + desc.Ps[idx]);
            break;
        } else if (flag == lookback::Flag::A) {
            T a = desc.As[idx];
            exclusive_prefix = static_cast<T>(exclusive_prefix + a);
            if (n_track < MAX_TRACK) {
                track[n_track++] = {idx, exclusive_prefix, a};
            }
            // Block 0's aggregate IS its inclusive prefix: we are done.
            // Without this guard, --idx would reach -1 and read Fs[-1].
            if (idx == 0) {
                break;
            }
            --idx;
        } else {
            break;  // defensive
        }
    }

    // Publish our own P first: successors arriving after this point terminate
    // at us without needing any promoted entry.
    T inclusive_prefix = static_cast<T>(exclusive_prefix + block_aggregate);
    publish_prefix(desc, block_idx, inclusive_prefix);

    // Promote tracked A entries. Idempotent: if another block promoted the
    // same entry first, both wrote the same deterministic value.
    int promote_idx[MAX_TRACK];
    int n_promote = 0;
    for (int k = 0; k < n_track; ++k) {
        if (desc.Fs[track[k].idx] == lookback::Flag::A) {
            T promoted = static_cast<T>(
                exclusive_prefix - track[k].running_at + track[k].a_value);
            desc.Ps[track[k].idx] = promoted;
            promote_idx[n_promote++] = track[k].idx;
        }
    }
    if (n_promote > 0) {
        __threadfence();                         // all Ps visible before any Fs
        for (int j = 0; j < n_promote; ++j) {
            desc.Fs[promote_idx[j]] = lookback::Flag::P;
        }
    }

    return exclusive_prefix;
}

} // namespace decoupled_lookback