#pragma once
// ---------------------------------------------------------------------------
// lookback_publish.cuh
//
// Status descriptor and publish primitives for the decoupled look-back.
//
// Contract:
//   - Fs / As / Ps are device arrays of length num_blocks.
//   - Fs is initialized to Flag::X by the host driver (P2) before launch.
//   - Publishing order is ALWAYS: write data -> __threadfence() -> write flag.
//     Writing the flag before the fence (or omitting the fence) lets a
//     predecessor observe A/P while the payload is still stale.
//   - publish_aggregate / publish_prefix must be called by exactly one
//     thread per block (typically tid == 0).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Flag enum.
//
// By default this header defines lookback::Flag locally. If P1's
// lookback_consume.cuh already provides lookback::Flag, define
// P3_USE_P1_FLAG before including this header and the local definition is
// skipped. Two coexisting Flag enums will silently break state comparisons.
// ---------------------------------------------------------------------------
#ifndef P3_USE_P1_FLAG
namespace lookback {
enum class Flag : uint32_t {
    X = 0,  // invalid: nothing published yet
    A = 1,  // aggregate available in As[idx]
    P = 2   // inclusive prefix available in Ps[idx]
};
} // namespace lookback
#else
#include "lookback_consume.cuh"   // P1: provides lookback::Flag
#endif

namespace decoupled_lookback {

// One descriptor per block. Fs/As/Ps are volatile to defeat compiler caching
// inside the busy-wait spin loop.
template <typename T>
struct Descriptor {
    volatile lookback::Flag* Fs;
    volatile T*              As;
    volatile T*              Ps;
    int                      num_blocks;
};

// Publish the block-wide aggregate: As[idx] = aggregate; fence; Fs[idx] = A.
// CALLING CONTRACT: single thread per block (tid == 0).
template <typename T>
__device__ __forceinline__ void publish_aggregate(
    Descriptor<T> desc, int block_idx, T block_aggregate)
{
    desc.As[block_idx] = block_aggregate;
    __threadfence();                          // make As visible to other blocks
    desc.Fs[block_idx] = lookback::Flag::A;
}

// Publish the block-wide inclusive prefix: Ps[idx] = prefix; fence; Fs[idx] = P.
// CALLING CONTRACT: single thread per block (tid == 0).
template <typename T>
__device__ __forceinline__ void publish_prefix(
    Descriptor<T> desc, int block_idx, T inclusive_prefix)
{
    desc.Ps[block_idx] = inclusive_prefix;
    __threadfence();                          // make Ps visible to other blocks
    desc.Fs[block_idx] = lookback::Flag::P;
}

} // namespace decoupled_lookback