#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda/atomic>

namespace lookback {

// Shared status encoding for P1/P2/P3.
// X: no published result.
// A: the tile aggregate is available.
// P: the inclusive prefix through this tile is available.
enum class Flag : uint32_t {
    X = 0,
    A = 1,
    P = 2
};

// Shared-memory storage for one window of up to 32 predecessors.
// T must match the value type used by Step1 and the global As/Ps arrays.
template<class T>
struct PrefetchWindow {
    Flag flags[32];
    T aggregates[32];
    T prefixes[32];

    // Number of valid predecessor entries: 0 <= valid_count <= 32.
    unsigned valid_count;
};

// Return the number of valid predecessors in the current window.
// end_exclusive == 0 means there are no predecessors.
__host__ __device__ __forceinline__
unsigned window_size(uint64_t end_exclusive) {
    return end_exclusive < 32
        ? static_cast<unsigned>(end_exclusive)
        : 32u;
}

// Map one lane to a predecessor tile, starting from the nearest one.
//
// Returns true if this lane has a valid predecessor.
// On success, tile_index contains that predecessor's logical index.
// On failure, tile_index is set to zero and MUST NOT be used for access.
__host__ __device__ __forceinline__
bool get_predecessor_index(
    uint64_t end_exclusive,
    unsigned lane,
    uint64_t& tile_index
) {
    tile_index = 0;

    const unsigned count = window_size(end_exclusive);

    // Check validity BEFORE subtracting to avoid unsigned underflow. 
    //Returning `false` simply indicates that the current lane has no valid predecessor.
    if (lane >= count) {
        return false;
    }

    tile_index = end_exclusive - 1 - static_cast<uint64_t>(lane);
    return true;
}

// Read one flag with device-scope acquire semantics.
//
// P3 integration contract:
// - Fs must be a naturally aligned array of Flag objects.
// - During concurrent execution, all accesses to each flag must use
//   device-scope atomic operations.
// - P3 must write the payload before publishing A/P with a release store.
// - Published payloads must remain unchanged until the scan completes.
__device__ __forceinline__
Flag load_flag_acquire(Flag* Fs, uint64_t tile_index) {
    cuda::atomic_ref<Flag, cuda::thread_scope_device> flag(
        Fs[tile_index]
    );

    return flag.load(cuda::memory_order_acquire);
}

/**
 * Prefetch one window of predecessor descriptors into shared memory.
 *
 * CALLING CONTRACT:
 * - Called collectively by all 32 lanes of the first warp.
 * - All participating lanes must pass identical arguments.
 * - Fs/As/Ps point to the beginnings of the global status arrays.
 * - end_exclusive must not exceed the number of allocated descriptors.
 * - window must refer to shared memory owned by this block.
 *
 * WINDOW LAYOUT:
 * - valid_count = min(end_exclusive, 32).
 * - Entry k represents tile (end_exclusive - 1 - k).
 * - Only entries [0, valid_count) represent real predecessors.
 * - end_exclusive == 0 produces an empty window.
 *
 * READ SEMANTICS:
 * - Read the flag before reading its corresponding payload.
 * - For A, read As; for P, read Ps; for X, read neither payload.
 * - Unused payload fields are initialized to zero.
 * - A zero payload for X does NOT mean a zero contribution.
 *
 * SYNCHRONIZATION:
 * - Global flag reads must acquire data published by P3.
 * - P3 must publish flags with matching device-scope release semantics.
 * - Published As/Ps values must remain unchanged during this scan.
 * - volatile alone does not provide this synchronization.
 * - Before return, synchronize the participating warp so all its lanes
 *   can read the completed window.
 * - Other warps need caller-provided block synchronization before access.
 *
 * SCOPE:
 * - Does not publish or modify global descriptors.
 * - Does not wait for X entries to become ready.
 * - Does not accumulate or return the final block prefix.
 * - A caller needing an X entry must refresh its global state;
 *   polling only the cached shared-memory flag cannot observe updates.
 */
template<class T>
__device__ __forceinline__
void prefetch_lookback_window(
    Flag* Fs,
    const volatile T* As,
    const volatile T* Ps,
    uint64_t end_exclusive,
    PrefetchWindow<T>& window
) {
    const unsigned lane = threadIdx.x & 31u;

    // All 32 lanes of the first warp must participate.
    // Finish this warp's previous reads before reusing the window.
    __syncwarp(0xffffffffu);

    uint64_t tile_index = 0;
    const bool valid = get_predecessor_index(
        end_exclusive, lane, tile_index
    );

    // Defaults for X entries and lanes without a predecessor.
    Flag observed_flag = Flag::X;
    T aggregate = T{0};
    T prefix = T{0};

    if (valid) {
        // Acquire the published status before reading its payload.
        observed_flag = load_flag_acquire(Fs, tile_index);

        if (observed_flag == Flag::A) {
            aggregate = As[tile_index];
        } else if (observed_flag == Flag::P) {
            prefix = Ps[tile_index];
        }

        // For X, do not read either payload.
        // A later look-back step must refresh this state if needed.
    }

    // Each lane writes one entry in the shared-memory window.
    window.flags[lane] = observed_flag;
    window.aggregates[lane] = aggregate;
    window.prefixes[lane] = prefix;

    // Only one lane writes the shared window size.
    if (lane == 0) {
        window.valid_count = window_size(end_exclusive);
    }

    // Make every entry available to all participating lanes.
    // Invalid lanes must also reach this synchronization point.
    __syncwarp(0xffffffffu);
}

} // namespace lookback