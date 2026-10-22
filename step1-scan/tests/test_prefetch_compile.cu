#include "warp_prefetch.cuh"

// Compile-only probe.
// If executed later, this kernel requires exactly 32 threads per block.
__global__ void compile_prefetch_u32(
    lookback::Flag* Fs,
    const uint32_t* As,
    const uint32_t* Ps,
    uint64_t end_exclusive,
    lookback::PrefetchWindow<uint32_t>* output
) {
    __shared__ lookback::PrefetchWindow<uint32_t> window;

    lookback::prefetch_lookback_window<uint32_t>(
        Fs, As, Ps, end_exclusive, window
    );

    if (threadIdx.x == 0) {
        *output = window;
    }
}