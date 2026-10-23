// ---------------------------------------------------------------------------
// test_baseline.cu
//
// Unit test for the baseline decoupled look-back state machine.
// Only the state machine is exercised: each block publishes
// aggregate = blockIdx.x + 1 and we verify the inclusive prefix equals
// 1, 3, 6, 10, ... (triangular numbers).
//
// The grid size must not exceed the number of simultaneously resident
// blocks, otherwise the spin loop can deadlock.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "lookback_baseline.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

__global__ void baseline_test_kernel(
    volatile lookback::Flag* Fs,
    volatile uint32_t*       As,
    volatile uint32_t*       Ps,
    uint32_t*                output,
    int                      num_blocks)
{
    if (threadIdx.x == 0) {
        decoupled_lookback::Descriptor<uint32_t> desc{Fs, As, Ps, num_blocks};
        uint32_t aggregate = static_cast<uint32_t>(blockIdx.x + 1);
        uint32_t excl = decoupled_lookback::baseline(
            desc, blockIdx.x, aggregate);
        output[blockIdx.x] = excl + aggregate;   // inclusive prefix
    }
}

int main() {
    const int num_blocks = 4;

    // Host-side state arrays.
    std::vector<lookback::Flag> h_Fs(num_blocks, lookback::Flag::X);
    std::vector<uint32_t>       h_As(num_blocks, 0);
    std::vector<uint32_t>       h_Ps(num_blocks, 0);
    std::vector<uint32_t>       h_out(num_blocks, 0);

    // Device allocations.
    lookback::Flag* d_Fs = nullptr;
    uint32_t*       d_As = nullptr;
    uint32_t*       d_Ps = nullptr;
    uint32_t*       d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fs, num_blocks * sizeof(lookback::Flag)));
    CUDA_CHECK(cudaMalloc(&d_As, num_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_Ps, num_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_Fs, h_Fs.data(),
                          num_blocks * sizeof(lookback::Flag),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_As, h_As.data(),
                          num_blocks * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ps, h_Ps.data(),
                          num_blocks * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Launch: one warp per block is enough for the state-machine test.
    baseline_test_kernel<<<num_blocks, 32>>>(
        d_Fs, d_As, d_Ps, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          num_blocks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // CPU reference: inclusive prefix of {1, 2, 3, 4} = {1, 3, 6, 10}.
    std::vector<uint32_t> ref(num_blocks, 0);
    uint32_t running = 0;
    for (int i = 0; i < num_blocks; ++i) {
        running += static_cast<uint32_t>(i + 1);
        ref[i] = running;
    }

    bool ok = true;
    for (int i = 0; i < num_blocks; ++i) {
        if (h_out[i] != ref[i]) {
            std::fprintf(stderr,
                         "Mismatch at block %d: got %u, expected %u\n",
                         i, h_out[i], ref[i]);
            ok = false;
        }
    }

    std::printf("test_baseline: %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Fs));
    CUDA_CHECK(cudaFree(d_As));
    CUDA_CHECK(cudaFree(d_Ps));
    CUDA_CHECK(cudaFree(d_out));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}