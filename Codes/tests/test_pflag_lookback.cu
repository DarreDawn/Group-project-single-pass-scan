// ---------------------------------------------------------------------------
// test_pflag_lookback.cu
//
// Correctness test for the P-flag propagation strategy.
// Runs both baseline and pflag on identical aggregates; verifies:
//   - both outputs match CPU reference (triangular numbers)
//   - baseline and pflag produce the same output
//   - after each kernel, every Fs entry is P (all blocks completed)
//
// grid must not exceed resident blocks; otherwise the spin loop can deadlock.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "lookback_baseline.cuh"
#include "lookback_pflag.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

template <bool UsePFlag>
__global__ void pflag_test_kernel(
    volatile lookback::Flag* Fs,
    volatile uint32_t*       As,
    volatile uint32_t*       Ps,
    uint32_t*                output,
    int                      num_blocks)
{
    if (threadIdx.x == 0) {
        decoupled_lookback::Descriptor<uint32_t> desc{Fs, As, Ps, num_blocks};
        uint32_t aggregate = static_cast<uint32_t>(blockIdx.x + 1);
        uint32_t excl;
        if constexpr (UsePFlag) {
            excl = decoupled_lookback::pflag(desc, blockIdx.x, aggregate);
        } else {
            excl = decoupled_lookback::baseline(desc, blockIdx.x, aggregate);
        }
        output[blockIdx.x] = excl + aggregate;   // inclusive prefix
    }
}

// Runs one kernel, checks output against CPU reference and (optionally)
// checks that all Fs entries are P after the kernel.
template <bool UsePFlag>
static bool run_one(const char* name, int num_blocks,
                    const std::vector<uint32_t>& ref)
{
    lookback::Flag* d_Fs = nullptr;
    uint32_t*       d_As = nullptr;
    uint32_t*       d_Ps = nullptr;
    uint32_t*       d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fs, num_blocks * sizeof(lookback::Flag)));
    CUDA_CHECK(cudaMalloc(&d_As, num_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_Ps, num_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(uint32_t)));

    // Initialize Fs to X.
    std::vector<lookback::Flag> h_Fs(num_blocks, lookback::Flag::X);
    CUDA_CHECK(cudaMemcpy(d_Fs, h_Fs.data(),
                          num_blocks * sizeof(lookback::Flag),
                          cudaMemcpyHostToDevice));

    pflag_test_kernel<UsePFlag><<<num_blocks, 32>>>(
        d_Fs, d_As, d_Ps, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t>       h_out(num_blocks, 0);
    std::vector<lookback::Flag> h_Fs_after(num_blocks, lookback::Flag::X);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          num_blocks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Fs_after.data(), d_Fs,
                          num_blocks * sizeof(lookback::Flag),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < num_blocks; ++i) {
        if (h_out[i] != ref[i]) {
            std::fprintf(stderr,
                         "  [%s] block %d: got %u, expected %u\n",
                         name, i, h_out[i], ref[i]);
            ok = false;
        }
        if (h_Fs_after[i] != lookback::Flag::P) {
            std::fprintf(stderr,
                         "  [%s] block %d: Fs not P after kernel (%u)\n",
                         name, i, static_cast<unsigned>(h_Fs_after[i]));
            ok = false;
        }
    }

    CUDA_CHECK(cudaFree(d_Fs));
    CUDA_CHECK(cudaFree(d_As));
    CUDA_CHECK(cudaFree(d_Ps));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

int main() {
    const int num_blocks = 8;

    std::vector<uint32_t> ref(num_blocks, 0);
    uint32_t running = 0;
    for (int i = 0; i < num_blocks; ++i) {
        running += static_cast<uint32_t>(i + 1);
        ref[i] = running;   // 1, 3, 6, 10, 15, 21, 28, 36
    }

    bool ok = true;
    ok &= run_one<false>("baseline", num_blocks, ref);
    ok &= run_one<true >("pflag",    num_blocks, ref);

    std::printf("test_pflag_lookback: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}