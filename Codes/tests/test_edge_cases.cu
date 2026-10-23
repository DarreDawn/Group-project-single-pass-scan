// ---------------------------------------------------------------------------
// test_edge_cases.cu
//
// Edge-case coverage for the decoupled look-back module:
//   1. num_blocks = 1, blockDim = 32  -> only block 0 exists (no look-back)
//   2. num_blocks = 2, blockDim = 32  -> minimal look-back (block 1 sees
//                                        block 0, which may still be A)
//   3. num_blocks = 4, blockDim = 1   -> single-thread blocks (no warp
//                                        support). Verified the module
//                                        does not rely on warp-level
//                                        primitives or intra-block sync.
//
// Baseline and pflag are both exercised on each case. All three cases are
// resident-safe on any CUDA-capable GPU.
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
__global__ void edge_kernel(
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
        output[blockIdx.x] = excl + aggregate;
    }
}

template <bool UsePFlag>
static bool run_one(const char* name, int num_blocks, int block_dim,
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
    CUDA_CHECK(cudaMemset(d_Fs, 0, num_blocks * sizeof(lookback::Flag)));

    edge_kernel<UsePFlag><<<num_blocks, block_dim>>>(
        d_Fs, d_As, d_Ps, d_out, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t>       h_out(num_blocks, 0);
    std::vector<lookback::Flag> h_Fs(num_blocks, lookback::Flag::X);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          num_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Fs.data(), d_Fs,
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
        if (h_Fs[i] != lookback::Flag::P) {
            std::fprintf(stderr,
                         "  [%s] block %d: Fs not P (%u)\n",
                         name, i, static_cast<unsigned>(h_Fs[i]));
            ok = false;
        }
    }
    if (!ok) {
        std::fprintf(stderr, "  [%s] FAILED\n", name);
    } else {
        std::printf("  [%s] OK (num_blocks=%d, block_dim=%d)\n",
                    name, num_blocks, block_dim);
    }

    CUDA_CHECK(cudaFree(d_Fs));
    CUDA_CHECK(cudaFree(d_As));
    CUDA_CHECK(cudaFree(d_Ps));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

static std::vector<uint32_t> make_ref(int num_blocks)
{
    std::vector<uint32_t> ref(num_blocks, 0);
    uint32_t running = 0;
    for (int i = 0; i < num_blocks; ++i) {
        running += static_cast<uint32_t>(i + 1);
        ref[i] = running;
    }
    return ref;
}

int main() {
    bool ok = true;

    // Case 1: single block, no look-back. Hits the block_idx == 0 early-out.
    {
        const int nb = 1;
        auto ref = make_ref(nb);
        ok &= run_one<false>("baseline/nb=1", nb, 32, ref);
        ok &= run_one<true >("pflag/nb=1",    nb, 32, ref);
    }

    // Case 2: two blocks, minimal look-back. Exercises the Fs[0]==A window
    // and the idx==0 break guard in both strategies.
    {
        const int nb = 2;
        auto ref = make_ref(nb);
        ok &= run_one<false>("baseline/nb=2", nb, 32, ref);
        ok &= run_one<true >("pflag/nb=2",    nb, 32, ref);
    }

    // Case 3: single-thread blocks. Confirms the module does not rely on
    // intra-block synchronization or warp primitives.
    {
        const int nb = 4;
        auto ref = make_ref(nb);
        ok &= run_one<false>("baseline/nb=4/th=1", nb, 1, ref);
        ok &= run_one<true >("pflag/nb=4/th=1",    nb, 1, ref);
    }

    std::printf("test_edge_cases: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}