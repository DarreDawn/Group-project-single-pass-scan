// ---------------------------------------------------------------------------
// test_pflag_large.cu
//
// Stress test for pflag with a large look-back distance.
//
// A soft global barrier forces every block into the A state before any block
// starts walking. The last block's walk then traverses many A entries,
// exercising the MAX_TRACK overflow branch in lookback_pflag.cuh.
//
// Correctness property: dropping entries affects only promotion coverage,
// not the accumulated exclusive prefix. The output must therefore still
// match the CPU reference (exact for int32, tolerance for float).
//
// The soft barrier requires all blocks to be simultaneously resident. The
// host side uses cudaOccupancyMaxActiveBlocksPerMultiprocessor (which accounts
// for register / shared-memory pressure) rather than the theoretical
// cudaDevAttrMaxBlocksPerMultiprocessor. If the GPU cannot host enough
// blocks, the case is SKIPPED rather than failing the suite.
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "lookback_publish.cuh"
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

template <typename T>
__global__ void pflag_large_kernel(
    volatile lookback::Flag* Fs,
    volatile T*              As,
    volatile T*              Ps,
    T*                       output,
    volatile unsigned int*   barrier_counter,   // volatile: cross-block spin
    int                      num_blocks)
{
    if (threadIdx.x == 0) {
        decoupled_lookback::Descriptor<T> desc{Fs, As, Ps, num_blocks};

        T aggregate;
        if constexpr (std::is_floating_point<T>::value) {
            aggregate = static_cast<T>(1) / static_cast<T>(blockIdx.x + 1);
        } else {
            aggregate = static_cast<T>(blockIdx.x + 1);
        }

        // Publish A first, then wait for every block to reach the same point.
        decoupled_lookback::publish_aggregate(desc, blockIdx.x, aggregate);
        __threadfence();
        atomicAdd((unsigned int*)barrier_counter, 1u);

        // Volatile spin: compiler must re-read on every iteration.
        while (*barrier_counter < static_cast<unsigned int>(num_blocks)) {
            // spin: every block must be in A state before any walk starts
        }

        // Now every block has published A. Last block's walk traverses
        // num_blocks - 1 A entries, exercising MAX_TRACK overflow.
        T excl = decoupled_lookback::pflag(desc, blockIdx.x, aggregate);
        output[blockIdx.x] = static_cast<T>(excl + aggregate);
    }
}

template <typename T>
static bool compare(const T* got, const T* ref, int n, double rel_tol)
{
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            double denom = std::max(std::fabs(static_cast<double>(ref[i])),
                                    1e-12);
            double err = std::fabs(static_cast<double>(got[i]) -
                                   static_cast<double>(ref[i])) / denom;
            if (err > rel_tol) {
                std::fprintf(stderr,
                             "  float mismatch at %d: got %.9g expected %.9g "
                             "(rel err %.3e > tol %.3e)\n",
                             i, static_cast<double>(got[i]),
                             static_cast<double>(ref[i]), err, rel_tol);
                return false;
            }
        } else {
            if (got[i] != ref[i]) {
                std::fprintf(stderr,
                             "  int mismatch at %d: got %lld expected %lld\n",
                             i, static_cast<long long>(got[i]),
                             static_cast<long long>(ref[i]));
                return false;
            }
        }
    }
    return true;
}

template <typename T>
static bool run_case(const char* name, int requested_blocks, double rel_tol)
{
    // Accurate resident-block count for this kernel.
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, pflag_large_kernel<T>, 32, 0));
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int slots = blocks_per_sm * sm_count;

    // Reserve one slot so the soft barrier cannot be starved.
    int num_blocks = std::min(requested_blocks, slots - 1);

    // We want > MAX_TRACK + 1 blocks to guarantee a walk of length > MAX_TRACK.
    constexpr int MIN_BLOCKS = 34;   // MAX_TRACK = 32, need distance >= 33
    if (num_blocks < MIN_BLOCKS) {
        std::fprintf(stderr,
                     "  [%s] SKIPPED: resident slots=%d (needs >= %d)\n",
                     name, slots, MIN_BLOCKS + 1);
        return true;
    }

    std::printf("  [%s] running with num_blocks=%d (slots=%d)\n",
                name, num_blocks, slots);

    // CPU reference.
    std::vector<T> ref(num_blocks, T{0});
    T running = T{0};
    for (int i = 0; i < num_blocks; ++i) {
        T agg;
        if constexpr (std::is_floating_point<T>::value) {
            agg = static_cast<T>(1) / static_cast<T>(i + 1);
        } else {
            agg = static_cast<T>(i + 1);
        }
        running = static_cast<T>(running + agg);
        ref[i] = running;
    }

    lookback::Flag* d_Fs = nullptr;
    T*              d_As = nullptr;
    T*              d_Ps = nullptr;
    T*              d_out = nullptr;
    unsigned int*   d_barrier = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Fs, num_blocks * sizeof(lookback::Flag)));
    CUDA_CHECK(cudaMalloc(&d_As, num_blocks * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_Ps, num_blocks * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, num_blocks * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_barrier, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_Fs, 0, num_blocks * sizeof(lookback::Flag)));
    CUDA_CHECK(cudaMemset(d_barrier, 0, sizeof(unsigned int)));

    pflag_large_kernel<T><<<num_blocks, 32>>>(
        d_Fs, d_As, d_Ps, d_out, d_barrier, num_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T>              h_out(num_blocks, T{0});
    std::vector<lookback::Flag> h_Fs(num_blocks, lookback::Flag::X);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Fs.data(), d_Fs,
                          num_blocks * sizeof(lookback::Flag),
                          cudaMemcpyDeviceToHost));

    bool ok = compare<T>(h_out.data(), ref.data(), num_blocks, rel_tol);
    for (int i = 0; i < num_blocks; ++i) {
        if (h_Fs[i] != lookback::Flag::P) {
            std::fprintf(stderr,
                         "  [%s] block %d: Fs not P after kernel (%u)\n",
                         name, i, static_cast<unsigned>(h_Fs[i]));
            ok = false;
        }
    }
    if (!ok) {
        std::fprintf(stderr, "  [%s] FAILED\n", name);
    } else {
        std::printf("  [%s] OK (num_blocks=%d)\n", name, num_blocks);
    }

    CUDA_CHECK(cudaFree(d_Fs));
    CUDA_CHECK(cudaFree(d_As));
    CUDA_CHECK(cudaFree(d_Ps));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_barrier));
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_case<int32_t>("int32", 48, 0.0);
    ok &= run_case<float>  ("float", 48, 1e-4);
    std::printf("test_pflag_large: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}