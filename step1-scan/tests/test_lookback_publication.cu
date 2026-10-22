#include "warp_prefetch_lookback.cuh"
#include <cuda/atomic>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kDescriptors = 65;
constexpr unsigned kLanes = 32;
constexpr unsigned kRepeats = 8;

void check_cuda(cudaError_t status, const char* operation, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " at line " +
            std::to_string(line) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(operation) check_cuda((operation), #operation, __LINE__)

template<class T>
class DeviceBuffer {
public:
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/**
 * Launch exactly one block of 64 threads.
 * Warp 0 reads descriptors; thread 32 in warp 1 publishes the missing payload.
 * Keeping both roles in one block avoids depending on inter-block scheduling.
 *
 * The initial prefetch/consume MUST observe X before publication is allowed.
 * The gate is only a test handshake, not part of the scan's production API.
 * The reader never waits for a producer-to-reader gate: payload visibility
 * must come from the actual Fs release/acquire protocol under test.
 *
 * After the initial Retry, call the unmodified complete look-back helper.
 * The producer may publish before or during that call. This test therefore
 * guarantees the initial Retry observation, but does NOT count or guarantee
 * Retry iterations INSIDE warp_prefetch_lookback().
 */
template<class T>
__global__ void publication_case(
    lookback::Flag* Fs, T* As, T* Ps, unsigned missing, uint64_t tile_id,
    lookback::Flag published_flag, T published_payload, unsigned* gate,
    lookback::WindowResult<T>* initial,
    lookback::LookbackResult<T>* output
) {
    __shared__ lookback::LookbackScratch<T> scratch;

    if (threadIdx.x < kLanes) {
        lookback::prefetch_lookback_window<T>(
            Fs, As, Ps, tile_id, scratch.window);
        if (threadIdx.x == 0) {
            *initial = lookback::consume_lookback_window<T>(scratch.window);
            // Release only AFTER consumption, so the initial snapshot is X.
            cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(*gate);
            ready.store(1u, cuda::memory_order_release);
        }
        __syncwarp(0xffffffffu);

        output[threadIdx.x] = lookback::warp_prefetch_lookback<T>(
            Fs, As, Ps, tile_id, scratch);
    } else if (threadIdx.x == 32) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(*gate);
        while (ready.load(cuda::memory_order_acquire) == 0u) {
            __nanosleep(64);
        }

        // P3 handoff example: write the payload BEFORE publishing its flag.
        // Only the field selected by the new flag is written, and it remains
        // immutable thereafter. No ordinary concurrent store to Fs is allowed.
        if (published_flag == lookback::Flag::A) {
            As[missing] = published_payload;
        } else {
            Ps[missing] = published_payload;
        }
        cuda::atomic_ref<lookback::Flag, cuda::thread_scope_device> flag(Fs[missing]);
        flag.store(published_flag, cuda::memory_order_release);
    }
}

template<class T>
unsigned run_type(const char* name) {
    using lookback::Flag;
    using Result = lookback::LookbackResult<T>;
    using Initial = lookback::WindowResult<T>;
    DeviceBuffer<Flag> d_flags(kDescriptors);
    DeviceBuffer<T> d_aggregates(kDescriptors), d_prefixes(kDescriptors);
    DeviceBuffer<unsigned> d_gate(1);
    DeviceBuffer<Initial> d_initial(1);
    DeviceBuffer<Result> d_output(kLanes);
    std::vector<Result> output(kLanes);
    unsigned cases = 0;

    for (unsigned repeat = 0; repeat < kRepeats; ++repeat) {
        std::array<T, kDescriptors> reference_aggregates{}, reference_prefixes{};
        T sum = T{0};
        for (unsigned i = 0; i < kDescriptors; ++i) {
            reference_aggregates[i] = static_cast<T>(
                0x0102030400000001ULL + i + 17u * repeat);
            sum = static_cast<T>(sum + reference_aggregates[i]);
            reference_prefixes[i] = sum;
        }

        for (unsigned missing : {0u, 7u, 32u}) {
            // Two newer A entries precede X in the first fetched window.
            // Their tentative sum must be discarded by the initial Retry.
            const uint64_t tile_id = missing + 3u;
            for (Flag published : {Flag::A, Flag::P}) {
                std::array<Flag, kDescriptors> flags{};
                flags.fill(Flag::A);
                flags[0] = Flag::P;
                flags[missing] = Flag::X;
                auto aggregates = reference_aggregates;
                auto prefixes = reference_prefixes;
                // The actual tested payload is not installed until the
                // publisher runs on the GPU, after X has been observed.
                aggregates[missing] = T{0};
                prefixes[missing] = T{0};
                const T payload = published == Flag::A
                    ? reference_aggregates[missing] : reference_prefixes[missing];

                CUDA_CHECK(cudaMemcpy(d_flags.ptr, flags.data(),
                    sizeof(flags), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_aggregates.ptr, aggregates.data(),
                    sizeof(aggregates), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_prefixes.ptr, prefixes.data(),
                    sizeof(prefixes), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_gate.ptr, 0, sizeof(unsigned)));
                CUDA_CHECK(cudaMemset(d_initial.ptr, 0, sizeof(Initial)));
                CUDA_CHECK(cudaMemset(d_output.ptr, 0, kLanes * sizeof(Result)));

                publication_case<T><<<1, 64>>>(d_flags.ptr, d_aggregates.ptr,
                    d_prefixes.ptr, missing, tile_id, published, payload,
                    d_gate.ptr, d_initial.ptr, d_output.ptr);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                Initial initial{};
                CUDA_CHECK(cudaMemcpy(&initial, d_initial.ptr,
                    sizeof(Initial), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output.data(), d_output.ptr,
                    output.size() * sizeof(Result), cudaMemcpyDeviceToHost));
                const std::string context = std::string(name) +
                    " repeat=" + std::to_string(repeat) +
                    " missing=" + std::to_string(missing) +
                    " published=" + (published == Flag::A ? "A" : "P");
                if (initial.action != lookback::WindowAction::Retry ||
                    initial.contribution != T{0}) {
                    throw std::runtime_error(context + " initial Retry check failed");
                }
                const T expected = reference_prefixes[static_cast<size_t>(tile_id - 1)];
                for (unsigned lane = 0; lane < kLanes; ++lane) {
                    if (!output[lane].success || output[lane].prefix != expected) {
                        throw std::runtime_error(context + " lane=" +
                            std::to_string(lane) + " final prefix mismatch");
                    }
                }
                ++cases;
            }
        }
    }
    std::cout << "PASS " << name << ": " << cases
              << " publication cases; initial Retry and all 32 lane results checked\n";
    return cases;
}
} // namespace

int main() {
    try {
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (properties.major < 7) {
            throw std::runtime_error("This test requires compute capability 7.0 or newer");
        }
        std::cout << "GPU: " << properties.name << '\n';
        unsigned cases = 0;
        cases += run_type<uint8_t>("u8");
        cases += run_type<uint16_t>("u16");
        cases += run_type<uint32_t>("u32");
        cases += run_type<uint64_t>("u64");
        std::cout << "PASS: " << cases << " publication cases, " << cases
                  << " initial Retry checks, " << cases * kLanes << " lane results\n"
                  << "Scope: one-block X-to-A/P publication; internal loop Retry "
                     "count is not observed; no A-to-P transition or full scan tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
