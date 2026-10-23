#include "warp_prefetch_lookback.cuh"
#include <cuda/atomic>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned kDescriptors = 96;
constexpr unsigned kLanes = 32;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(operation) check_cuda((operation), #operation)

template<class T>
class DeviceBuffer {
public:
    T* ptr = nullptr;

    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/*
 * Launch one block with exactly 64 threads.
 *
 * Warp 0 runs the actual complete look-back helper.
 * Thread 32 publishes only AFTER the helper reports an internal Retry.
 *
 * No separate prefetch/consume call is performed before the helper.
 * Therefore, the signal proves that the helper's own Retry branch ran.
 *
 * The handshake goes from reader to publisher. Payload visibility in the
 * reverse direction still depends on the Fs release/acquire protocol.
 */
template<class T>
__global__ void retry_case(
    lookback::Flag* Fs,
    T* As,
    T* Ps,
    unsigned missing,
    uint64_t tile_id,
    lookback::Flag published_flag,
    T payload,
    unsigned* retry_signal,
    lookback::LookbackResult<T>* output
) {
    __shared__ lookback::LookbackScratch<T> scratch;

    if (threadIdx.x < kLanes) {
        output[threadIdx.x] =
            lookback::warp_prefetch_lookback<T>(
                Fs, As, Ps, tile_id, scratch, retry_signal);
    } else if (threadIdx.x == 32) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>
            signal(*retry_signal);

        // Publication cannot happen before the actual helper retries.
        while (signal.load(cuda::memory_order_acquire) == 0u) {
            __nanosleep(64);
        }

        // Publish the payload first, then its availability flag.
        if (published_flag == lookback::Flag::A) {
            As[missing] = payload;
        } else {
            Ps[missing] = payload;
        }

        cuda::atomic_ref<lookback::Flag, cuda::thread_scope_device>
            flag(Fs[missing]);

        flag.store(published_flag, cuda::memory_order_release);
    }
}

template<class T>
unsigned run_type(const char* name) {
    using lookback::Flag;
    using Result = lookback::LookbackResult<T>;

    std::array<T, kDescriptors> reference_aggregates{};
    std::array<T, kDescriptors> reference_prefixes{};

    // Independent CPU reference using unsigned modular addition.
    T sum = T{0};
    for (unsigned i = 0; i < kDescriptors; ++i) {
        reference_aggregates[i] =
            static_cast<T>(0x0102030400000001ULL + i);

        sum = static_cast<T>(sum + reference_aggregates[i]);
        reference_prefixes[i] = sum;
    }

    DeviceBuffer<Flag> d_flags(kDescriptors);
    DeviceBuffer<T> d_aggregates(kDescriptors);
    DeviceBuffer<T> d_prefixes(kDescriptors);
    DeviceBuffer<unsigned> d_signal(1);
    DeviceBuffer<Result> d_output(kLanes);

    unsigned cases = 0;

    for (unsigned missing : {0u, 7u, 32u}) {
        /*
         * First window: 32 available A entries.
         * Second window: two newer A entries, then the missing X.
         *
         * This checks preservation of the first window's accumulated sum,
         * and rejection of the second window's tentative sum on Retry.
         */
        const uint64_t tile_id = missing + 35u;

        for (Flag published : {Flag::A, Flag::P}) {
            std::array<Flag, kDescriptors> flags{};
            flags.fill(Flag::A);
            flags[0] = Flag::P;
            flags[missing] = Flag::X;

            auto aggregates = reference_aggregates;
            auto prefixes = reference_prefixes;

            // The missing payload is installed later by the GPU publisher.
            aggregates[missing] = T{0};
            prefixes[missing] = T{0};

            const T payload = published == Flag::A
                ? reference_aggregates[missing]
                : reference_prefixes[missing];

            CUDA_CHECK(cudaMemcpy(
                d_flags.ptr, flags.data(),
                sizeof(flags), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_aggregates.ptr, aggregates.data(),
                sizeof(aggregates), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_prefixes.ptr, prefixes.data(),
                sizeof(prefixes), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemset(
                d_signal.ptr, 0, sizeof(unsigned)));

            CUDA_CHECK(cudaMemset(
                d_output.ptr, 0, kLanes * sizeof(Result)));

            retry_case<T><<<1, 64>>>(
                d_flags.ptr,
                d_aggregates.ptr,
                d_prefixes.ptr,
                missing,
                tile_id,
                published,
                payload,
                d_signal.ptr,
                d_output.ptr);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned observed_retry = 0;
            std::array<Result, kLanes> output{};

            CUDA_CHECK(cudaMemcpy(
                &observed_retry, d_signal.ptr,
                sizeof(unsigned), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(
                output.data(), d_output.ptr,
                sizeof(output), cudaMemcpyDeviceToHost));

            const std::string context =
                std::string(name) +
                " missing=" + std::to_string(missing) +
                " published=" + (published == Flag::A ? "A" : "P");

            if (observed_retry != 1u) {
                throw std::runtime_error(
                    context + ": internal Retry was not observed");
            }

            const T expected =
                reference_prefixes[static_cast<size_t>(tile_id - 1)];

            for (unsigned lane = 0; lane < kLanes; ++lane) {
                if (!output[lane].success ||
                    output[lane].prefix != expected) {
                    throw std::runtime_error(
                        context +
                        " lane=" + std::to_string(lane) +
                        ": incorrect prefix after Retry");
                }
            }

            ++cases;
        }
    }

    std::cout << "PASS " << name << ": " << cases
              << " forced internal Retry cases; "
                 "all 32 lane results checked\n";

    return cases;
}

} // namespace

int main() {
    try {
        unsigned cases = 0;

        cases += run_type<uint8_t>("u8");
        cases += run_type<uint16_t>("u16");
        cases += run_type<uint32_t>("u32");
        cases += run_type<uint64_t>("u64");

        std::cout
            << "PASS: " << cases
            << " forced internal Retry cases, "
            << cases * kLanes << " lane results\n"
            << "Scope: one-block X-to-A/P publication with a test signal; "
               "no A-to-P transition, inter-block scheduling, "
               "or full scan tested.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}