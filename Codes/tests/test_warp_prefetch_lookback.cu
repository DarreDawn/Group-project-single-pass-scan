#include "warp_prefetch_lookback.cuh"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kDescriptors = 65;
constexpr unsigned kQueries = 8;
constexpr unsigned kLanes = 32;

struct Queries {
    uint64_t ends[kQueries];
};

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

template<class T>
__global__ void capture_prefixes(
    lookback::Flag* Fs, const T* As, const T* Ps, Queries queries,
    lookback::LookbackResult<T>* output
) {
    __shared__ lookback::LookbackScratch<T> scratch;
    if (threadIdx.x < kLanes) {
        for (unsigned q = 0; q < kQueries; ++q) {
            const auto result = lookback::warp_prefetch_lookback<T>(
                Fs, As, Ps, queries.ends[q], scratch);
            // Capture every lane's result, including lanes other than lane 0.
            output[q * kLanes + threadIdx.x] = result;
        }
    }
}

template<class T>
unsigned run_type(const char* name) {
    using lookback::Flag;
    using Result = lookback::LookbackResult<T>;
    std::array<T, kDescriptors> aggregates{}, prefixes{};
    std::array<Flag, kDescriptors> flags{};

    // Independent CPU oracle: a forward sum, without using window consumers.
    T sum = T{0};
    for (unsigned i = 0; i < kDescriptors; ++i) {
        aggregates[i] = static_cast<T>(0x0102030400000001ULL + i);
        sum = static_cast<T>(sum + aggregates[i]);
        prefixes[i] = sum;
    }

    DeviceBuffer<Flag> d_flags(kDescriptors);
    DeviceBuffer<T> d_aggregates(kDescriptors), d_prefixes(kDescriptors);
    DeviceBuffer<Result> d_output(kQueries * kLanes);
    std::vector<Result> output(kQueries * kLanes);
    CUDA_CHECK(cudaMemcpy(d_aggregates.ptr, aggregates.data(),
        sizeof(aggregates), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefixes.ptr, prefixes.data(),
        sizeof(prefixes), cudaMemcpyHostToDevice));

    unsigned checks = 0;
    for (unsigned fixture = 0; fixture < 5; ++fixture) {
        Queries queries{{40, 3, 0, 1, 31, 32, 33, 65}};
        flags.fill(Flag::A);
        flags[0] = Flag::P;
        switch (fixture) {
            case 0:
                // A chain of up to 64 aggregates, ending at P in tile 0.
                break;
            case 1:
                flags.fill(Flag::P); // Immediate completion.
                break;
            case 2:
                flags.fill(Flag::A); // Reach the beginning without a P flag.
                break;
            case 3:
                // Older X entries are irrelevant once P at tile 32 is reached.
                // These are synthetic snapshots, not live publication states.
                for (unsigned i = 0; i < 32; ++i) flags[i] = Flag::X;
                flags[32] = Flag::P;
                queries = Queries{{33, 34, 40, 64, 65, 0, 33, 65}};
                // Never query a range that requires one of the static X entries.
                break;
            case 4:
                flags[64] = static_cast<Flag>(99); // Explicit failure path.
                break;
        }
        CUDA_CHECK(cudaMemcpy(d_flags.ptr, flags.data(),
            sizeof(flags), cudaMemcpyHostToDevice));

        for (unsigned block_size : {32u, 128u}) {
            CUDA_CHECK(cudaMemset(d_output.ptr, 0xa5, output.size() * sizeof(Result)));
            capture_prefixes<T><<<1, block_size>>>(d_flags.ptr,
                d_aggregates.ptr, d_prefixes.ptr, queries, d_output.ptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(output.data(), d_output.ptr,
                output.size() * sizeof(Result), cudaMemcpyDeviceToHost));

            for (unsigned q = 0; q < kQueries; ++q) {
                const uint64_t end = queries.ends[q];
                const bool expected_success = !(fixture == 4 && end == 65);
                const T expected_prefix = !expected_success || end == 0
                    ? T{0} : prefixes[static_cast<size_t>(end - 1)];
                for (unsigned lane = 0; lane < kLanes; ++lane) {
                    const auto& actual = output[q * kLanes + lane];
                    if (actual.success != expected_success ||
                        actual.prefix != expected_prefix) {
                        throw std::runtime_error(std::string(name) +
                            " fixture=" + std::to_string(fixture) +
                            " block=" + std::to_string(block_size) +
                            " end=" + std::to_string(end) +
                            " lane=" + std::to_string(lane) + " result mismatch");
                    }
                }
                ++checks;
            }
        }
    }
    std::cout << "PASS " << name << ": " << checks
              << " GPU look-back queries; all 32 lane results checked\n";
    return checks;
}
} // namespace

int main() {
    try {
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::cout << "GPU: " << properties.name << '\n';
        unsigned checks = 0;
        checks += run_type<uint8_t>("u8");
        checks += run_type<uint16_t>("u16");
        checks += run_type<uint32_t>("u32");
        checks += run_type<uint64_t>("u64");
        std::cout << "PASS: " << checks << " GPU look-back queries, "
                  << checks * kLanes << " lane results\n"
                  << "Scope: static descriptors only; no required-X retry or "
                     "concurrent publisher was tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
