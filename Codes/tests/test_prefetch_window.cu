#include "warp_prefetch.cuh"

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kDescriptors = 65;
constexpr unsigned kWindows = 8;
constexpr unsigned kLanes = 32;

// Full -> partial -> empty explicitly exercises shared-memory reuse.
struct WindowCases {
    uint64_t ends[kWindows];
};

void check_cuda(cudaError_t status, const char* operation, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " at line " + std::to_string(line) +
            ": " + cudaGetErrorString(status));
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

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Only the first warp calls the helper, even when the block has 128 threads.
// Each participating lane copies the WHOLE window, testing visibility beyond
// the entry written by that lane. The next call reuses the same shared storage.
template<class T>
__global__ void capture_windows(
    lookback::Flag* Fs,
    const T* As,
    const T* Ps,
    WindowCases cases,
    lookback::PrefetchWindow<T>* snapshots
) {
    __shared__ lookback::PrefetchWindow<T> window;

    if (threadIdx.x < kLanes) {
        const unsigned lane = threadIdx.x;
        for (unsigned c = 0; c < kWindows; ++c) {
            lookback::prefetch_lookback_window<T>(
                Fs, As, Ps, cases.ends[c], window);

            snapshots[c * kLanes + lane] = window;
            // The helper's entry warp barrier protects the preceding reads
            // before another lane starts overwriting the window on the next call.
        }
    }
}

lookback::Flag fixture_flag(unsigned pattern, unsigned tile) {
    using lookback::Flag;
    switch (pattern) {
        case 0: return Flag::X;
        case 1: return Flag::A;
        case 2: return Flag::P;
        default:
            return tile % 3 == 0 ? Flag::P
                 : tile % 3 == 1 ? Flag::A : Flag::X;
    }
}

// Independent CPU oracle: do not call the indexing helpers under test.
template<class T>
lookback::PrefetchWindow<T> expected_window(
    const std::array<lookback::Flag, kDescriptors>& flags,
    const std::array<T, kDescriptors>& aggregates,
    const std::array<T, kDescriptors>& prefixes,
    uint64_t end
) {
    lookback::PrefetchWindow<T> result{};
    const uint64_t begin = end > kLanes ? end - kLanes : 0;
    unsigned slot = 0;

    for (uint64_t tile = end; tile > begin;) {
        --tile;
        result.flags[slot] = flags[tile];
        if (flags[tile] == lookback::Flag::A) {
            result.aggregates[slot] = aggregates[tile];
        } else if (flags[tile] == lookback::Flag::P) {
            result.prefixes[slot] = prefixes[tile];
        }
        ++slot;
    }

    result.valid_count = slot;
    return result;
}

template<class T>
void compare_window(
    const lookback::PrefetchWindow<T>& actual,
    const lookback::PrefetchWindow<T>& expected,
    const std::string& context
) {
    if (actual.valid_count != expected.valid_count) {
        throw std::runtime_error(context + ": valid_count=" +
            std::to_string(actual.valid_count) + ", expected=" +
            std::to_string(expected.valid_count));
    }

    // Check all slots, including padding and unused payload fields.
    // Compare fields rather than bytes: struct padding is not part of the API.
    for (unsigned slot = 0; slot < kLanes; ++slot) {
        if (actual.flags[slot] != expected.flags[slot] ||
            actual.aggregates[slot] != expected.aggregates[slot] ||
            actual.prefixes[slot] != expected.prefixes[slot]) {
            std::ostringstream message;
            message << context << ": slot=" << slot
                    << " actual(flag,A,P)=("
                    << static_cast<unsigned>(actual.flags[slot]) << ','
                    << static_cast<uint64_t>(actual.aggregates[slot]) << ','
                    << static_cast<uint64_t>(actual.prefixes[slot]) << ')'
                    << " expected=("
                    << static_cast<unsigned>(expected.flags[slot]) << ','
                    << static_cast<uint64_t>(expected.aggregates[slot]) << ','
                    << static_cast<uint64_t>(expected.prefixes[slot]) << ')';
            throw std::runtime_error(message.str());
        }
    }
}

template<class T>
void run_type(const char* type_name) {
    using Window = lookback::PrefetchWindow<T>;
    const WindowCases cases{{40, 3, 0, 1, 31, 32, 33, 65}};
    const char* pattern_names[] = {"all-X", "all-A", "all-P", "mixed-PAX"};
    const unsigned block_sizes[] = {32, 128};

    std::array<lookback::Flag, kDescriptors> flags{};
    std::array<T, kDescriptors> aggregates{};
    std::array<T, kDescriptors> prefixes{};
    std::vector<Window> snapshots(kWindows * kLanes);

    DeviceBuffer<lookback::Flag> d_flags(kDescriptors);
    DeviceBuffer<T> d_aggregates(kDescriptors);
    DeviceBuffer<T> d_prefixes(kDescriptors);
    DeviceBuffer<Window> d_snapshots(snapshots.size());

    for (unsigned tile = 0; tile < kDescriptors; ++tile) {
        // Different payloads catch reading As where Ps was required (or vice
        // versa). High bits also exercise actual 64-bit payload transfers.
        aggregates[tile] = static_cast<T>(0x1234567800000000ULL + 17 * tile + 1);
        prefixes[tile] = static_cast<T>(0xABCDEF1200000000ULL + 19 * tile + 101);
    }

    CUDA_CHECK(cudaMemcpy(d_aggregates.ptr, aggregates.data(),
        kDescriptors * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefixes.ptr, prefixes.data(),
        kDescriptors * sizeof(T), cudaMemcpyHostToDevice));

    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        for (unsigned tile = 0; tile < kDescriptors; ++tile) {
            flags[tile] = fixture_flag(pattern, tile);
        }

        // Initialize before launch. No publisher changes these fixtures while
        // the kernel runs; this test does not validate concurrent publication.
        CUDA_CHECK(cudaMemcpy(d_flags.ptr, flags.data(),
            kDescriptors * sizeof(lookback::Flag), cudaMemcpyHostToDevice));

        for (unsigned block_size : block_sizes) {
            // Poison destination memory so unwritten fields cannot silently
            // inherit a valid result from an earlier kernel invocation.
            CUDA_CHECK(cudaMemset(d_snapshots.ptr, 0xA5,
                snapshots.size() * sizeof(Window)));

            capture_windows<T><<<1, block_size>>>(
                d_flags.ptr, d_aggregates.ptr, d_prefixes.ptr,
                cases, d_snapshots.ptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(snapshots.data(), d_snapshots.ptr,
                snapshots.size() * sizeof(Window), cudaMemcpyDeviceToHost));

            for (unsigned c = 0; c < kWindows; ++c) {
                const Window expected = expected_window<T>(
                    flags, aggregates, prefixes, cases.ends[c]);
                for (unsigned lane = 0; lane < kLanes; ++lane) {
                    const std::string context = std::string(type_name) + " " +
                        pattern_names[pattern] + " B=" + std::to_string(block_size) +
                        " end=" + std::to_string(cases.ends[c]) +
                        " observer_lane=" + std::to_string(lane);
                    compare_window(snapshots[c * kLanes + lane], expected, context);
                }
            }
        }
    }

    std::cout << "PASS " << type_name
              << ": 64 GPU prefetch windows; all 32 lane snapshots checked\n";
}

} // namespace

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << '\n';

        run_type<uint8_t>("u8");
        run_type<uint16_t>("u16");
        run_type<uint32_t>("u32");
        run_type<uint64_t>("u64");

        std::cout << "PASS: 256 GPU prefetch windows, 8192 lane snapshots\n"
                  << "Scope: static descriptors only; no concurrent publisher "
                     "or complete look-back was tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
