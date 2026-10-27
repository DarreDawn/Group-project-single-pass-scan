#include "full_scan.cuh"
#include "validation.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Keep CUDA failures explicit so a failed allocation, launch, or copy cannot
// be mistaken for a correctness-test result.
void check_cuda(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(expression) + " at line " + std::to_string(line) +
            ": " + cudaGetErrorString(result));
    }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __LINE__)

template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device buffer size overflow");
        }
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        }
    }
    ~DeviceBuffer() { if (ptr_) (void)cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

constexpr int B = 128;
constexpr int Q = 4;

// Run one input through a selected P2/P3 look-back strategy and compare every
// output element with the independent CPU inclusive-scan reference.
template<class T, p2::ScanStrategy Strategy>
bool run_case(std::size_t n, const std::string& pattern, uint64_t seed) {
    const std::size_t tile_size = static_cast<std::size_t>(B) * Q;
    const std::size_t num_tiles = step1_test::tile_count(n, tile_size);
    if (num_tiles > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("test grid exceeds dynamic counter range");
    }

    const auto input = step1_test::generate_dataset<T>(n, pattern, seed);
    std::vector<T> expected(n);
    step1_test::cpu_scan(input.data(), expected.data(), n);
    std::vector<T> actual(n);

    if (n != 0) {
        DeviceBuffer<T> d_input(n), d_output(n), d_aggregates(num_tiles),
            d_prefixes(num_tiles);
        DeviceBuffer<lookback::Flag> d_flags(num_tiles);
        DeviceBuffer<uint64_t> d_packed(num_tiles);
        DeviceBuffer<unsigned> d_next(1), d_error(1);

        CUDA_CHECK(cudaMemcpy(
            d_input.get(), input.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_flags.get(), 0,
                              num_tiles * sizeof(lookback::Flag)));
        CUDA_CHECK(cudaMemset(d_packed.get(), 0,
                              num_tiles * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_next.get(), 0, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(d_error.get(), 0, sizeof(unsigned)));

        p2::Descriptors<T> descriptors{
            d_flags.get(), d_aggregates.get(), d_prefixes.get(),
            d_next.get(), d_error.get(), d_packed.get()};
        constexpr std::size_t shared_bytes =
            sizeof(step1::BlockScanScratch<T, B, Q>);

        p2::full_kernel<T, B, Q, Strategy>
            <<<static_cast<unsigned>(num_tiles), B, shared_bytes>>>(
                d_input.get(), d_output.get(), n, num_tiles, descriptors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(), d_output.get(), n * sizeof(T),
            cudaMemcpyDeviceToHost));

        unsigned assigned = 0;
        unsigned error = 0;
        CUDA_CHECK(cudaMemcpy(&assigned, d_next.get(), sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&error, d_error.get(), sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        if (assigned != num_tiles || error != 0) {
            std::cerr << "P2 protocol failure: assigned=" << assigned
                      << " expected=" << num_tiles << " error=" << error
                      << '\n';
            return false;
        }
    }

    std::ostringstream errors;
    const bool match = step1_test::compare_exact(
        actual.data(), expected.data(), n, "P2 scan", errors);
    if (!match) {
        std::cerr << errors.str();
    }
    return match;
}

// Exercise empty input, short tails, tile boundaries, unsigned overflow, and
// a multi-window look-back for every integrated look-back strategy.
template<class T>
bool run_type(const char* name) {
    struct Case {
        std::size_t n;
        const char* pattern;
        uint64_t seed;
    };
    const Case cases[] = {
        {0, "zeros", 1},
        {1, "ones", 2},
        {511, "random", 3},
        {512, "impulse", 4},
        {513, "max", 5},
        {4099, "random", 6},
        {66559, "ones", 7}
    };
    bool ok = true;
    for (const auto& test : cases) {
        const bool baseline = run_case<T, p2::ScanStrategy::Baseline>(
            test.n, test.pattern, test.seed);
        const bool prefetch = run_case<T, p2::ScanStrategy::WarpPrefetch>(
            test.n, test.pattern, test.seed);
        const bool reduced = run_case<T, p2::ScanStrategy::WarpReduce>(
            test.n, test.pattern, test.seed);
        const bool fencefree = run_case<T, p2::ScanStrategy::FenceFree>(
            test.n, test.pattern, test.seed);
        if (!baseline || !prefetch || !reduced || !fencefree) {
            std::cerr << name << " failed at n=" << test.n
                      << " pattern=" << test.pattern << '\n';
            ok = false;
        }
    }
    return ok;
}

} // namespace

int main() {
    try {
        // The test initializes the X state with cudaMemset, so lock down the
        // shared status encoding used by P1 and P2.
        static_assert(static_cast<uint32_t>(lookback::Flag::X) == 0,
                      "host initialization requires X to be zero");
        bool ok = true;
        ok = run_type<uint8_t>("u8") && ok;
        ok = run_type<uint16_t>("u16") && ok;
        ok = run_type<uint32_t>("u32") && ok;
        ok = run_type<uint64_t>("u64") && ok;
        std::cout << (ok ? "PASS" : "FAIL")
                  << ": P2 dynamic assignment and scan tests\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
