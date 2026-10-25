#include "full_scan.cuh"
#include "validation.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Keep CUDA failures visible to the caller instead of continuing with an
// incomplete benchmark or invalid comparison.
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
            throw std::overflow_error("device allocation size overflow");
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

// Warm up before timing, then use CUDA events around repeated launches. The
// launch callable is responsible for keeping work in the measured stream.
template<class Launch>
float measure_ms(Launch&& launch, unsigned iterations) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    try {
        CUDA_CHECK(cudaEventRecord(start));
        for (unsigned i = 0; i < iterations; ++i) {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        if (!(total_ms > 0.0f)) {
            throw std::runtime_error("CUDA event timer reported zero elapsed time");
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return total_ms / static_cast<float>(iterations);
    } catch (...) {
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
        throw;
    }
}

// Reset per-scan state in the same stream as the following kernel launch.
// This keeps repeated benchmark iterations independent and includes setup in
// the measured full-scan operation.
template<class T, bool UseWarpReduction>
void launch_full_scan(
    const T* input,
    T* output,
    uint64_t n,
    uint64_t num_tiles,
    p2::Descriptors<T> descriptors,
    std::size_t shared_bytes,
    cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(
        descriptors.flags, 0,
        static_cast<std::size_t>(num_tiles) * sizeof(lookback::Flag), stream));
    CUDA_CHECK(cudaMemsetAsync(
        descriptors.next_tile, 0, sizeof(unsigned), stream));
    CUDA_CHECK(cudaMemsetAsync(
        descriptors.error, 0, sizeof(unsigned), stream));

    p2::full_kernel<T, 128, 4, UseWarpReduction>
        <<<static_cast<unsigned>(num_tiles), 128, shared_bytes, stream>>>(
            input, output, n, num_tiles, descriptors);
    CUDA_CHECK(cudaGetLastError());
}

// Copy a completed device result to the host and report the first mismatch.
template<class T>
bool validate_result(
    const char* label,
    const T* device_result,
    std::size_t n,
    const std::vector<T>& reference) {
    std::vector<T> actual(n);
    CUDA_CHECK(cudaMemcpy(
        actual.data(), device_result, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::ostringstream errors;
    const bool match = step1_test::compare_exact(
        actual.data(), reference.data(), n, label, errors);
    if (!match) {
        std::cerr << errors.str();
    }
    return match;
}

// Run correctness checks before reporting timings so performance numbers are
// never presented for a scan that disagrees with the CPU reference.
template<class T>
void run(std::size_t n, unsigned iterations) {
    constexpr std::size_t tile_size = 128 * 4;
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "n exceeds the CUB DeviceScan item-count limit");
    }
    const std::size_t num_tiles = step1_test::tile_count(n, tile_size);
    if (num_tiles > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("input requires too many scan tiles");
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (num_tiles > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::overflow_error("input exceeds the device grid limit");
    }

    const auto input = step1_test::generate_dataset<T>(n, "random", 20261025);
    std::vector<T> reference(n);
    step1_test::cpu_scan(input.data(), reference.data(), n);
    if (n == 0) {
        std::cout << "n=0: empty input; no benchmark kernels launched\n";
        return;
    }

    DeviceBuffer<T> d_input(n), d_full_serial(n), d_full_reduced(n),
        d_cub_output(n), d_copy(n), d_aggregates(num_tiles), d_prefixes(num_tiles);
    DeviceBuffer<lookback::Flag> d_flags(num_tiles);
    DeviceBuffer<unsigned> d_next(1), d_error(1);
    CUDA_CHECK(cudaMemcpy(
        d_input.get(), input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    p2::Descriptors<T> descriptors{
        d_flags.get(), d_aggregates.get(), d_prefixes.get(),
        d_next.get(), d_error.get()};
    constexpr std::size_t shared_bytes =
        sizeof(step1::BlockScanScratch<T, 128, 4>);
    const cudaStream_t stream = nullptr;

    const float serial_ms = measure_ms([&] {
        launch_full_scan<T, false>(
            d_input.get(), d_full_serial.get(), n, num_tiles,
            descriptors, shared_bytes, stream);
    }, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned protocol_error = 0;
    CUDA_CHECK(cudaMemcpy(
        &protocol_error, d_error.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (protocol_error != 0) {
        throw std::runtime_error("serial look-back reported invalid descriptor state");
    }
    unsigned assigned_tiles = 0;
    CUDA_CHECK(cudaMemcpy(
        &assigned_tiles, d_next.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (assigned_tiles != num_tiles) {
        throw std::runtime_error("serial full scan did not assign every tile");
    }
    if (!validate_result("serial full scan", d_full_serial.get(), n, reference)) {
        throw std::runtime_error("serial full scan differs from CPU reference");
    }

    const float reduced_ms = measure_ms([&] {
        launch_full_scan<T, true>(
            d_input.get(), d_full_reduced.get(), n, num_tiles,
            descriptors, shared_bytes, stream);
    }, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        &protocol_error, d_error.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (protocol_error != 0) {
        throw std::runtime_error("warp-reduction look-back reported invalid descriptor state");
    }
    CUDA_CHECK(cudaMemcpy(
        &assigned_tiles, d_next.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (assigned_tiles != num_tiles) {
        throw std::runtime_error("warp-reduction full scan did not assign every tile");
    }
    if (!validate_result("warp-reduction full scan",
                         d_full_reduced.get(), n, reference)) {
        throw std::runtime_error("warp-reduction full scan differs from CPU reference");
    }

    std::size_t cub_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, cub_temp_bytes, d_input.get(), d_cub_output.get(), n, stream));
    DeviceBuffer<unsigned char> d_cub_temp(cub_temp_bytes);
    const float cub_ms = measure_ms([&] {
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            d_cub_temp.get(), cub_temp_bytes, d_input.get(), d_cub_output.get(),
            n, stream));
    }, iterations);
    if (!validate_result("CUB scan", d_cub_output.get(), n, reference)) {
        throw std::runtime_error("CUB scan differs from CPU reference");
    }

    const float memcpy_ms = measure_ms([&] {
        CUDA_CHECK(cudaMemcpyAsync(
            d_copy.get(), d_input.get(), n * sizeof(T),
            cudaMemcpyDeviceToDevice, stream));
    }, iterations);

    // Count one input read and one output write for scans. The copy metric
    // counts only its payload, so the two bandwidth figures describe
    // different byte counts.
    const double input_bytes = static_cast<double>(n) * sizeof(T);
    const double scan_bytes = input_bytes * 2.0;
    const auto bandwidth = [](double bytes, float ms) {
        return bytes / (static_cast<double>(ms) * 1.0e6);
    };
    std::cout << "n=" << n << " type=u" << sizeof(T) * 8
              << " tiles=" << num_tiles << " iterations=" << iterations << '\n'
              << "  full serial:       " << serial_ms << " ms, effective "
              << bandwidth(scan_bytes, serial_ms) << " GB/s\n"
              << "  full warp reduce:  " << reduced_ms << " ms, effective "
              << bandwidth(scan_bytes, reduced_ms) << " GB/s\n"
              << "  CUB DeviceScan:    " << cub_ms << " ms, effective "
              << bandwidth(scan_bytes, cub_ms) << " GB/s\n"
              << "  D2D memcpy:        " << memcpy_ms << " ms, payload "
              << bandwidth(input_bytes, memcpy_ms) << " GB/s\n";
}

uint64_t parse_number(const char* text, const char* label) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string(label) + " must be a nonnegative integer");
    }
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    try {
        // Keep command-line parsing strict so malformed sizes and iteration
        // counts fail before any device allocation or launch.
        if (argc < 2 || argc > 4) {
            std::cerr << "Usage: p2_scan <n> [u8|u16|u32|u64] [iterations]\n";
            return 2;
        }
        const uint64_t parsed_n = parse_number(argv[1], "n");
        if (parsed_n > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("n exceeds host size_t");
        }
        const std::string type = argc >= 3 ? argv[2] : "u32";
        const uint64_t parsed_iterations = argc >= 4
            ? parse_number(argv[3], "iterations") : 10;
        if (parsed_iterations == 0 ||
            parsed_iterations > std::numeric_limits<unsigned>::max()) {
            throw std::invalid_argument("iterations must be in 1..UINT_MAX");
        }
        const auto n = static_cast<std::size_t>(parsed_n);
        const auto iterations = static_cast<unsigned>(parsed_iterations);

        if (type == "u8") run<uint8_t>(n, iterations);
        else if (type == "u16") run<uint16_t>(n, iterations);
        else if (type == "u32") run<uint32_t>(n, iterations);
        else if (type == "u64") run<uint64_t>(n, iterations);
        else throw std::invalid_argument("type must be u8, u16, u32, or u64");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
