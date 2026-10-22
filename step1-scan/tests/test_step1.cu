#include "block_scan.cuh"
#include "validation.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Correctness runner, NOT a benchmark: copies, allocations and synchronization
// are intentional. The CPU-supplied prefix tests do NOT implement Step2/Step3.
namespace {

void check_cuda(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " at line " +
            std::to_string(line) + ": " + cudaGetErrorString(result));
    }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __LINE__)

template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("Device allocation size overflow");
        }
        if (count != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) (void)cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

struct Options {
    std::string suite = "quick";
    std::string type;                // Empty = u32 for quick, all for full.
    std::string pattern = "all";
    uint64_t seed = 1;
    uint64_t n = 0;
    unsigned repeat = 1;
    int block = 0, items = 0, device = 0;
    bool custom_n = false, verbose = false;
};

void usage() {
    std::cout <<
        "Step1 correctness tests (unsigned inclusive addition)\n"
        "  --suite quick|full     quick: B=128,Q=4,u32; full: all built-in configurations\n"
        "  --type u8|u16|u32|u64|all\n"
        "  --block 32|128|256|1024\n"
        "  --items 1|4|8\n"
        "  --n N                  replace the suite's lengths with one exact length\n"
        "  --pattern zeros|ones|ramp|random|impulse|max|all\n"
        "  --seed N               deterministic 64-bit seed (default 1)\n"
        "  --repeat N             repeat with seed,seed+1,... (default 1)\n"
        "  --device N             CUDA device index (default 0)\n"
        "  --verbose              print every successful kernel check\n"
        "  --help\n"
        "Each dataset tests the standalone wrapper, all-thread aggregates,\n"
        "zero/constant/CPU-supplied prefixes and forward/reversed tile assignment.\n";
}

uint64_t number(const std::string& value, const std::string& option) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(option + " expects a nonnegative decimal integer");
    }
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size()) throw std::invalid_argument("Invalid number for " + option);
    return result;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") { usage(); std::exit(0); }
        if (option == "--verbose") { o.verbose = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--suite") o.suite = value;
        else if (option == "--type") o.type = value;
        else if (option == "--pattern") o.pattern = value;
        else if (option == "--seed") o.seed = number(value, option);
        else if (option == "--n") { o.n = number(value, option); o.custom_n = true; }
        else if (option == "--repeat") {
            const auto n = number(value, option);
            if (n == 0 || n > 100000) throw std::invalid_argument("repeat must be 1..100000");
            o.repeat = static_cast<unsigned>(n);
        } else if (option == "--block" || option == "--items" || option == "--device") {
            const auto n = number(value, option);
            if (n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument(option + " is too large");
            }
            if (option == "--block") o.block = static_cast<int>(n);
            if (option == "--items") o.items = static_cast<int>(n);
            if (option == "--device") o.device = static_cast<int>(n);
        } else throw std::invalid_argument("Unknown option: " + option);
    }
    if (o.suite != "quick" && o.suite != "full") throw std::invalid_argument("Invalid suite");
    if (o.type.empty()) o.type = o.suite == "full" ? "all" : "u32";
    if (o.type != "all" && o.type != "u8" && o.type != "u16" && o.type != "u32" && o.type != "u64") {
        throw std::invalid_argument("Invalid type");
    }
    if (o.block != 0 && o.block != 32 && o.block != 128 && o.block != 256 && o.block != 1024) {
        throw std::invalid_argument("Supported block sizes: 32,128,256,1024");
    }
    if (o.items != 0 && o.items != 1 && o.items != 4 && o.items != 8) {
        throw std::invalid_argument("Supported items/thread: 1,4,8");
    }
    const auto& patterns = step1_test::patterns();
    if (o.pattern != "all" && std::find(patterns.begin(), patterns.end(), o.pattern) == patterns.end()) {
        throw std::invalid_argument("Invalid pattern");
    }
    if (o.seed > std::numeric_limits<uint64_t>::max() - (o.repeat - 1)) {
        throw std::invalid_argument("seed + repeat would overflow");
    }
    return o;
}

struct Stats { uint64_t kernels = 0, empty = 0, skipped = 0; };

// Instrumented integration probe. There are NO cross-block dependencies.
// Reverse assignment ensures that the core uses tile_id rather than blockIdx.x.
// Every thread records its aggregate, checking the documented all-thread result.
template<class T, int B, int Q>
__global__ void probe_kernel(const T* input, T* output, T* totals,
                             T* thread_aggregates, const T* prefixes,
                             uint64_t n, bool reverse) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& scratch = *reinterpret_cast<step1::BlockScanScratch<T, B, Q>*>(storage);
    const uint64_t tile_id = reverse ? uint64_t(gridDim.x) - 1 - blockIdx.x : blockIdx.x;
    T items[Q];
    const T aggregate = step1::scan_local_tile<T, B, Q>(input, n, tile_id, items, scratch);
    thread_aggregates[tile_id * B + threadIdx.x] = aggregate;
    if (threadIdx.x == 0) totals[tile_id] = aggregate;
    // Reading precomputed CPU prefixes is a TEST FIXTURE, not a scan algorithm.
    const T prefix = prefixes[tile_id];
    step1::store_tile_with_prefix<T, B, Q>(output, n, tile_id, prefix, items, scratch);
}

std::vector<std::size_t> lengths(const Options& o, std::size_t b, std::size_t q) {
    if (o.custom_n) {
        if (o.n > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("n too large");
        return {static_cast<std::size_t>(o.n)};
    }
    const auto s = b * q;
    std::vector<std::size_t> ns{0,1,31,32,33,b-1,b,b+1,s-1,s,s+1,2*s-1,2*s,2*s+3};
    if (o.suite == "full") { ns.push_back(65537); ns.push_back(1048583); }
    std::sort(ns.begin(), ns.end());
    ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
    return ns;
}

template<class T, int B, int Q>
void run_dataset(const Options& o, const cudaDeviceProp& prop, Stats& stats,
                 std::size_t n, const std::string& pattern, uint64_t seed) {
    constexpr std::size_t tile = B * Q;
    constexpr std::size_t guard = 32;
    constexpr T canary = static_cast<T>(0xa5a5a5a5a5a5a5a5ULL);
    const auto blocks = step1_test::tile_count(n, tile);
    if (blocks > static_cast<uint64_t>(prop.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned>::max()) throw std::runtime_error("Grid too large");
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T) - 2 * guard ||
        blocks > std::numeric_limits<std::size_t>::max() / sizeof(T) / B) {
        throw std::overflow_error("Test buffer size overflow");
    }

    const auto input = step1_test::generate_dataset<T>(n, pattern, seed);
    std::vector<T> local(n), global(n), expected_totals(blocks);
    step1_test::cpu_scan(input.data(), global.data(), n);
    step1_test::cpu_block_scan(input.data(), local.data(), expected_totals.data(), n, tile);
    if (n == 0) {
        // Explicit contract: no zero-grid launch and no dereference for n == 0.
        ++stats.empty;
        return;
    }

    DeviceBuffer<T> d_input(n), d_output(n + 2 * guard), d_totals(blocks);
    DeviceBuffer<T> d_threads(blocks * B), d_prefixes(blocks);
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    std::vector<T> output(n + 2 * guard), actual_totals(blocks), actual_threads(blocks * B);
    std::vector<T> prefixes(blocks), expected(n);
    constexpr auto shared_bytes = sizeof(step1::BlockScanScratch<T, B, Q>);

    // mode -1 checks the public standalone wrapper; 0/1/2 check core integration.
    for (int mode = -1; mode < 3; ++mode) {
        for (std::size_t k = 0; k < blocks; ++k) {
            prefixes[k] = mode == 1 ? T{7} :
                (mode == 2 && k != 0 ? global[k * tile - 1] : T{0});
        }
        for (std::size_t i = 0; i < n; ++i) {
            expected[i] = mode == 2 ? global[i] :
                static_cast<T>(local[i] + (mode == 1 ? T{7} : T{0}));
        }
        CUDA_CHECK(cudaMemcpy(d_prefixes.get(), prefixes.data(), blocks * sizeof(T), cudaMemcpyHostToDevice));
        const int assignments = mode == -1 ? 1 : 2;
        for (int reverse = 0; reverse < assignments; ++reverse) {
            std::fill(output.begin(), output.end(), canary);
            CUDA_CHECK(cudaMemcpy(d_output.get(), output.data(), output.size() * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_totals.get(), 0xcd, blocks * sizeof(T)));
            CUDA_CHECK(cudaMemset(d_threads.get(), 0xcd, blocks * B * sizeof(T)));
            if (mode == -1) {
                step1::block_local_scan_kernel<T, B, Q><<<static_cast<unsigned>(blocks), B, shared_bytes>>>(
                    d_input.get(), d_output.get() + guard, d_totals.get(), n);
            } else {
                probe_kernel<T, B, Q><<<static_cast<unsigned>(blocks), B, shared_bytes>>>(
                    d_input.get(), d_output.get() + guard, d_totals.get(), d_threads.get(),
                    d_prefixes.get(), n, reverse != 0);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), output.size() * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(actual_totals.data(), d_totals.get(), blocks * sizeof(T), cudaMemcpyDeviceToHost));
            if (!step1_test::compare_exact(output.data() + guard, expected.data(), n, "output", std::cerr) ||
                !step1_test::compare_exact(actual_totals.data(), expected_totals.data(), blocks, "block_totals", std::cerr)) {
                throw std::runtime_error("Result mismatch: prefix_mode=" + std::to_string(mode) +
                                         " reverse=" + std::to_string(reverse));
            }
            for (std::size_t i = 0; i < guard; ++i) {
                if (output[i] != canary || output[guard + n + i] != canary) {
                    throw std::runtime_error("Output guard overwritten (out-of-range store)");
                }
            }
            if (mode != -1) {
                CUDA_CHECK(cudaMemcpy(actual_threads.data(), d_threads.get(), actual_threads.size() * sizeof(T), cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < actual_threads.size(); ++i) {
                    if (actual_threads[i] != expected_totals[i / B]) {
                        std::cerr << "aggregate: tile=" << i / B << " tid=" << i % B
                                  << " expected=" << uint64_t(expected_totals[i / B])
                                  << " actual=" << uint64_t(actual_threads[i]) << '\n';
                        throw std::runtime_error("Aggregate is not valid in every thread");
                    }
                }
            }
            ++stats.kernels;
            if (o.verbose) std::cout << "PASS u" << sizeof(T)*8 << " B=" << B << " Q=" << Q
                << " n=" << n << " pattern=" << pattern << " seed=" << seed
                << " mode=" << mode << " reverse=" << reverse << '\n';
        }
    }
    std::vector<T> input_after(n);
    CUDA_CHECK(cudaMemcpy(input_after.data(), d_input.get(), n * sizeof(T), cudaMemcpyDeviceToHost));
    if (!step1_test::compare_exact(input_after.data(), input.data(), n, "input_modified", std::cerr)) {
        throw std::runtime_error("The input buffer was modified");
    }
}

template<class T, int B, int Q>
void run_config(const Options& o, const cudaDeviceProp& prop, Stats& stats) {
    cudaFuncAttributes wrapper{}, probe{};
    CUDA_CHECK(cudaFuncGetAttributes(&wrapper, step1::block_local_scan_kernel<T, B, Q>));
    CUDA_CHECK(cudaFuncGetAttributes(&probe, probe_kernel<T, B, Q>));
    const auto bytes = sizeof(step1::BlockScanScratch<T, B, Q>);
    if (B > prop.maxThreadsPerBlock || B > wrapper.maxThreadsPerBlock || B > probe.maxThreadsPerBlock ||
        bytes + std::max(wrapper.sharedSizeBytes, probe.sharedSizeBytes) > prop.sharedMemPerBlock) {
        ++stats.skipped;
        std::cout << "SKIP u" << sizeof(T)*8 << " B=" << B << " Q=" << Q
                  << " shared=" << bytes << ": exceeds this GPU's default resource limits\n";
        return;
    }
    std::cout << "CONFIG u" << sizeof(T)*8 << " B=" << B << " Q=" << Q
              << " shared=" << bytes << " bytes\n";
    const auto ns = lengths(o, B, Q);
    const auto ps = o.pattern == "all" ? step1_test::patterns() : std::vector<std::string>{o.pattern};
    for (unsigned iteration = 0; iteration < o.repeat; ++iteration) {
        const auto seed = o.seed + iteration;
        for (const auto n : ns) for (const auto& pattern : ps) {
            try {
                run_dataset<T, B, Q>(o, prop, stats, n, pattern, seed);
            } catch (...) {
                std::cerr << "REPRODUCE: ./test_step1 --type u" << sizeof(T)*8
                          << " --block " << B << " --items " << Q << " --n " << n
                          << " --pattern " << pattern << " --seed " << seed
                          << " --device " << o.device << " --verbose\n";
                throw;
            }
        }
    }
}

template<class T, int B>
void dispatch_items(const Options& o, const cudaDeviceProp& prop, Stats& stats) {
    const auto qs = o.items ? std::vector<int>{o.items} :
        (o.suite == "full" ? std::vector<int>{1,4,8} : std::vector<int>{4});
    for (int q : qs) {
        if (q == 1) run_config<T,B,1>(o,prop,stats);
        if (q == 4) run_config<T,B,4>(o,prop,stats);
        if (q == 8) run_config<T,B,8>(o,prop,stats);
    }
}

template<class T>
void dispatch_blocks(const Options& o, const cudaDeviceProp& prop, Stats& stats) {
    const auto bs = o.block ? std::vector<int>{o.block} :
        (o.suite == "full" ? std::vector<int>{32,128,256,1024} : std::vector<int>{128});
    for (int b : bs) {
        if (b == 32) dispatch_items<T,32>(o,prop,stats);
        if (b == 128) dispatch_items<T,128>(o,prop,stats);
        if (b == 256) dispatch_items<T,256>(o,prop,stats);
        if (b == 1024) dispatch_items<T,1024>(o,prop,stats);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse_options(argc, argv);
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (o.device >= device_count) throw std::runtime_error("Requested CUDA device is unavailable");
        CUDA_CHECK(cudaSetDevice(o.device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
        if (prop.warpSize != 32) throw std::runtime_error("This module requires 32-thread warps");
        int runtime_version = 0, driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        std::cout << "GPU: " << prop.name << " compute=" << prop.major << '.' << prop.minor
                  << " runtime=" << runtime_version << " driver=" << driver_version << '\n'
                  << "suite=" << o.suite << " seed=" << o.seed << " repeat=" << o.repeat << '\n';
        Stats stats;
        if (o.type == "all" || o.type == "u8") dispatch_blocks<uint8_t>(o,prop,stats);
        if (o.type == "all" || o.type == "u16") dispatch_blocks<uint16_t>(o,prop,stats);
        if (o.type == "all" || o.type == "u32") dispatch_blocks<uint32_t>(o,prop,stats);
        if (o.type == "all" || o.type == "u64") dispatch_blocks<uint64_t>(o,prop,stats);
        if (stats.kernels == 0 && stats.empty == 0) throw std::runtime_error("No supported configuration was tested");
        std::cout << "PASS: " << stats.kernels << " GPU kernel checks, " << stats.empty
                  << " empty-input host checks; SKIP configurations=" << stats.skipped << '\n';
        std::cout << "CPU-prefix fixtures test the Step1 interface only; no look-back was executed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
