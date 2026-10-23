#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace step1_test {

template<class T>
constexpr void require_type() {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T> &&
                  !std::is_same_v<T, bool> && sizeof(T) <= 8,
                  "References support unsigned integers up to 64 bits");
}

inline std::size_t tile_count(std::size_t n, std::size_t tile_size) {
    if (tile_size == 0) throw std::invalid_argument("tile_size must be positive");
    // Avoid overflow from the conventional (n + tile_size - 1) / tile_size.
    return n / tile_size + (n % tile_size != 0);
}

template<class T>
void cpu_scan(const T* input, T* output, std::size_t n) {
    require_type<T>();
    if (n != 0 && (!input || !output)) {
        throw std::invalid_argument("cpu_scan: null nonempty array");
    }
    T acc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        acc = static_cast<T>(acc + input[i]);
        output[i] = acc;
    }
}

// Reset at every tile boundary. block_totals has tile_count(n,tile_size) entries.
template<class T>
void cpu_block_scan(const T* input, T* output, T* block_totals,
                    std::size_t n, std::size_t tile_size) {
    require_type<T>();
    (void)tile_count(n, tile_size); // Validate tile_size even for empty input.
    if (n != 0 && (!input || !output || !block_totals)) {
        throw std::invalid_argument("cpu_block_scan: null nonempty array");
    }
    std::size_t block = 0;
    for (std::size_t begin = 0; begin < n; ++block) {
        const std::size_t count = std::min(tile_size, n - begin);
        T acc = 0;
        for (std::size_t j = 0; j < count; ++j) {
            acc = static_cast<T>(acc + input[begin + j]);
            output[begin + j] = acc;
        }
        block_totals[block] = acc;
        begin += count;
    }
}

inline const std::vector<std::string>& patterns() {
    static const std::vector<std::string> names{
        "zeros", "ones", "ramp", "random", "impulse", "max"};
    return names;
}

// Same pattern/seed/type/n reproduces the same data. Random uses the defined
// mt19937_64 engine directly, without implementation-dependent distributions.
template<class T>
std::vector<T> generate_dataset(std::size_t n, const std::string& pattern,
                                uint64_t seed) {
    require_type<T>();
    if (std::find(patterns().begin(), patterns().end(), pattern) == patterns().end()) {
        throw std::invalid_argument("Unknown pattern: " + pattern);
    }
    std::vector<T> data(n, T{0});
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        if (pattern == "ones") data[i] = T{1};
        else if (pattern == "ramp") data[i] = static_cast<T>(i % 19);
        else if (pattern == "random") data[i] = static_cast<T>(rng());
        else if (pattern == "max") data[i] = std::numeric_limits<T>::max();
    }
    if (pattern == "impulse" && n != 0) data[n / 2] = T{13};
    return data;
}

template<class T>
bool compare_exact(const T* actual, const T* expected, std::size_t n,
                   const std::string& label, std::ostream& errors) {
    require_type<T>();
    if (n != 0 && (!actual || !expected)) {
        throw std::invalid_argument("compare_exact: null nonempty array");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            errors << label << ": first mismatch at index=" << i
                   << " expected=" << static_cast<uint64_t>(expected[i])
                   << " actual=" << static_cast<uint64_t>(actual[i]) << '\n';
            return false;
        }
    }
    return true;
}

} // namespace step1_test
