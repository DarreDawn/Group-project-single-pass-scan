#include "validation.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int checks = 0;
void expect(bool condition, const std::string& description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

template<class T>
void check_type() {
    // Hand-computed values: this test does not duplicate the GPU algorithm.
    const std::vector<T> input{3,1,4,2,5,1,2};
    std::vector<T> global(7), local(7), totals(2);
    step1_test::cpu_scan(input.data(), global.data(), input.size());
    step1_test::cpu_block_scan(input.data(), local.data(), totals.data(), input.size(), 4);
    expect(global == std::vector<T>({3,4,8,10,15,16,18}), "global hand example");
    expect(local == std::vector<T>({3,4,8,10,5,6,8}), "local hand example including tail");
    expect(totals == std::vector<T>({10,8}), "hand-computed block totals");

    const T m = std::numeric_limits<T>::max();
    const std::vector<T> wrap_input{m,1,2,m,4};
    std::vector<T> wrap_global(5), wrap_local(5), wrap_totals(2);
    step1_test::cpu_scan(wrap_input.data(), wrap_global.data(), 5);
    step1_test::cpu_block_scan(wrap_input.data(), wrap_local.data(), wrap_totals.data(), 5, 3);
    expect(wrap_global == std::vector<T>({m,0,2,1,5}), "global modular overflow");
    expect(wrap_local == std::vector<T>({m,0,2,m,3}), "local modular overflow");
    expect(wrap_totals == std::vector<T>({2,3}), "modular block totals");

    // Empty arrays permit nullptr, but a zero tile size is always invalid.
    step1_test::cpu_scan<T>(nullptr, nullptr, 0);
    step1_test::cpu_block_scan<T>(nullptr, nullptr, nullptr, 0, 512);
    bool threw = false;
    try { step1_test::cpu_block_scan<T>(nullptr,nullptr,nullptr,0,0); }
    catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "zero tile size must be rejected");
    threw = false;
    try { step1_test::cpu_scan<T>(nullptr,nullptr,1); }
    catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "nonempty null input must be rejected");

    const auto a = step1_test::generate_dataset<T>(64, "random", 12345);
    const auto b = step1_test::generate_dataset<T>(64, "random", 12345);
    const auto c = step1_test::generate_dataset<T>(64, "random", 12346);
    expect(a == b, "same seed must reproduce inputs");
    expect(a != c, "these distinct seeds must produce different fixtures");
    expect(step1_test::generate_dataset<T>(5,"impulse",0) == std::vector<T>({0,0,13,0,0}),
           "impulse fixture");
    expect(step1_test::generate_dataset<T>(4,"max",0) == std::vector<T>(4,m), "max fixture");
    expect(step1_test::generate_dataset<T>(0,"zeros",0).empty(), "empty fixture");

    // A longer known-answer test exercises narrow-integer wrapping and tails.
    auto ones = step1_test::generate_dataset<T>(1027, "ones", 0);
    std::vector<T> result(ones.size()), sums(step1_test::tile_count(ones.size(), 65));
    step1_test::cpu_block_scan(ones.data(),result.data(),sums.data(),ones.size(),65);
    for (std::size_t i = 0; i < result.size(); ++i) {
        expect(result[i] == static_cast<T>(i % 65 + 1), "ones reference value");
    }
    expect(sums.back() == T{52}, "ones tail aggregate");

    std::ostringstream errors;
    expect(step1_test::compare_exact(a.data(),b.data(),a.size(),"same",errors), "exact match");
    auto changed = a;
    changed[3] = static_cast<T>(changed[3] ^ T{1});
    expect(!step1_test::compare_exact(changed.data(),a.data(),a.size(),"changed",errors), "mismatch detection");
    expect(errors.str().find("index=3") != std::string::npos, "first mismatch index");
}
} // namespace

int main() {
    try {
        expect(step1_test::tile_count(0,512) == 0, "empty tile count");
        expect(step1_test::tile_count(513,512) == 2, "tail tile count");
        expect(step1_test::tile_count(std::numeric_limits<std::size_t>::max(),2) ==
               std::numeric_limits<std::size_t>::max()/2 + 1, "overflow-safe tile count");
        check_type<uint8_t>();
        check_type<uint16_t>();
        check_type<uint32_t>();
        check_type<uint64_t>();
        std::cout << "PASS: " << checks << " CPU reference/data/validation checks (u8/u16/u32/u64)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
