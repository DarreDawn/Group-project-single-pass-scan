#include "lookback_consume.cuh"

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

unsigned checks = 0;

template<class T>
void expect(const char* type_name, const char* case_name,
            const lookback::PrefetchWindow<T>& window,
            lookback::WindowAction action, T contribution) {
    const auto result = lookback::consume_lookback_window(window);
    if (result.action != action || result.contribution != contribution) {
        std::ostringstream message;
        message << type_name << " / " << case_name
                << ": got action=" << static_cast<unsigned>(result.action)
                << " contribution=" << static_cast<uint64_t>(result.contribution)
                << "; expected action=" << static_cast<unsigned>(action)
                << " contribution=" << static_cast<uint64_t>(contribution);
        throw std::runtime_error(message.str());
    }
    ++checks;
}

template<class T>
void run_type(const char* name) {
    using lookback::Flag;
    using lookback::WindowAction;
    using Window = lookback::PrefetchWindow<T>;
    Window w{};

    expect(name, "empty", w, WindowAction::Done, T{0});

    w.valid_count = 3;
    w.flags[0] = w.flags[1] = w.flags[2] = Flag::A;
    w.aggregates[0] = T{5};
    w.aggregates[1] = T{7};
    w.aggregates[2] = T{2};
    expect(name, "all A", w, WindowAction::Continue, T{14});

    w.flags[2] = Flag::P;
    w.prefixes[2] = T{100};
    expect(name, "A A P", w, WindowAction::Done, T{112});

    w = Window{};
    w.valid_count = 3;
    w.flags[0] = Flag::P;
    w.prefixes[0] = T{70};
    expect(name, "P before older X", w, WindowAction::Done, T{70});

    w = Window{};
    w.valid_count = 3;
    w.flags[0] = Flag::A;
    w.aggregates[0] = T{5};
    w.flags[1] = Flag::P;
    w.prefixes[1] = T{100};
    expect(name, "A P X", w, WindowAction::Done, T{105});

    w = Window{};
    w.valid_count = 3;
    w.flags[1] = Flag::A;
    w.aggregates[1] = T{7};
    w.flags[2] = Flag::P;
    w.prefixes[2] = T{100};
    expect(name, "X at nearest entry", w, WindowAction::Retry, T{0});

    w = Window{};
    w.valid_count = 3;
    w.flags[0] = Flag::A;
    w.aggregates[0] = T{5};
    w.flags[2] = Flag::P;
    w.prefixes[2] = T{100};
    expect(name, "A X P discards tentative sum", w, WindowAction::Retry, T{0});

    w = Window{};
    w.valid_count = 2;
    w.flags[0] = w.flags[1] = Flag::A;
    w.aggregates[0] = T{5};
    w.aggregates[1] = T{7};
    w.flags[2] = Flag::P;
    w.prefixes[2] = T{200};
    expect(name, "ignore entries beyond valid_count", w, WindowAction::Continue, T{12});

    w = Window{};
    w.valid_count = 2;
    w.flags[0] = w.flags[1] = Flag::A;
    w.aggregates[0] = std::numeric_limits<T>::max();
    w.aggregates[1] = T{1};
    expect(name, "modular A addition", w, WindowAction::Continue, T{0});

    w.flags[1] = Flag::P;
    w.prefixes[1] = T{2};
    expect(name, "modular P addition", w, WindowAction::Done, T{1});

    w = Window{};
    w.valid_count = 33;
    expect(name, "invalid count", w, WindowAction::Invalid, T{0});

    w = Window{};
    w.valid_count = 1;
    w.flags[0] = static_cast<Flag>(99);
    expect(name, "invalid flag", w, WindowAction::Invalid, T{0});

    std::cout << "PASS " << name << ": 12 window-consumption cases\n";
}

} // namespace

int main() {
    try {
        run_type<uint8_t>("u8");
        run_type<uint16_t>("u16");
        run_type<uint32_t>("u32");
        run_type<uint64_t>("u64");
        std::cout << "PASS: " << checks << " CPU window-consumption checks\n"
                  << "Scope: one-window decisions only; no GPU kernel, "
                     "retry loop, or concurrent publisher was tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
