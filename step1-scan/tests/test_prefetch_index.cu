#include "warp_prefetch.cuh"

#include <cassert>
#include <iostream>

int main() {
    using namespace lookback;

    // Check window sizes.
    assert(window_size(0) == 0);
    assert(window_size(3) == 3);
    assert(window_size(32) == 32);
    assert(window_size(40) == 32);

    uint64_t index = 999;

    // Tile 0 has no predecessors.
    assert(!get_predecessor_index(0, 0, index));
    assert(index == 0);

    // A partial window containing tiles 2, 1, 0.
    assert(get_predecessor_index(3, 0, index) && index == 2);
    assert(get_predecessor_index(3, 1, index) && index == 1);
    assert(get_predecessor_index(3, 2, index) && index == 0);
    assert(!get_predecessor_index(3, 3, index));

    // A full window ending at tile 0.
    assert(get_predecessor_index(32, 31, index) && index == 0);

    // A full window containing tiles 39 through 8.
    assert(get_predecessor_index(40, 0, index) && index == 39);
    assert(get_predecessor_index(40, 31, index) && index == 8);

    // A warp window has only lanes 0 through 31.
    assert(!get_predecessor_index(40, 32, index));

    std::cout << "PASS: Opt1 window indexing checks\n";
    return 0;
}