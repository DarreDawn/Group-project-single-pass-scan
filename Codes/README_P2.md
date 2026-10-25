# P2: Dynamic Block Index and Warp-Reduction Look-back

P2 connects P1's block-local scan to a complete inclusive scan. It provides
dynamic logical tile assignment, a serial look-back reference, Opt2 warp
reduction, the integrated `full_kernel`, and a host driver for correctness and
performance measurements.

The implementation supports unsigned 8-, 16-, 32-, and 64-bit values with
modular addition, matching P1.

---

## 1. Files

```
include/
  full_scan.cuh                   # Integrated P1 + P2 full scan kernel
  warp_reduction_lookback.cuh     # Descriptors, dynamic counter, serial reference, Opt2
src/
  main.cu                         # Host driver, validation, and benchmark
tests/
  test_p2.cu                      # Full-kernel correctness tests
run_p2_tests.sh                   # P2 check and sanitizer runner
```

The P2 CUDA code is in namespace `p2`. It reuses P1's
`step1::scan_local_tile`, `step1::store_tile_with_prefix`,
`step1::BlockScanScratch`, and `lookback::Flag`.

---

## 2. Scan API

### 2.1 Descriptors

```cpp
template<class T>
struct p2::Descriptors {
    lookback::Flag* flags;
    T* aggregates;
    T* prefixes;
    unsigned* next_tile;
    unsigned* error;
};
```

Allocate the three descriptor arrays with one entry per logical tile. The
counter and error word each have one entry. Before each independent scan,
initialize `flags` to X, `next_tile` to zero, and `error` to zero. X is encoded
as zero. `aggregates` and `prefixes` are written by the kernel before use.

### 2.2 Full kernel

```cpp
template<class T, int B, int Q, bool UseWarpReduction>
__global__ void p2::full_kernel(
    const T* input,
    T* output,
    uint64_t n,
    uint64_t num_tiles,
    p2::Descriptors<T> descriptors);
```

Launch a one-dimensional grid with one CTA per logical tile and exactly B
threads per CTA. Dynamic IDs come from an atomic fetch-add on `next_tile`;
`blockIdx.x` is not used as the logical tile ID. The `UseWarpReduction`
template argument selects the serial reference (`false`) or Opt2 (`true`).

The kernel performs the following steps:

1. Assigns a dynamic logical tile ID and broadcasts it within the CTA.
2. Runs P1's block-local inclusive scan and obtains the tile aggregate.
3. Publishes the aggregate with a device-scope release store of A.
4. Computes the exclusive tile prefix with serial or warp-reduction look-back.
5. Publishes the inclusive tile prefix with a device-scope release store of P.
6. Adds the exclusive prefix to P1's local results and stores final output.

Readers acquire-load descriptor flags before reading the associated
aggregate or prefix. Payloads are not modified after their status is
published. `error` is set if the look-back sees an invalid descriptor state;
the host driver rejects scans that report an error.

### 2.3 Look-back strategies

The serial reference assigns one thread to walk predecessor descriptors,
starting with the nearest tile. It accumulates A aggregates until it finds a P
inclusive prefix, or reaches tile zero.

Opt2, `warp_reduction_lookback`, uses all 32 lanes of the first warp. Lanes
load up to 32 predecessors in nearest-to-farthest order. A warp reduction
accumulates windows containing only A entries. The nearest P terminates the
walk; an X requires the same window to be reloaded until it becomes available.
Contributions farther than the nearest P are not included.

The counter-based assignment is important for progress: blocks receive
logical IDs as they begin execution, so a waiting block's predecessor IDs
have already been assigned to CTAs that can run or are running. The full
kernel launches one CTA per tile; changing the assignment scheme or adding
other spin-wait dependencies requires rechecking the forward-progress
assumptions.

---

## 3. Host driver and benchmark

`src/main.cu` allocates and initializes device buffers, generates deterministic
input, computes a CPU reference, launches both P2 strategies, and validates
their outputs. It also checks CUB's `DeviceScan::InclusiveSum` against the
same reference.

Build and run from `Codes/`:

```sh
make p2
./build/p2-tests/p2_scan 1048583 u32 10
```

Arguments are `n`, optional type (`u8`, `u16`, `u32`, or `u64`), and optional
iteration count. The default type is `u32`; the default iteration count is
10. Empty input is reported without launching GPU kernels.

CUDA events measure the average time for repeated launches. Full-scan timings
include flag, counter, and error initialization on the scan stream. They
exclude host allocations, input generation, CPU reference calculation, and
result validation. The driver reports:

- Serial full scan and Opt2 full scan.
- CUB `DeviceScan::InclusiveSum`.
- Device-to-device memcpy bandwidth.

Scan effective bandwidth counts one input read and one output write. The
memcpy bandwidth counts the copied payload only, so it is a reference
measurement rather than an identical workload. No performance result is
claimed by the correctness test; run the driver on the target GPU to collect
measurements.

---

## 4. Testing

Run the P2 suite from `Codes/` on a CUDA machine:

```sh
bash run_p2_tests.sh check
bash run_p2_tests.sh sanitize
```

`check` builds the test and host driver, then runs the full-kernel test.
`sanitize` runs the P2 test under Compute Sanitizer's memcheck, racecheck, and
synccheck tools.

The test covers both look-back strategies, all four supported value types,
and these input length/pattern pairs:

| Length | Pattern |
|---:|---|
| 0 | zeros |
| 1 | ones |
| 511 | random |
| 512 | impulse |
| 513 | maximum unsigned value |
| 4,099 | random |
| 66,559 | ones |

These cases exercise empty input, tile boundaries and tails, multi-tile
look-back, and a 130-tile scan. Each nonempty case is checked against the CPU
inclusive-scan reference; the test also verifies the number of dynamically
assigned tiles and the device protocol error flag.

---

## 5. Current status

- **P1:** The supplied Hendrix test output reports `PASS: P1 check suite completed`.
  It records 26,082 GPU kernel checks and 282 empty-input host checks passed,
  with one shared-memory configuration explicitly skipped.
- **P2:** The supplied Hendrix CUDA 12.8 / A100 test output reports
  `PASS: P2 dynamic assignment and scan tests` and
  `PASS: P2 check suite completed`.
- The P2 result verifies the tested full-scan correctness cases for the
  serial and warp-reduction paths. It is not a sanitizer result.
- Performance measurements against CUB and device-to-device memcpy are
  available through `src/main.cu`, but no benchmark measurements are recorded
  here.
