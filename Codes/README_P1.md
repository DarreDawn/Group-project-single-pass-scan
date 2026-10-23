## P1 integration guide for P2 and P3

This project provides P1's block-local inclusive scan, warp-prefetch look-back optimization (Opt1), CPU validation utilities, and module tests. It exposes reusable device functions for P2/P3 integration. Dynamic tile assignment, production descriptor publication, the complete scan framework, and end-to-end performance evaluation remain integration work. See `TEST_STATUS.md` for verified results and remaining validation.

### Files and responsibilities

| File | Purpose |
|---|---|
| `include/block_scan.cuh` | Tile-local inclusive scan and final output storage with a supplied block prefix. |
| `include/warp_prefetch.cuh` | Shared flag definitions and cooperative loading of predecessor descriptors. |
| `include/lookback_consume.cuh` | Sequential consumption of one prefetched window. |
| `include/warp_prefetch_lookback.cuh` | Complete Opt1 look-back loop using warp prefetch and lane-0 consumption. |
| `include/validation.hpp` | CPU reference, input generation, and result-validation utilities. |
| `tests/` | Test programs; these are not dependencies of the production kernel. |

Use the existing `lookback::Flag` definition consistently across modules. Do not introduce a conflicting X/A/P encoding.

### Calling contracts

- `step1::scan_local_tile<T, B, Q>` must be called by every thread in a one-dimensional block of exactly B threads. It returns the current tile's aggregate to every thread and leaves tile-local inclusive results in each thread's `items[Q]`.
- Keep `items` alive throughout look-back. The local scan does not publish descriptors or write final output.
- `lookback::warp_prefetch_lookback<T>` must be called collectively by all 32 lanes of the first warp, with identical arguments and shared scratch storage.
- The look-back result is an exclusive tile prefix: the sum of all elements in preceding tiles. It excludes the current tile's aggregate.
- Check `result.success`. A failed result must not be treated as a valid zero prefix.
- Broadcast the prefix through shared memory and a block-wide barrier before calling `step1::store_tile_with_prefix<T, B, Q>`.
- Every thread in the block must call the store function with the same prefix.
- The optional `retry_signal` argument is test instrumentation. Production callers should omit it.

Input and output pointers refer to the entire arrays, not pointers already offset to a tile. Use the same logical `tile_id` throughout local scan, descriptor access, look-back, and output storage.

Supported values are unsigned 8-, 16-, 32-, and 64-bit integers, using modular addition. Input/output overlap is outside the current tested contract.

### Integration sequence

1. P2 obtains a dynamic logical tile ID and broadcasts it to the block.
2. All threads perform the local scan.
3. Publish the current tile's aggregate before entering look-back.
4. Obtain the exclusive tile prefix using the selected baseline or optimized look-back implementation.
5. Publish the current tile's inclusive prefix: exclusive tile prefix plus tile aggregate.
6. Broadcast the exclusive tile prefix to the block and store the final element-wise inclusive results.

The first tile has an exclusive prefix of zero. Its inclusive tile prefix equals its aggregate.

The standalone Step1 kernel deliberately uses a zero prefix for each tile. Keep it available for isolated testing; the full scan needs its own integration kernel.

### Descriptor publication contract

- Initialize all flags to X before the scan starts, with initialization ordered before the scan kernel.
- Reset the dynamic tile counter before each independent scan.
- Write `As[tile_id]` before publishing A.
- Write `Ps[tile_id]` before publishing P.
- Concurrent flag access must use compatible device-scope atomic operations: release publication and acquire reading.
- Published As and Ps values must remain unchanged until the scan finishes. Publishing P must not overwrite As.
- `volatile` alone does not implement this synchronization.
- A required X must be refreshed from global memory. Repeatedly checking only a cached shared-memory X cannot observe publication.
- P2/P3 must ensure that predecessor producers can make progress.

### Validation status and limits

The reported GPU runs passed:

- Step1 correctness tests and its memory, race, and synchronization checks.
- Static Opt1 prefetch and complete look-back tests.
- Memory, race, and synchronization checks for the static complete look-back test.
- 192 X-to-A/P publication cases.
- 24 forced internal-Retry cases, checking all 768 lane results.

The forced-Retry test checks that a previously accumulated window is preserved and that a partially consumed window is not counted twice.

These results do not establish correctness of the future full scan. A-to-P transitions under concurrent reads, inter-block scheduling, complete scan integration, and performance comparisons still require validation.

Sanitizer results for the static tests must not be reported as sanitizer coverage of the newer publication and forced-Retry tests.


## File dependencies and test entry points

### Production header dependencies

In the following table, “includes” means a direct C++ header dependency.

| File | Includes project header | Responsibility |
|---|---|---|
| `include/block_scan.cuh` | None | Local inclusive scan and final output storage with a supplied prefix |
| `include/warp_prefetch.cuh` | None | Flag definitions, predecessor indexing, acquire flag reads, and warp prefetch |
| `include/lookback_consume.cuh` | `warp_prefetch.cuh` | Sequential consumption of one prefetched window |
| `include/warp_prefetch_lookback.cuh` | `lookback_consume.cuh` | Complete Opt1 loop: prefetch, consume, retry, and move between windows |
| `include/validation.hpp` | None | CPU references, dataset generation, and result comparison |

“None” refers only to project-local headers. CUDA and standard-library headers are still required where used.

Including `warp_prefetch_lookback.cuh` also brings in `lookback_consume.cuh` and `warp_prefetch.cuh` through their include chain.

`block_scan.cuh` and `warp_prefetch_lookback.cuh` are separate modules. The future full kernel connects them; neither module automatically invokes the other.

### Functional integration order

1. P2 assigns and broadcasts a logical tile ID.
2. `scan_local_tile()` computes the tile aggregate and retains local inclusive results in each thread's `items`.
3. The publication code makes the aggregate available to other tiles.
4. The selected look-back implementation computes the exclusive tile prefix. For Opt1, use `warp_prefetch_lookback()`.
5. The publication code publishes the current tile's inclusive prefix.
6. The caller broadcasts the exclusive tile prefix to the entire block.
7. `store_tile_with_prefix()` adds that prefix and writes final output.

The local scan and store functions require participation by the entire block. The Opt1 helper requires participation by all 32 lanes of the first warp.

This sequence describes integration responsibilities, not an already implemented full kernel.

### Test dependencies

| Test source | Primary module under test | Execution |
|---|---|---|
| `tests/test_cpu.cpp` | `validation.hpp` | CPU |
| `tests/test_step1.cu` | `block_scan.cuh`, with CPU validation utilities | CPU host code and GPU kernels |
| `tests/test_prefetch_index.cu` | Indexing helpers in `warp_prefetch.cuh` | CPU checks, compiled with nvcc |
| `tests/test_prefetch_compile.cu` | `warp_prefetch.cuh` | Compile-only probe |
| `tests/test_prefetch_window.cu` | `warp_prefetch.cuh` | GPU |
| `tests/test_lookback_consume.cu` | `lookback_consume.cuh` | CPU checks, compiled with nvcc |
| `tests/test_warp_prefetch_lookback.cu` | `warp_prefetch_lookback.cuh` | GPU, static descriptors |
| `tests/test_lookback_publication.cu` | Opt1 helper and publication protocol | GPU, concurrent publisher |
| `tests/test_lookback_retry.cu` | Opt1 internal Retry path | GPU, controlled publication |

Production code must not include files from `tests/`. Test kernels and their simulated publishers are validation fixtures, not the production full-kernel implementation.

### Build and runner dependencies

| Entry point | Dependencies and behavior |
|---|---|
| `run_p1_tests.sh` | P1-wide entry point; builds Opt1 tests directly with nvcc and calls `run_tests.sh` for Step1 |
| `run_tests.sh` | Step1 entry point; invokes Makefile targets and Step1 test executables |
| `Makefile` | Builds `test_cpu` and `test_step1`; provides Step1 test and sanitizer targets |

Keep both scripts and the Makefile. The unified runner still depends on the Step1 runner.

Use `bash run_p1_tests.sh check` for the complete P1 correctness suite.

Use `bash run_p1_tests.sh sanitize` for the configured P1 sanitizer suite.

Use `bash run_tests.sh quick` when only a quick Step1 check is needed.

The current Makefile does not build the Opt1 tests; that compilation is handled by `run_p1_tests.sh`.

### Generated files and GitHub handoff

Commit:

- The `include/` and `tests/` source directories.
- `Makefile`, `run_tests.sh`, and `run_p1_tests.sh`.
- `README_P1.md`, `TEST_STATUS.md`, and `.gitignore`.

Generated files are not required to build the project:

| Generated item | Created by |
|---|---|
| `test_cpu`, `test_step1` | Makefile |
| `build/p1-tests/` | Unified P1 runner |
| `logs/` | Test runners |
| Earlier `/tmp/test_*` executables | Manual compilation commands |

Keep logs locally as validation evidence; selected logs can be shared separately if needed.

P2/P3 do not need to study every test implementation before integration. They should understand the header contracts, retain the tests, and run regression checks after changes. Use `TEST_STATUS.md` to distinguish verified module behavior from integration work that remains.