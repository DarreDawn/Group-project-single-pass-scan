# P1 Test Status

Updated: 2026-10-23 (Europe/Copenhagen).

This document records observed test results for the P1 modules. It does not certify the future integrated P2/P3 single-pass scan.

## Test environment

- GPU: NVIDIA A100-PCIE-40GB.
- Compute capability: 8.0.
- Compilation target: `sm_80`.
- CUDA Toolkit: 12.8, nvcc V12.8.61.
- Language standard: C++17.
- Value types: unsigned 8-, 16-, 32-, and 64-bit integers.
- Arithmetic: unsigned modular addition.

## Latest unified correctness run

Command: `bash run_p1_tests.sh check`

Result: **PASS — P1 check suite completed.**

The run compiled the Opt1 test programs, compiled the compile-only probe, executed the Step1 full suite, and executed the Opt1 correctness tests.

| Test | Observed result | Scope |
|---|---|---|
| CPU reference and validation utilities | 4,179 checks passed | CPU reference, data generation, and validation utilities |
| Step1 full suite | 26,082 GPU kernel checks and 282 empty-input host checks passed; 1 configuration skipped | Local scan and prefix-application interface |
| Prefetch indexing | Passed | Host-side predecessor/window indexing |
| Prefetch compilation probe | Compiled successfully | Compilation only; no runtime execution |
| Window consumption | 48 CPU checks passed | Single-window A/P/X decisions |
| GPU prefetch | 256 windows and 8,192 lane snapshots passed | Static descriptor loading and shared-window visibility |
| Complete Opt1 look-back with static descriptors | 320 queries and 10,240 lane results passed | Multi-window accumulation and termination |
| Concurrent publication | 192 cases, 192 initial Retry checks, and 6,144 lane results passed | One-block X-to-A/P publication |
| Forced internal Retry | 24 cases and 768 lane results passed | Actual Retry branch inside the complete look-back helper |

Counts describe different kinds of checks and should not be combined into one total.

### Skipped Step1 configuration

The configuration `u64, B=1024, Q=8` requires 65,792 bytes of shared memory and exceeded the default resource limit used by the test.

It was explicitly skipped, not counted as a passing configuration. The test does not request an increased opt-in shared-memory limit.

### Forced-Retry coverage

The forced-Retry test delays publication until the complete look-back helper signals that its internal Retry branch has executed.

The reader first accumulates an available window, then encounters X in a later window. The final result checks that:

- Previously accumulated contributions are preserved.
- Tentative contributions from the window containing X are not added twice.
- Global state is refreshed after publication.
- All 32 participating lanes return the expected prefix.

The optional `retry_signal` argument is test instrumentation. Normal integration calls omit it.

## Sanitizer results already observed

Earlier runs reported:

| Tested module/configuration | memcheck | racecheck | synccheck |
|---|---|---|---|
| Step1 sanitizer suite | 0 errors | 0 hazards / 0 warnings | 0 errors |
| Static complete Opt1 look-back test | 0 errors | 0 hazards / 0 warnings | 0 errors |
prefetch window、concurrent publication、forced internal Retry all pass

These are records of the earlier runs. They are not evidence that every newer test or subsequent source revision has passed sanitizer checks.

The unified command `bash run_p1_tests.sh sanitize` completed successfully. The configured Step1 sanitizer cases and all four Opt1 GPU test programs—prefetch window, static complete look-back, concurrent publication, and forced internal Retry—passed memcheck, racecheck, and synccheck. The reported summaries showed zero errors and zero race hazards or warnings.

These results apply to the tested configurations and execution paths. They do not establish correctness of the future integrated P2/P3 full scan.

## Limits and remaining validation

The following are not established by the current results:

- Concurrent A-to-P transitions.
- Dynamic tile assignment and inter-block forward progress.
- Integrated Step1, descriptor publication, look-back, and final output in a complete scan kernel.
- P2's warp-reduction optimization or P3's additional optimization.
- End-to-end validation of the integrated implementation.
- Performance improvements, bandwidth measurements, or comparisons with CUB and memcpy.

Step1 tests that supply CPU-computed prefixes validate the prefix-application interface; they do not execute GPU look-back.

The Opt1 publication tests use two warps within one block. They do not establish deadlock freedom across independently scheduled blocks.

## Reproduction

Run from the project root with the CUDA environment loaded:

- Correctness suite: `bash run_p1_tests.sh check`
- Sanitizer suite: `bash run_p1_tests.sh sanitize`

The unified runner stores logs under `logs/` and rebuilds Opt1 executables under `build/p1-tests/`.

When changing the Step1 compiler flags or target architecture, run `make clean` before rebuilding. Record the tested Git commit when sharing final results.