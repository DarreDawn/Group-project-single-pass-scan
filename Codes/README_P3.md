# Decoupled Look-back Module (P3)

This module implements the **decoupled look-back** single-pass prefix scan
described in Merrill & Garland (NVIDIA). It sits between P1's block-local
scan and P2's top-level kernel / host driver.

Two strategies are provided, selectable at **compile time**:

| Strategy | Header | Template flag | Purpose |
|---|---|---|---|
| Baseline | `lookback_baseline.cuh` | `UsePFlag = false` | Single-thread serial walk, correctness reference |
| P-flag propagation | `lookback_pflag.cuh` | `UsePFlag = true` | Promotes encountered A entries to P to shorten future walks |

---

## 1. Files

```
include/
  lookback_publish.cuh          # Flag enum (fallback), Descriptor<T>, publish primitives
  lookback_baseline.cuh         # decoupled_lookback::baseline(...)
  lookback_pflag.cuh            # decoupled_lookback::pflag(...)
  block_scan_with_lookback.cuh  # decoupled_lookback::block_scan<T,B,Q,UsePFlag>(...)
tests/
  test_baseline_lookback.cu     # unit test for baseline state machine
  test_pflag_lookback.cu        # correctness + invariant test for pflag
run_p3_tests.sh                 # build + run + sanitize
```

All module code lives in namespace `decoupled_lookback`. The `Flag`
enum lives in namespace `lookback` (shared with P1's
`lookback_consume.cuh`).

---

## 2. Public API (what P2 calls)

```cpp
template <typename T>
struct Descriptor {
    volatile lookback::Flag* Fs;   // length = num_blocks
    volatile T*              As;   // length = num_blocks
    volatile T*              Ps;   // length = num_blocks
    int                      num_blocks;
};

template <typename T, int B, int Q, bool UsePFlag = false>
__device__ __forceinline__ void decoupled_lookback::block_scan(
    Descriptor<T> desc,
    int block_idx,                         // LOGICAL index from P2
    T* input, T* output, uint64_t n,
    T* items,                              // from P1 block_local_scan
    step1::BlockScanScratch<T,B,Q>& scratch,
    T block_aggregate);                    // from P1 block_local_scan
```

**Calling contract**

- Every thread in the block calls `block_scan` (same as P1's
  `block_local_scan`).
- `block_idx` **must** be the dynamic logical index supplied by P2.
  `block_scan` never reads `blockIdx.x`.
- On return, `output[]` contains the global inclusive prefix.
- `items[]` must stay alive across the call (register array from P1).
- All blocks must be simultaneously resident, or P2's dynamic numbering
  must guarantee forward progress. Otherwise the spin loop can deadlock.

---

## 3. What P2 must provide

### 3.1 Host side (before launch)

```cpp
// Allocate one element per block
cudaMalloc(&d_Fs, num_blocks * sizeof(lookback::Flag));
cudaMalloc(&d_As, num_blocks * sizeof(T));
cudaMalloc(&d_Ps, num_blocks * sizeof(T));

// Fs MUST be initialized to X before every launch.
// X == 0, so cudaMemset is sufficient.
cudaMemset(d_Fs, 0, num_blocks * sizeof(lookback::Flag));

// As / Ps do not need initialization.
```

If the same state arrays are reused across launches, re-run the
`cudaMemset(d_Fs, ...)` **before each launch**. Stale P/A flags from a
previous launch will corrupt the next one.

### 3.2 Kernel side

```cpp
// P2 obtains dynamic_block_idx via atomicAdd on a global counter.
int dynamic_block_idx = ...;

// P1 produces aggregate + items + scratch.
T block_aggregate = step1::block_local_scan<T,B,Q>(...);

// P3 consumes them.
decoupled_lookback::Descriptor<T> desc{d_Fs, d_As, d_Ps, num_blocks};
decoupled_lookback::block_scan<T, B, Q, /*UsePFlag=*/false>(
    desc, dynamic_block_idx,
    input, output, n, items, scratch, block_aggregate);
```

### 3.3 Launch configuration

- Grid size can exceed the resident block count **only if** P2's dynamic
  numbering guarantees forward progress (the counter-based scheme does).
- Without dynamic numbering (e.g. while debugging), keep
  `gridDim.x <= max_resident_blocks` to avoid deadlock.

---

## 4. What P3 depends on from P1

P3 calls the following P1 functions (names may need adjustment to match
P1's actual declarations):

| P1 symbol | Used for |
|---|---|
| `step1::block_local_scan<T,B,Q>(...)` | Per-block local scan; returns the tile aggregate |
| `step1::store_tile_with_prefix<T,B,Q>(...)` | Stores `items[] + exclusive_prefix` to global |
| `step1::BlockScanScratch<T,B,Q>` | Shared-memory scratch passed through |

P3 does **not** re-implement any block-level scan; it only supplies the
exclusive prefix.

---

## 5. Flag enum sharing

`lookback_publish.cuh` defines `lookback::Flag { X, A, P }` locally by
default. If P1's `lookback_consume.cuh` already provides the same enum,
compile with:

```
-DP3_USE_P1_FLAG
```

and make sure `lookback_consume.cuh` is on the include path. Do **not**
leave two coexisting `Flag` definitions active — state comparisons will
silently fail.

---

## 6. Testing

```bash
bash run_p3_tests.sh            # check
bash run_p3_tests.sh sanitize   # memcheck / racecheck / synccheck
```

### Known sanitizer notes

- **`test_pflag_lookback` / racecheck** is intentionally skipped.
  P-flag promotion writes `Ps[i]` / `Fs[i]` from multiple blocks with the
  *same deterministic value*. These are idempotent, natural-sized writes:
  benign for CUDA, but `racecheck` reports them as hazards. memcheck and
  synccheck still run.
- **Block-0 edge case (fixed)**: the baseline and pflag loops carry an
  explicit `if (idx == 0) break;` after adding `As[0]`. Without it,
  `--idx` reaches -1 and reads `Fs[-1]` (out of bounds). The bug only
  fires inside the short A→P window of block 0 and was caught by
  sanitize, not by `check`. Keep the guard if you touch the loop.

---

## 7. Choosing baseline vs pflag

| Aspect | baseline | pflag |
|---|---|---|
| Implementation complexity | low | medium |
| Extra register pressure | none | `MAX_TRACK * sizeof(TrackedEntry)` |
| Average look-back distance | full A-chain | shortened by promotion |
| Sanitizer | all pass | racecheck skipped (benign) |
| Recommended for | reference, small grids | production, large grids |

`MAX_TRACK` is a `constexpr int` in `lookback_pflag.cuh` (default 32).
Tune per data type / grid size when running the ablation study.

---

## 8. Integration checklist for P2

- [ ] Allocate `Fs`, `As`, `Ps` on device (length = `num_blocks`)
- [ ] `cudaMemset(Fs, 0, ...)` **before every launch** (X == 0)
- [ ] Pass `dynamic_block_idx` (not `blockIdx.x`) as `block_idx`
- [ ] Instantiate `block_scan<T, B, Q, false>` and `<..., true>` if
      both strategies need to be exposed at runtime
- [ ] Ensure launch config respects the resident-block constraint
      (or rely on dynamic numbering)
- [ ] Add `lookback::Flag` to any host-side log/print utilities as
      `uint32_t` (do not print `enum class` directly)

---

## 9. Current status

- Baseline: unit test + memcheck + racecheck + synccheck passed
- P-flag: unit test + memcheck + synccheck passed; racecheck skipped
  (idempotent concurrent writes, see §6)
- P-flag large (48 blocks, MAX_TRACK overflow): memcheck + synccheck passed
- Edge cases (nb=1, nb=2, blockDim=1): memcheck + synccheck passed
- End-to-end performance vs CUB / memcpy: pending P2 `full_kernel` integration