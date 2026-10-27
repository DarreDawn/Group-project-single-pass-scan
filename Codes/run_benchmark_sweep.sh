#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_benchmark_sweep.sh
#
# Sweep the integrated single-pass scan benchmark (p2_scan) over the four
# look-back strategies, several array lengths, all four unsigned integer types,
# and the B x Q tile-granularity ablation, then emit a CSV the report can turn
# into tables/plots.
#
# Each strategy line is validated against the CPU reference before its timing
# is reported, so a run that disagrees with the reference aborts that size
# instead of recording a bogus number.
#
# CSV columns:
#   type,n,B,Q,tile,baseline_GBs,warp_prefetch_GBs,warp_reduce_GBs,
#   fencefree_GBs,cub_GBs,memcpy_GBs
#   - the first four strategy columns are per (B,Q) configuration;
#   - cub_GBs / memcpy_GBs are measured once per (n, type) and are identical
#     for every (B,Q) row of that input.
#   - fencefree is paper §4.4's single-word descriptor strategy, replacing the
#     earlier P-flag variant (which never beat baseline in a full-space sweep).
#
# Usage:
#   bash run_benchmark_sweep.sh [output.csv]
# Env overrides:
#   TYPES="u8 u16 u32 u64"  ITERATIONS=20  SIZES="1048576 ..."  CUDA_ARCH=sm_80
# ---------------------------------------------------------------------------
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_ARCH="${CUDA_ARCH:-sm_80}"
types="${TYPES:-u8 u16 u32 u64}"
# Multiple array lengths from 2^16 to 2^28 (dense enough to expose where each
# look-back optimization wins: small n / few blocks vs long chains).
sizes="${SIZES:-65536 262144 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456}"
iterations="${ITERATIONS:-20}"
out="${1:-bench_sweep.csv}"

for tool in nvcc; do
    command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 1; }
done

echo "BUILD: p2_scan (CUDA_ARCH=$CUDA_ARCH)"
make p2

printf 'type,n,B,Q,tile,baseline_GBs,warp_prefetch_GBs,warp_reduce_GBs,fencefree_GBs,cub_GBs,memcpy_GBs\n' > "$out"

# Extract the numeric value immediately before "GB/s" on a matching line.
gbs() { # file keyword -> number
    local file="$1" key="$2"
    grep -E "^  .*${key}.*GB/s" "$file" \
        | grep -oE '[0-9]+([.][0-9]+)? +GB/s' \
        | sed -E 's/ +GB\/s//' \
        | tail -1
}

# Turn one p2_scan log into one CSV row per "Config B=.. Q=.." block.
# t / n / cub / mem are injected; B, Q, tile and the four strategy GB/s come
# from the log itself. Rows are emitted in log order.
emit_rows() { # t n cub mem logfile
    local t="$1" n="$2" cub="$3" mem="$4" log="$5"
    awk -v t="$t" -v n="$n" -v cub="$cub" -v mem="$mem" '
        /^Config / {
            if (hascfg) emit();
            split($2, b, "="); cfgB = b[2];
            split($3, q, "="); cfgQ = q[2];
            split($4, tl, "="); cfgTile = tl[2];
            hascfg = 1; bl = pf = rd = ff = "";
        }
        /effective .*GB\/s/ {
            val = $0;
            sub(/^.*effective /, "", val);
            sub(/ GB\/s.*$/, "", val);
            if (index($0, "baseline")) bl = val;
            else if (index($0, "prefetch")) pf = val;
            else if (index($0, "warp-reduce")) rd = val;
            else if (index($0, "fence-free")) ff = val;
        }
        function emit() {
            print t "," n "," cfgB "," cfgQ "," cfgTile "," bl "," pf "," rd "," ff "," cub "," mem;
        }
        END { if (hascfg) emit(); }
    ' "$log"
}

for t in $types; do
    for n in $sizes; do
        log="logs/sweep-${t}-${n}.log"
        mkdir -p logs
        if ! ./build/p2-tests/p2_scan "$n" "$t" "$iterations" > "$log" 2>&1; then
            echo "  [FAIL] type=$t n=$n (see $log)" >&2
            continue
        fi
        cub=$(gbs "$log" 'CUB')
        mem=$(gbs "$log" 'memcpy')
        emit_rows "$t" "$n" "$cub" "$mem" "$log" >> "$out"
        echo "  recorded type=$t n=$n (6 tile configs; cub=$cub GB/s)"
    done
done

echo "Wrote $out"
