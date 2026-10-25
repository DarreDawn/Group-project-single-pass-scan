#!/usr/bin/env bash
# Usage: bash run_p2_tests.sh [check|sanitize]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-check}"
case "$mode" in
    check|sanitize) ;;
    *) echo "Usage: bash run_p2_tests.sh [check|sanitize]" >&2; exit 2 ;;
esac

export CUDA_ARCH="${CUDA_ARCH:-sm_80}"
test_timeout="${TEST_TIMEOUT:-120s}"
sanitizer_timeout="${SANITIZER_TIMEOUT:-600s}"

for tool in nvcc timeout tee; do
    command -v "$tool" >/dev/null || {
        echo "Missing tool: $tool" >&2
        exit 1
    }
done

if [[ "$mode" == sanitize ]]; then
    command -v compute-sanitizer >/dev/null || {
        echo "Missing tool: compute-sanitizer" >&2
        exit 1
    }
fi

mkdir -p build/p2-tests logs
log="logs/p2-$(date +%Y%m%d-%H%M%S)-${mode}-$$.log"
echo "Log: $log"

{
    echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "mode=$mode CUDA_ARCH=$CUDA_ARCH"
    nvcc --version

    echo "BUILD: test_p2"
    nvcc -std=c++17 -O2 -lineinfo "-arch=$CUDA_ARCH" -Iinclude \
        tests/test_p2.cu -o build/p2-tests/test_p2
    echo "BUILD: p2_scan"
    nvcc -std=c++17 -O3 -lineinfo "-arch=$CUDA_ARCH" -Iinclude \
        src/main.cu -o build/p2-tests/p2_scan

    if [[ "$mode" == check ]]; then
        echo "RUN: test_p2"
        timeout "$test_timeout" build/p2-tests/test_p2
    else
        for checker in memcheck racecheck synccheck; do
            echo "SANITIZE: test_p2 ($checker)"
            timeout "$sanitizer_timeout" compute-sanitizer \
                --error-exitcode 1 --tool "$checker" build/p2-tests/test_p2
        done
    fi

    echo "PASS: P2 $mode suite completed"
    echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$log"
