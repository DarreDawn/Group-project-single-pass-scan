#!/usr/bin/env bash
# Usage: bash run_p1_tests.sh [check|sanitize]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-check}"
case "$mode" in
    check|sanitize) ;;
    *) echo "Usage: bash run_p1_tests.sh [check|sanitize]" >&2; exit 2 ;;
esac

# Override these variables for a different GPU or slower test tools.
export CUDA_ARCH="${CUDA_ARCH:-sm_80}"
test_timeout="${TEST_TIMEOUT:-60s}"
sanitizer_timeout="${SANITIZER_TIMEOUT:-300s}"

cpu_tests=(
    test_prefetch_index
    test_lookback_consume
)

gpu_tests=(
    test_prefetch_window
    test_warp_prefetch_lookback
    test_lookback_publication
    test_lookback_retry
)

# Check tools and source files before starting.
for tool in nvcc make timeout tee; do
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

for source in "${cpu_tests[@]}" "${gpu_tests[@]}" test_prefetch_compile; do
    [[ -f "tests/${source}.cu" ]] || {
        echo "Missing source: tests/${source}.cu" >&2
        exit 1
    }
done

[[ -f run_tests.sh ]] || {
    echo "Missing Step1 runner: run_tests.sh" >&2
    exit 1
}

mkdir -p build/p1-tests logs
log="logs/p1-$(date +%Y%m%d-%H%M%S)-${mode}-$$.log"
echo "Log: $log"

{
    echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "mode=$mode CUDA_ARCH=$CUDA_ARCH"
    nvcc --version

    # Rebuild every Opt1 test to include the latest header changes.
    for test in "${cpu_tests[@]}" "${gpu_tests[@]}"; do
        echo "BUILD: $test"
        nvcc -std=c++17 -O2 -lineinfo "-arch=$CUDA_ARCH" -Iinclude "tests/${test}.cu" -o "build/p1-tests/$test"
    done

    # This probe has no main function; compile without linking or running.
    echo "COMPILE ONLY: test_prefetch_compile"
    nvcc -std=c++17 -O2 -lineinfo "-arch=$CUDA_ARCH" -Iinclude -c tests/test_prefetch_compile.cu -o build/p1-tests/test_prefetch_compile.o

    # Reuse the existing Step1 test runner.
    if [[ "$mode" == check ]]; then
        bash run_tests.sh full
    else
        bash run_tests.sh sanitize
    fi

    # These tests execute their checks on the CPU.
    for test in "${cpu_tests[@]}"; do
        echo "RUN: $test"
        timeout "$test_timeout" "build/p1-tests/$test"
    done

    # GPU correctness checks or sanitizer checks.
    for test in "${gpu_tests[@]}"; do
        if [[ "$mode" == check ]]; then
            echo "RUN: $test"
            timeout "$test_timeout" "build/p1-tests/$test"
        else
            for checker in memcheck racecheck synccheck; do
                echo "SANITIZE: $test ($checker)"
                timeout "$sanitizer_timeout" compute-sanitizer --error-exitcode 1 --tool "$checker" "build/p1-tests/$test"
            done
        fi
    done

    echo "PASS: P1 $mode suite completed"
    echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$log"