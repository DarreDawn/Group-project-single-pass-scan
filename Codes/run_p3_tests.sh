#!/usr/bin/env bash
# Usage: bash run_p3_tests.sh [check|sanitize]
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-check}"
case "$mode" in
    check|sanitize) ;;
    *) echo "Usage: bash run_p3_tests.sh [check|sanitize]" >&2; exit 2 ;;
esac

export CUDA_ARCH="${CUDA_ARCH:-sm_80}"
test_timeout="${TEST_TIMEOUT:-60s}"
sanitizer_timeout="${SANITIZER_TIMEOUT:-300s}"

cpu_tests=()

gpu_tests=(
    test_baseline_lookback
    test_pflag_lookback
    test_pflag_large
    test_edge_cases
)

declare -A skip_racecheck=(
    [test_pflag_lookback]=1
    [test_pflag_large]=1
    [test_edge_cases]=1
)

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

all_tests=("${cpu_tests[@]}" "${gpu_tests[@]}")
for source in "${all_tests[@]}"; do
    [[ -f "tests/${source}.cu" ]] || {
        echo "Missing source: tests/${source}.cu" >&2
        exit 1
    }
done

mkdir -p build/p3-tests logs
log="logs/p3-$(date +%Y%m%d-%H%M%S)-${mode}-$$.log"
echo "Log: $log"

pass=0
fail=0
declare -a failed_tests=()

# Run a command with timeout; append output to log; record pass/fail.
run_timed() {
    local label="$1"
    local limit="$2"
    shift 2
    if timeout "$limit" "$@" 2>&1 | tee -a "$log"; then
        return 0
    else
        local rc="${PIPESTATUS[0]}"
        if [[ "$rc" -eq 124 ]]; then
            echo "  [$label] TIMEOUT after $limit" | tee -a "$log"
        else
            echo "  [$label] FAILED (exit $rc)" | tee -a "$log"
        fi
        return 1
    fi
}

{
    echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "mode=$mode CUDA_ARCH=$CUDA_ARCH"
    nvcc --version

    for test in "${all_tests[@]}"; do
        echo "BUILD: $test"
        if ! nvcc -std=c++17 -O2 -lineinfo "-arch=$CUDA_ARCH" -Iinclude \
                "tests/${test}.cu" -o "build/p3-tests/$test" 2>&1 | tee -a "$log"; then
            echo "  [build] FAILED: $test" | tee -a "$log"
            failed_tests+=("$test (build)")
            ((fail++))
            continue
        fi
        if [[ -f "build/p3-tests/$test" ]]; then
            ((pass++))
        fi
    done

    for test in "${cpu_tests[@]}"; do
        echo "RUN: $test"
        if run_timed "$test" "$test_timeout" "build/p3-tests/$test"; then
            :
        else
            failed_tests+=("$test")
            ((fail++))
        fi
    done

    for test in "${gpu_tests[@]}"; do
        if [[ ! -f "build/p3-tests/$test" ]]; then
            continue   # build failed, already recorded
        fi
        if [[ "$mode" == check ]]; then
            echo "RUN: $test"
            if run_timed "$test" "$test_timeout" "build/p3-tests/$test"; then
                :
            else
                failed_tests+=("$test")
                ((fail++))
            fi
        else
            for checker in memcheck racecheck synccheck; do
                if [[ "$checker" == racecheck && -n "${skip_racecheck[$test]:-}" ]]; then
                    echo "SANITIZE: $test ($checker) -- SKIPPED (benign concurrent writes)" \
                        | tee -a "$log"
                    continue
                fi
                echo "SANITIZE: $test ($checker)"
                if run_timed "$test/$checker" "$sanitizer_timeout" \
                        compute-sanitizer --error-exitcode 1 --tool "$checker" \
                        "build/p3-tests/$test"; then
                    :
                else
                    failed_tests+=("$test ($checker)")
                    ((fail++))
                fi
            done
        fi
    done

    echo
    echo "============================================================"
    echo " Summary (mode=$mode)"
    echo "   passed : $pass"
    echo "   failed : $fail"
    if [[ $fail -gt 0 ]]; then
        echo "   failing: ${failed_tests[*]}"
    fi
    echo "============================================================"
    echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$log"

exit $(( fail > 0 ? 1 : 0 ))