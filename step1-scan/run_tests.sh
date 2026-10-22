#!/usr/bin/env bash
# Run from any directory. Usage: bash run_tests.sh [quick|full|sanitize] [seed] [repeat]
# Set CUDA_ARCH if your course requires an explicit architecture, e.g. sm_80.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

suite="${1:-quick}"
seed="${2:-1}"
repeat="${3:-1}"
case "$suite" in
  quick|full|sanitize) ;;
  *) echo "Usage: bash run_tests.sh [quick|full|sanitize] [seed] [repeat]" >&2; exit 2 ;;
esac

mkdir -p logs
log="logs/$(date +%Y%m%d-%H%M%S)-${suite}-$$.log"
{
  echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "suite=$suite seed=$seed repeat=$repeat CUDA_ARCH=${CUDA_ARCH:-compiler-default}"
  nvcc --version
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  make all
  ./test_cpu
  if [[ "$suite" == sanitize ]]; then
    echo "Sanitizer target uses its fixed small cases and default seed/repeat."
    make sanitize
  else
    ./test_step1 --suite "$suite" --seed "$seed" --repeat "$repeat"
  fi
  echo "Finished successfully: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} 2>&1 | tee "$log"
echo "Saved log: $log"
