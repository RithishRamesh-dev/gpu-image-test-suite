#!/usr/bin/env bash
# =============================================================================
# ROCm 7.2.0 GPU Droplet — End-to-End Validation Suite
# Main entry point
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/test_run.log"

# ── Configurable via environment ─────────────────────────────────────────────
ROCM_EXPECTED_VERSION="${ROCM_EXPECTED_VERSION:-7.2.0}"
BASELINE_VERSION="${BASELINE_VERSION:-7.0.2}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3-0324}"
TP_SIZE="${TP_SIZE:-8}"          # tensor parallel size (= GPU count)
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SKIP_VLLM="${SKIP_VLLM:-false}"
SKIP_RCCL="${SKIP_RCCL:-false}"
SKIP_STRESS="${SKIP_STRESS:-false}"
SKIP_MICROBENCH="${SKIP_MICROBENCH:-false}"
STRESS_DURATION="${STRESS_DURATION:-600}"       # seconds
VLLM_LONGEVITY_DURATION="${VLLM_LONGEVITY_DURATION:-1800}"  # seconds
# ── Micro-benchmark thresholds (module 15) ───────────────────────────────────
export MB_GEMM_FP16_MIN_TFLOPS="${MB_GEMM_FP16_MIN_TFLOPS:-100}"
export MB_GEMM_FP32_MIN_TFLOPS="${MB_GEMM_FP32_MIN_TFLOPS:-20}"
export MB_HBM_BW_MIN_GBPS="${MB_HBM_BW_MIN_GBPS:-800}"
export MB_PCIE_BW_MIN_GBPS="${MB_PCIE_BW_MIN_GBPS:-20}"
export MB_THERMAL_MAX_TEMP="${MB_THERMAL_MAX_TEMP:-90}"
export MB_GEMM_DURATION="${MB_GEMM_DURATION:-30}"
export MB_MEM_DURATION="${MB_MEM_DURATION:-60}"
export MB_PCIE_ITER="${MB_PCIE_ITER:-200}"
export MB_THERMAL_DURATION="${MB_THERMAL_DURATION:-120}"
FAIL_FAST="${FAIL_FAST:-false}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── State ─────────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0; WARN=0
declare -A TEST_RESULTS   # name → PASS|FAIL|SKIP|WARN
declare -A TEST_DETAILS   # name → detail message

# =============================================================================
# Helpers
# =============================================================================
log()  { echo -e "$(date '+%H:%M:%S') $*" | tee -a "$LOG_FILE"; }
info() { log "${BLUE}[INFO]${RESET}  $*"; }
ok()   { log "${GREEN}[PASS]${RESET}  $*"; }
fail() { log "${RED}[FAIL]${RESET}  $*"; }
warn() { log "${YELLOW}[WARN]${RESET}  $*"; }
skip() { log "${CYAN}[SKIP]${RESET}  $*"; }
section() {
  log ""
  log "${BOLD}${BLUE}════════════════════════════════════════${RESET}"
  log "${BOLD}${BLUE}  $*${RESET}"
  log "${BOLD}${BLUE}════════════════════════════════════════${RESET}"
}

record_pass() { TEST_RESULTS["$1"]="PASS"; TEST_DETAILS["$1"]="${2:-}"; ((PASS++)); ok  "$1${2:+ — $2}"; }
record_fail() { TEST_RESULTS["$1"]="FAIL"; TEST_DETAILS["$1"]="${2:-}"; ((FAIL++)); fail "$1${2:+ — $2}";
                [[ "$FAIL_FAST" == "true" ]] && { log "FAIL_FAST=true — aborting"; print_summary; exit 1; }; }
record_skip() { TEST_RESULTS["$1"]="SKIP"; TEST_DETAILS["$1"]="${2:-}"; ((SKIP++)); skip "$1${2:+ — $2}"; }
record_warn() { TEST_RESULTS["$1"]="WARN"; TEST_DETAILS["$1"]="${2:-}"; ((WARN++)); warn "$1${2:+ — $2}"; }

cmd_exists() { command -v "$1" &>/dev/null; }
run_cmd()    { "$@" 2>&1 | tee -a "$LOG_FILE" || true; }

# Save output of a command to results dir AND log the command + output clearly
capture() {
  local name="$1"; shift
  local outfile="$RESULTS_DIR/${name}.txt"
  local cmd_str="$*"
  {
    echo "════════════════════════════════════════"
    echo "CMD : $cmd_str"
    echo "TIME: $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════"
    { "$@" 2>&1 || true; }
    echo ""
  } | tee "$outfile" >> "$LOG_FILE"
}

# =============================================================================
# Source shared helpers first, then individual test modules
# =============================================================================
source "$LIB_DIR/_common.sh"
source "$LIB_DIR/01_os_kernel.sh"
source "$LIB_DIR/02_rocm_stack.sh"
source "$LIB_DIR/03_gpu_enumeration.sh"
source "$LIB_DIR/04_observability.sh"
source "$LIB_DIR/05_docker_runtime.sh"
source "$LIB_DIR/06_vllm.sh"
source "$LIB_DIR/07_rccl.sh"
source "$LIB_DIR/08_hip_compute.sh"
source "$LIB_DIR/09_power_thermal.sh"
source "$LIB_DIR/10_permissions.sh"
source "$LIB_DIR/11_package_consistency.sh"
source "$LIB_DIR/12_stress_longevity.sh"
source "$LIB_DIR/13_failure_recovery.sh"
source "$LIB_DIR/14_regression_matrix.sh"
source "$LIB_DIR/15_gpu_microbenchmarks.sh"

# =============================================================================
# Summary
# =============================================================================
print_summary() {
  local total=$((PASS + FAIL + SKIP + WARN))
  section "FINAL SUMMARY"
  log "${BOLD}Total checks : $total${RESET}"
  log "${GREEN}Passed       : $PASS${RESET}"
  log "${RED}Failed       : $FAIL${RESET}"
  log "${YELLOW}Warnings     : $WARN${RESET}"
  log "${CYAN}Skipped      : $SKIP${RESET}"
  log ""
  log "${BOLD}Per-test breakdown:${RESET}"
  for t in "${!TEST_RESULTS[@]}"; do
    local status="${TEST_RESULTS[$t]}"
    local detail="${TEST_DETAILS[$t]}"
    case "$status" in
      PASS) log "  ${GREEN}✔${RESET} $t${detail:+: $detail}" ;;
      FAIL) log "  ${RED}✘${RESET} $t${detail:+: $detail}" ;;
      WARN) log "  ${YELLOW}⚠${RESET} $t${detail:+: $detail}" ;;
      SKIP) log "  ${CYAN}−${RESET} $t${detail:+: $detail}" ;;
    esac
  done
  log ""

  # Write machine-readable JSON summary
  python3 - <<PYEOF
import json, os, datetime
results = {}
$(for t in "${!TEST_RESULTS[@]}"; do
    echo "results[$(printf '%q' "$t")] = {\"status\": $(printf '%q' "${TEST_RESULTS[$t]}"), \"detail\": $(printf '%q' "${TEST_DETAILS[$t]:-}")}";
  done)
summary = {
    "timestamp": "$(date -Iseconds)",
    "rocm_expected": "$ROCM_EXPECTED_VERSION",
    "baseline": "$BASELINE_VERSION",
    "host": "$(hostname)",
    "totals": {"pass": $PASS, "fail": $FAIL, "warn": $WARN, "skip": $SKIP},
    "tests": results
}
with open("$RESULTS_DIR/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary written to $RESULTS_DIR/summary.json")
PYEOF

  if [[ $FAIL -eq 0 ]]; then
    log "${GREEN}${BOLD}✔ ALL CHECKS PASSED — Droplet is production-ready${RESET}"
    return 0
  else
    log "${RED}${BOLD}✘ $FAIL CHECK(S) FAILED — Review log: $LOG_FILE${RESET}"
    return 1
  fi
}

# =============================================================================
# MAIN
# =============================================================================
main() {
  mkdir -p "$RESULTS_DIR"
  log "${BOLD}ROCm $ROCM_EXPECTED_VERSION GPU Droplet Validation Suite${RESET}"
  log "Results dir : $RESULTS_DIR"
  log "Log file    : $LOG_FILE"
  log "Host        : $(hostname)"
  log "Date        : $(date)"
  log ""

  test_os_kernel
  test_rocm_stack
  test_gpu_enumeration
  test_observability
  test_docker_runtime
  [[ "$SKIP_VLLM"   != "true" ]] && test_vllm   || record_skip "vllm_suite" "SKIP_VLLM=true"
  [[ "$SKIP_RCCL"   != "true" ]] && test_rccl   || record_skip "rccl_suite" "SKIP_RCCL=true"
  test_hip_compute
  test_power_thermal
  test_permissions
  test_package_consistency
  [[ "$SKIP_STRESS" != "true" ]] && test_stress_longevity || record_skip "stress_suite" "SKIP_STRESS=true"
  test_failure_recovery
  [[ "$SKIP_MICROBENCH" != "true" ]] && test_gpu_microbenchmarks || record_skip "microbench_suite" "SKIP_MICROBENCH=true"
  test_regression_matrix

  print_summary
}

main "$@"
