#!/usr/bin/env bash
# Module 07 — RCCL Multi-GPU Communication
#
# Designed to complete in ~2-3 minutes total:
#   - Single fixed message size (1G) — no sweep
#   - timeout wrapper on every binary call
#   - Only core collectives; dtype variants skipped by default

RCCL_TESTS_DIR="${RCCL_TESTS_DIR:-/opt/rccl-tests/build}"
RCCL_GPU_COUNT="${TP_SIZE}"
RCCL_MIN_BW_GBPS="${RCCL_MIN_BW_GBPS:-100}"
RCCL_MSG_SIZE="${RCCL_MSG_SIZE:-1G}"      # single message size — no sweep
RCCL_ITERS="${RCCL_ITERS:-20}"            # iterations per test
RCCL_TIMEOUT="${RCCL_TIMEOUT:-120}"       # seconds per binary before kill

_run_rccl_test() {
  local binary="$1"; shift
  local test_name="$1"; shift
  local extra_args="${*:-}"
  local outfile="$RESULTS_DIR/rccl_${test_name}.txt"

  if [[ ! -x "$RCCL_TESTS_DIR/$binary" ]]; then
    record_skip "rccl_${test_name}" "$RCCL_TESTS_DIR/$binary not found"
    return
  fi

  info "RCCL: $binary -b $RCCL_MSG_SIZE -e $RCCL_MSG_SIZE -n $RCCL_ITERS -g $RCCL_GPU_COUNT $extra_args (timeout ${RCCL_TIMEOUT}s)"

  # timeout prevents any single test from hanging the suite
  # shellcheck disable=SC2086
  timeout "$RCCL_TIMEOUT" \
    "$RCCL_TESTS_DIR/$binary" \
      -b "$RCCL_MSG_SIZE" \
      -e "$RCCL_MSG_SIZE" \
      -n "$RCCL_ITERS" \
      -g "$RCCL_GPU_COUNT" \
      $extra_args 2>&1 | tee "$outfile" | tee -a "$LOG_FILE" || true

  local exit_code="${PIPESTATUS[0]}"

  # timeout exits 124 on kill
  if [[ "$exit_code" -eq 124 ]]; then
    record_fail "rccl_${test_name}" "Timed out after ${RCCL_TIMEOUT}s — possible hang"
    return
  fi

  # Output sanity
  local lines
  lines=$(wc -l < "$outfile" 2>/dev/null | tr -d '[:space:]')
  lines=$(_int "$lines")
  if [[ "$lines" -lt 3 ]]; then
    record_fail "rccl_${test_name}" "Output too short ($lines lines)"
    return
  fi

  # Error detection
  if grep -qiE "RCCL Error|^# ERROR|failed\b" "$outfile" 2>/dev/null; then
    record_fail "rccl_${test_name}" "Errors detected in output"
    return
  fi

  # Extract bus bandwidth from last data row (rightmost float before N/A or newline)
  local bw
  bw=$(grep -oP "^\s*[0-9].*\s\K[0-9]+\.[0-9]+" "$outfile" 2>/dev/null | tail -1 || echo "0")
  bw="${bw:-0}"
  echo "$bw" > "$RESULTS_DIR/rccl_${test_name}_bw.txt"

  if awk "BEGIN{exit !($bw > 0 && $bw < $RCCL_MIN_BW_GBPS)}"; then
    record_warn "rccl_${test_name}" "Bus BW ${bw} GB/s below threshold ${RCCL_MIN_BW_GBPS} GB/s"
  else
    record_pass "rccl_${test_name}" "Bus BW: ${bw} GB/s (msg=${RCCL_MSG_SIZE})"
  fi
}

test_rccl() {
  section "07 · RCCL MULTI-GPU COMMUNICATION"

  if [[ "$RCCL_GPU_COUNT" -lt 2 ]]; then
    record_skip "rccl_suite" "TP_SIZE=$TP_SIZE — RCCL tests require ≥ 2 GPUs"
    return
  fi

  # Build rccl-tests if not present
  if [[ ! -d "$RCCL_TESTS_DIR" ]]; then
    info "rccl-tests not found, attempting build (may take a few minutes)..."
    if git clone --depth=1 https://github.com/ROCmSoftwarePlatform/rccl-tests \
        /opt/rccl-tests 2>&1 | tee -a "$LOG_FILE"; then
      pushd /opt/rccl-tests >/dev/null
      make MPI=0 HIP_HOME=/opt/rocm 2>&1 | tee -a "$LOG_FILE" || true
      popd >/dev/null
    fi
  fi

  # RCCL library check
  if find /opt/rocm /usr/lib 2>/dev/null -name "librccl.so*" -print -quit | grep -q .; then
    record_pass "rccl_library_found" "librccl.so present"
  else
    record_warn "rccl_library_found" "librccl.so not found"
  fi

  # Core collectives — each capped at RCCL_TIMEOUT seconds
  _run_rccl_test all_reduce_perf     "all_reduce"
  _run_rccl_test all_gather_perf     "all_gather"
  _run_rccl_test reduce_scatter_perf "reduce_scatter"
  _run_rccl_test broadcast_perf      "broadcast"
  _run_rccl_test reduce_perf         "reduce"

  # MPI mode (optional)
  if cmd_exists mpirun; then
    _run_rccl_test all_reduce_perf "all_reduce_mpi" "-m 1"
    record_pass "rccl_mpi_available"
  else
    record_skip "rccl_mpi_available" "mpirun not found"
  fi

  # Bandwidth summary CSV
  {
    echo "test,bandwidth_GBs"
    for f in "$RESULTS_DIR"/rccl_*_bw.txt; do
      [[ -f "$f" ]] || continue
      local name bw
      name=$(basename "$f" _bw.txt | sed 's/rccl_//')
      bw=$(cat "$f")
      echo "$name,$bw"
    done
  } > "$RESULTS_DIR/rccl_bandwidth_summary.csv"
  info "RCCL bandwidth summary: $RESULTS_DIR/rccl_bandwidth_summary.csv"
}
