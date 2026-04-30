#!/usr/bin/env bash
# Module 07 — RCCL Multi-GPU Communication

RCCL_TESTS_DIR="${RCCL_TESTS_DIR:-/opt/rccl-tests/build}"
RCCL_GPU_COUNT="${TP_SIZE}"
RCCL_MIN_BW_GBPS="${RCCL_MIN_BW_GBPS:-100}"  # minimum acceptable bus BW per GPU-pair (GB/s)

_run_rccl_test() {
  local binary="$1"; shift
  local test_name="$1"; shift
  local extra_args="${*:-}"
  local outfile="$RESULTS_DIR/rccl_${test_name}.txt"

  if [[ ! -x "$RCCL_TESTS_DIR/$binary" ]]; then
    record_skip "rccl_${test_name}" "$RCCL_TESTS_DIR/$binary not found — build rccl-tests first"
    return
  fi

  info "RCCL: $binary $extra_args"
  # shellcheck disable=SC2086
  "$RCCL_TESTS_DIR/$binary" \
    -b 16G -e 16G \
    -g "$RCCL_GPU_COUNT" \
    $extra_args 2>&1 | tee "$outfile" | tee -a "$LOG_FILE" || true

  # Extract bus bandwidth
  local bw
  bw=$(grep -oP "^\s*[0-9]+\s+.*\s+\K[0-9]+\.[0-9]+" "$outfile" 2>/dev/null | tail -1 || echo "0")
  echo "$bw" > "$RESULTS_DIR/rccl_${test_name}_bw.txt"

  # Check for hangs — file should have content
  local lines
  lines=$(wc -l < "$outfile" 2>/dev/null || echo "0")
  if [[ "$lines" -lt 5 ]]; then
    record_fail "rccl_${test_name}" "Output suspiciously short ($lines lines) — possible hang"
    return
  fi

  # Check for errors in output
  if grep -qiE "^#.*ERROR\|RCCL Error\|failed" "$outfile" 2>/dev/null; then
    record_fail "rccl_${test_name}" "Errors detected in output"
    return
  fi

  # Check bandwidth threshold
  if awk "BEGIN{exit !($bw > 0 && $bw < $RCCL_MIN_BW_GBPS)}"; then
    record_warn "rccl_${test_name}" "Bandwidth ${bw} GB/s below threshold ${RCCL_MIN_BW_GBPS} GB/s"
  else
    record_pass "rccl_${test_name}" "Bus bandwidth: ${bw} GB/s"
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
    info "rccl-tests not found at $RCCL_TESTS_DIR, attempting build..."
    if [[ -d /opt/rccl-tests ]] || git clone https://github.com/ROCmSoftwarePlatform/rccl-tests /opt/rccl-tests 2>&1 | tee -a "$LOG_FILE"; then
      pushd /opt/rccl-tests >/dev/null
      make MPI=0 HIP_HOME=/opt/rocm 2>&1 | tee -a "$LOG_FILE" || true
      popd >/dev/null
    fi
  fi

  # Core collective operations
  _run_rccl_test all_reduce_perf     "all_reduce"
  _run_rccl_test all_gather_perf     "all_gather"
  _run_rccl_test reduce_scatter_perf "reduce_scatter"
  _run_rccl_test broadcast_perf      "broadcast"
  _run_rccl_test reduce_perf         "reduce"
  _run_rccl_test alltoall_perf       "all_to_all"   "-b 1G -e 8G"

  # In-place test for all_reduce
  _run_rccl_test all_reduce_perf "all_reduce_inplace" "--op sum"

  # Different data types
  for dtype in fp16 bf16 fp32; do
    _run_rccl_test all_reduce_perf "all_reduce_${dtype}" "-d $dtype"
  done

  # Multi-process mode (if MPI available)
  if cmd_exists mpirun; then
    info "Testing RCCL with MPI..."
    _run_rccl_test all_reduce_perf "all_reduce_mpi" "-m 1"
    record_pass "rccl_mpi_available"
  else
    record_skip "rccl_mpi_available" "mpirun not found — MPI test skipped"
  fi

  # RCCL environment variable checks
  info "RCCL_TOPO_FILE=${RCCL_TOPO_FILE:-not set}"
  info "NCCL_DEBUG=${NCCL_DEBUG:-not set}"

  # Check RCCL library presence
  if find /opt/rocm /usr/lib 2>/dev/null | grep -q "librccl.so"; then
    record_pass "rccl_library_found" "librccl.so located"
  else
    record_warn "rccl_library_found" "librccl.so not found in standard paths"
  fi

  # Generate bandwidth comparison CSV
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
