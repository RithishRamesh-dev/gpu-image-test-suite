#!/usr/bin/env bash
# Module 14 — Regression Comparison Matrix

BASELINE_RESULTS_DIR="${BASELINE_RESULTS_DIR:-}"

test_regression_matrix() {
  section "14 · REGRESSION COMPARISON MATRIX"

  # Build this run's metrics snapshot
  local metrics_file="$RESULTS_DIR/regression_metrics.json"
  python3 - "$metrics_file" <<'PYEOF'
import json, subprocess, sys, re

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""

def parse_float(s):
    m = re.search(r'[0-9]+(?:\.[0-9]+)?', s)
    return float(m.group()) if m else None

metrics = {}

# GPU count
out = run("rocminfo 2>/dev/null | grep -c 'Device Type.*GPU'")
metrics["gpu_count"] = int(out) if out.isdigit() else None

# ROCm version
ver = run("cat /opt/rocm/.info/version 2>/dev/null | head -1")
metrics["rocm_version"] = ver or None

# RCCL bandwidth (all_reduce)
bw_file = sys.argv[1].replace("regression_metrics.json", "rccl_all_reduce_bw.txt")
try:
    with open(bw_file) as f:
        metrics["rccl_all_reduce_bw_gbs"] = parse_float(f.read())
except FileNotFoundError:
    metrics["rccl_all_reduce_bw_gbs"] = None

# vLLM throughput
bench_file = sys.argv[1].replace("regression_metrics.json", "vllm_bench_tp8_standard_throughput.txt")
try:
    with open(bench_file) as f:
        metrics["vllm_throughput_req_s"] = parse_float(f.read())
except FileNotFoundError:
    metrics["vllm_throughput_req_s"] = None

# Memory bandwidth
bw_out = run("rocm_bandwidth_test 2>/dev/null | grep -oP '[0-9]+\.[0-9]+(?=\s*GB/s)' | tail -1")
metrics["memory_bandwidth_gbs"] = parse_float(bw_out) if bw_out else None

# Peak temp
temp_out = run("rocm-smi --showtemp 2>/dev/null | grep -oP '[0-9]+(?=c)' | sort -rn | head -1")
metrics["peak_temp_c"] = int(temp_out) if temp_out.isdigit() else None

with open(sys.argv[1], "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Regression metrics saved to {sys.argv[1]}")
PYEOF

  # Compare against baseline if available
  if [[ -n "$BASELINE_RESULTS_DIR" && -f "$BASELINE_RESULTS_DIR/regression_metrics.json" ]]; then
    info "Comparing against baseline: $BASELINE_RESULTS_DIR"
    python3 - "$BASELINE_RESULTS_DIR/regression_metrics.json" "$metrics_file" "$RESULTS_DIR/regression_delta.json" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f: baseline = json.load(f)
with open(sys.argv[2]) as f: current  = json.load(f)

THRESHOLDS = {
    "rccl_all_reduce_bw_gbs":   0.95,  # allow 5% regression
    "vllm_throughput_req_s":    0.95,
    "memory_bandwidth_gbs":     0.95,
    "peak_temp_c":              1.10,  # allow 10% increase (warn if higher)
}

deltas = {}
for k, baseline_val in baseline.items():
    current_val = current.get(k)
    if baseline_val is None or current_val is None:
        deltas[k] = {"baseline": baseline_val, "current": current_val, "status": "SKIP"}
        continue
    if isinstance(baseline_val, (int, float)) and baseline_val != 0:
        ratio = current_val / baseline_val
        threshold = THRESHOLDS.get(k, 1.0)
        if k == "peak_temp_c":
            status = "PASS" if ratio <= threshold else "WARN"
        else:
            status = "PASS" if ratio >= threshold else "FAIL"
        deltas[k] = {
            "baseline": baseline_val,
            "current": current_val,
            "ratio": round(ratio, 3),
            "pct_change": round((ratio - 1) * 100, 1),
            "status": status
        }
    else:
        deltas[k] = {"baseline": baseline_val, "current": current_val,
                     "status": "PASS" if baseline_val == current_val else "WARN"}

with open(sys.argv[3], "w") as f:
    json.dump(deltas, f, indent=2)

# Print table
print(f"\n{'Metric':<35} {'Baseline':>12} {'Current':>12} {'Change':>8} {'Status':>6}")
print("-" * 80)
for k, d in deltas.items():
    pct = f"{d.get('pct_change', 'N/A'):+.1f}%" if isinstance(d.get('pct_change'), float) else "N/A"
    status = d['status']
    sym = "✔" if status == "PASS" else ("✘" if status == "FAIL" else "⚠" if status == "WARN" else "−")
    print(f"{sym} {k:<33} {str(d['baseline']):>12} {str(d['current']):>12} {pct:>8} {status:>6}")
PYEOF

    # Read delta results and record
    local regression_lines
    regression_lines=$(python3 - "$RESULTS_DIR/regression_delta.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    deltas = json.load(f)
for k, d in deltas.items():
    print(f"REGRESSION:{k}:{d['status']}:{d.get('current','N/A')}")
PYEOF
)
    while IFS=: read -r _ metric status value; do
        case "$status" in
          PASS) record_pass "regression_${metric}" "value=$value" ;;
          FAIL) record_fail "regression_${metric}" "value=$value (regression detected)" ;;
          WARN) record_warn "regression_${metric}" "value=$value" ;;
          SKIP) record_skip "regression_${metric}" "insufficient data" ;;
        esac
    done <<< "$regression_lines"
  else
    info "No baseline results dir provided — saving current metrics for future comparison"
    info "Set BASELINE_RESULTS_DIR=<previous run dir> to enable regression comparison"
    info "Current metrics saved to: $metrics_file"
    record_skip "regression_comparison" "No BASELINE_RESULTS_DIR set"
  fi

  # Print final matrix table
  {
    echo ""
    echo "┌─────────────────────────────────┬───────────────────────────────────────┐"
    echo "│ Area                            │ Result                                │"
    echo "├─────────────────────────────────┼───────────────────────────────────────┤"
    local gpu_count rocm_ver
    gpu_count=$(python3 -c "import json; d=json.load(open('$metrics_file')); print(d.get('gpu_count','?'))" 2>/dev/null || echo "?")
    rocm_ver=$(python3 -c "import json; d=json.load(open('$metrics_file')); print(d.get('rocm_version','?'))" 2>/dev/null || echo "?")
    echo "│ GPU Detection                   │ $gpu_count GPU(s)                              │"
    echo "│ ROCm Version                    │ ${rocm_ver}                                    │"
    echo "│ RCCL Bandwidth                  │ see $RESULTS_DIR/rccl_bandwidth_summary.csv    │"
    echo "│ vLLM Throughput                 │ see $RESULTS_DIR/vllm_benchmark_summary.csv    │"
    echo "│ Regression vs $BASELINE_VERSION             │ see $RESULTS_DIR/regression_delta.json        │"
    echo "└─────────────────────────────────┴───────────────────────────────────────┘"
  } | tee -a "$LOG_FILE"
}
