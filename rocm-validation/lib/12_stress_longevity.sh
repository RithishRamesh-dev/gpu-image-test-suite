#!/usr/bin/env bash
# Module 12 — Stress & Longevity

test_stress_longevity() {
  section "12 · STRESS & LONGEVITY"

  # ── GPU Stress with stress-ng ─────────────────────────────────────────────
  if cmd_exists stress-ng; then
    info "Running GPU stress for ${STRESS_DURATION}s..."
    local stress_log="$RESULTS_DIR/stress_ng.log"

    stress-ng --gpu 1 --timeout "${STRESS_DURATION}" \
              --metrics-brief --log-file "$stress_log" 2>&1 | tee -a "$LOG_FILE" &
    local stress_pid=$!

    local temp_spike=0
    while kill -0 "$stress_pid" 2>/dev/null; do
      sleep 30
      local curr_max
      curr_max=$(rocm-smi --showtemp 2>/dev/null | \
                 grep -oP "[0-9]+(?:\.[0-9]+)?(?=c)" | \
                 awk 'BEGIN{m=0} {v=int($1); if(v>m) m=v} END{print m}')
      curr_max=$(_int "$curr_max")
      [[ "$curr_max" -gt "$temp_spike" ]] && temp_spike="$curr_max"
      info "  Stress in progress... peak temp: ${temp_spike}°C"
    done
    wait "$stress_pid" 2>/dev/null || true

    if [[ "$temp_spike" -le 90 ]]; then
      record_pass "stress_temperature" "Peak temperature under stress: ${temp_spike}°C"
    else
      record_fail "stress_temperature" "Temperature spiked to ${temp_spike}°C under stress"
    fi

    local post_stress_resets
    post_stress_resets=$(dmesg 2>/dev/null | grep -iE "gpu reset|amdgpu.*reset" | \
                         wc -l | tr -d '[:space:]')
    post_stress_resets=$(_int "$post_stress_resets")
    if [[ "$post_stress_resets" -eq 0 ]]; then
      record_pass "stress_no_gpu_resets"
    else
      record_fail "stress_no_gpu_resets" "$post_stress_resets GPU reset(s) after stress"
    fi

    local post_gpu_count
    post_gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type.*GPU" | tr -d '[:space:]')
    post_gpu_count=$(_int "$post_gpu_count")
    if [[ "$post_gpu_count" -gt 0 ]]; then
      record_pass "stress_gpus_survive" "$post_gpu_count GPU(s) still detected after stress"
    else
      record_fail "stress_gpus_survive" "GPUs lost after stress test"
    fi
  else
    record_skip "stress_ng" "stress-ng not installed (apt-get install stress-ng)"
  fi

  # ── rocm-bandwidth-test ───────────────────────────────────────────────────
  if find /opt/rocm/bin -name "rocm_bandwidth_test" 2>/dev/null | grep -q .; then
    local bw_bin
    bw_bin=$(find /opt/rocm/bin -name "rocm_bandwidth_test" | head -1)
    local bw_out="$RESULTS_DIR/rocm_bandwidth_test.txt"
    info "Running rocm_bandwidth_test..."
    "$bw_bin" 2>&1 | tee "$bw_out" | tee -a "$LOG_FILE" || true
    if grep -qiE "pass|GB/s" "$bw_out" 2>/dev/null; then
      record_pass "rocm_bandwidth_test" "Completed"
    else
      record_warn "rocm_bandwidth_test" "Unexpected output"
    fi
  else
    record_skip "rocm_bandwidth_test" "rocm_bandwidth_test not found"
  fi

  # ── Continuous vLLM longevity ─────────────────────────────────────────────
  if [[ "$SKIP_VLLM" != "true" && "$VLLM_LONGEVITY_DURATION" -gt 0 ]]; then
    if curl -sf --max-time 5 "http://${VLLM_HOST}:${VLLM_PORT}/health" &>/dev/null; then
      info "Starting ${VLLM_LONGEVITY_DURATION}s vLLM longevity run..."
      local end_time req_count=0 err_count=0
      end_time=$(( $(date +%s) + VLLM_LONGEVITY_DURATION ))

      while [[ $(date +%s) -lt "$end_time" ]]; do
        local resp
        resp=$(curl -sf --max-time 30 \
          -H "Content-Type: application/json" \
          -d '{"model":"'"$MODEL"'","prompt":"Hello","max_tokens":10}' \
          "http://${VLLM_HOST}:${VLLM_PORT}/v1/completions" 2>/dev/null || echo "")
        if [[ -n "$resp" ]]; then
          (( req_count++ )) || true
        else
          (( err_count++ )) || true
          warn "longevity: request failed ($err_count so far)"
        fi
        sleep 5
      done

      local total=$(( req_count + err_count ))
      local success_rate=0
      [[ "$total" -gt 0 ]] && success_rate=$(( req_count * 100 / total ))

      if [[ "$success_rate" -ge 95 ]]; then
        record_pass "vllm_longevity" \
          "${VLLM_LONGEVITY_DURATION}s: $req_count/$total succeeded (${success_rate}%)"
      else
        record_fail "vllm_longevity" \
          "Success rate ${success_rate}% ($err_count errors) over ${VLLM_LONGEVITY_DURATION}s"
      fi
    else
      record_skip "vllm_longevity" "vLLM server not reachable at ${VLLM_HOST}:${VLLM_PORT}"
    fi
  else
    record_skip "vllm_longevity" "SKIP_VLLM=true or VLLM_LONGEVITY_DURATION=0"
  fi

  # ── VRAM leak check ───────────────────────────────────────────────────────
  # VRAM leak check — use Python to handle large byte values safely
  local vram_result
  vram_result=$(python3 - 2>/dev/null <<'PYEOF2'
import subprocess, re
def get_used():
    out = subprocess.run(["rocm-smi","--showmeminfo","vram"],
        capture_output=True, text=True).stdout
    vals = re.findall(r"VRAM Used Memory.*?:\s*(\d+)", out)
    return sum(int(v) for v in vals)
before = get_used()
import os; os.system("rocminfo > /dev/null 2>&1"); os.system("rocm-smi > /dev/null 2>&1")
after  = get_used()
delta  = after - before
abs_d  = abs(delta)
status = "PASS" if abs_d < 500_000_000 else "WARN"
print(f"{status}:{delta}")
PYEOF2
)
  local vram_status vram_delta
  IFS=: read -r vram_status vram_delta <<< "${vram_result:-WARN:unknown}"
  if [[ "$vram_status" == "PASS" ]]; then
    record_pass "no_vram_leak" "VRAM delta: ${vram_delta} bytes (stable)"
  else
    record_warn "no_vram_leak" "VRAM grew by ${vram_delta} bytes — potential leak"
  fi
}
