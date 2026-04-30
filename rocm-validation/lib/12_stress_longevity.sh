#!/usr/bin/env bash
# Module 12 — Stress & Longevity

test_stress_longevity() {
  section "12 · STRESS & LONGEVITY"

  # ── GPU Stress with stress-ng ─────────────────────────────────────────────
  if cmd_exists stress-ng; then
    info "Running GPU stress for ${STRESS_DURATION}s..."
    local stress_log="$RESULTS_DIR/stress_ng.log"

    stress-ng --gpu 1 --timeout "${STRESS_DURATION}" \
              --metrics-brief \
              --log-file "$stress_log" 2>&1 | tee -a "$LOG_FILE" &
    local stress_pid=$!

    # Monitor temps every 30s during stress
    local temp_spike=0
    while kill -0 "$stress_pid" 2>/dev/null; do
      sleep 30
      local curr_max=0
      while read -r t; do
        t_int="${t%%.*}"
        [[ "$t_int" =~ ^[0-9]+$ && "$t_int" -gt "$curr_max" ]] && curr_max="$t_int"
      done < <(rocm-smi --showtemp 2>/dev/null | grep -oP "[0-9]+(\.[0-9]+)?(?=c)" || true)
      [[ "$curr_max" -gt "$temp_spike" ]] && temp_spike="$curr_max"
      info "  Stress in progress... peak temp so far: ${temp_spike}°C"
    done

    wait "$stress_pid" 2>/dev/null || true

    if [[ "$temp_spike" -le 90 ]]; then
      record_pass "stress_temperature" "Peak temperature under stress: ${temp_spike}°C"
    else
      record_fail "stress_temperature" "Temperature spiked to ${temp_spike}°C under stress"
    fi

    # Check for crashes post-stress
    local post_stress_resets
    post_stress_resets=$(dmesg 2>/dev/null | grep -ciE "gpu reset|amdgpu.*reset" || echo "0")
    if [[ "$post_stress_resets" -eq 0 ]]; then
      record_pass "stress_no_gpu_resets"
    else
      record_fail "stress_no_gpu_resets" "$post_stress_resets GPU reset(s) after stress"
    fi

    # GPUs still enumerated post-stress
    local post_gpu_count
    post_gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type.*GPU" || echo "0")
    if [[ "$post_gpu_count" -gt 0 ]]; then
      record_pass "stress_gpus_survive" "$post_gpu_count GPU(s) still detected after stress"
    else
      record_fail "stress_gpus_survive" "GPUs lost after stress test"
    fi
  else
    record_skip "stress_ng" "stress-ng not installed (install with: apt-get install stress-ng)"
  fi

  # ── rocm-bandwidth-test (memory bandwidth stress) ─────────────────────────
  if find /opt/rocm/bin -name "rocm_bandwidth_test" 2>/dev/null | grep -q .; then
    local bw_bin
    bw_bin=$(find /opt/rocm/bin -name "rocm_bandwidth_test" | head -1)
    local bw_out="$RESULTS_DIR/rocm_bandwidth_test.txt"
    info "Running rocm_bandwidth_test..."
    "$bw_bin" 2>&1 | tee "$bw_out" | tee -a "$LOG_FILE" || true
    if grep -qi "pass\|GB/s" "$bw_out" 2>/dev/null; then
      record_pass "rocm_bandwidth_test" "Completed"
    else
      record_warn "rocm_bandwidth_test" "Unexpected output"
    fi
  else
    record_skip "rocm_bandwidth_test" "rocm_bandwidth_test not found"
  fi

  # ── Continuous vLLM load (longevity) ─────────────────────────────────────
  if [[ "$SKIP_VLLM" != "true" && "$VLLM_LONGEVITY_DURATION" -gt 0 ]]; then
    info "Checking if vLLM server is reachable for longevity test..."
    if _vllm_running 2>/dev/null || {
        # Try to start
        export -f _start_vllm _stop_vllm _vllm_running cmd_exists info warn record_pass record_fail record_skip record_warn
        _start_vllm "$TP_SIZE"
    }; then
      info "Starting ${VLLM_LONGEVITY_DURATION}s vLLM longevity run..."
      local end_time=$(( $(date +%s) + VLLM_LONGEVITY_DURATION ))
      local req_count=0 err_count=0

      while [[ $(date +%s) -lt "$end_time" ]]; do
        local resp
        resp=$(curl -sf --max-time 30 \
          -H "Content-Type: application/json" \
          -d '{"model":"'"$MODEL"'","prompt":"Hello","max_tokens":10}' \
          "http://${VLLM_HOST}:${VLLM_PORT}/v1/completions" 2>/dev/null || echo "")
        if [[ -n "$resp" ]]; then
          ((req_count++))
        else
          ((err_count++))
          warn "longevity: request $((req_count+err_count)) failed"
        fi
        sleep 5
      done

      local total=$((req_count + err_count))
      local success_rate=0
      [[ "$total" -gt 0 ]] && success_rate=$(( req_count * 100 / total ))

      if [[ "$success_rate" -ge 95 ]]; then
        record_pass "vllm_longevity" "${VLLM_LONGEVITY_DURATION}s run: $req_count/$total succeeded (${success_rate}%)"
      else
        record_fail "vllm_longevity" "Success rate ${success_rate}% ($err_count errors) over ${VLLM_LONGEVITY_DURATION}s"
      fi
    else
      record_skip "vllm_longevity" "vLLM server not available"
    fi
  else
    record_skip "vllm_longevity" "SKIP_VLLM=true or VLLM_LONGEVITY_DURATION=0"
  fi

  # ── Memory leak check ─────────────────────────────────────────────────────
  local vram_used_before vram_used_after
  vram_used_before=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP "VRAM Used Memory.*:\s*\K[0-9]+" | head -1 || echo "0")
  # Run a quick workload
  rocminfo &>/dev/null; rocm-smi &>/dev/null
  vram_used_after=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP "VRAM Used Memory.*:\s*\K[0-9]+" | head -1 || echo "0")
  local vram_delta=$(( vram_used_after - vram_used_before ))
  if [[ "${vram_delta#-}" -lt 500000000 ]]; then  # <500 MB growth
    record_pass "no_vram_leak" "VRAM delta: ${vram_delta} bytes (stable)"
  else
    record_warn "no_vram_leak" "VRAM grew by ${vram_delta} bytes — potential leak"
  fi
}
