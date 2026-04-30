#!/usr/bin/env bash
# Module 09 — Power + Thermal

MAX_SAFE_TEMP="${MAX_SAFE_TEMP:-85}"
MAX_SAFE_POWER_W="${MAX_SAFE_POWER_W:-500}"

test_power_thermal() {
  section "09 · POWER + THERMAL"

  capture "rocm_smi_power" bash -c "rocm-smi --showpower  2>/dev/null || true"
  capture "rocm_smi_temp"  bash -c "rocm-smi --showtemp   2>/dev/null || true"
  capture "rocm_smi_fan"   bash -c "rocm-smi --showfan    2>/dev/null || true"
  capture "rocm_smi_clock" bash -c "rocm-smi --showclocks 2>/dev/null || true"

  # Temperature — parse all °c values via awk, one record per GPU
  local gpu_idx=0 max_temp=0
  while read -r temp_val; do
    local temp_int
    temp_int=$(_int "${temp_val%%.*}")
    [[ "$temp_int" =~ ^[0-9]+$ ]] || continue
    [[ "$temp_int" -gt "$max_temp" ]] && max_temp="$temp_int"
    if [[ "$temp_int" -gt "$MAX_SAFE_TEMP" ]]; then
      record_fail "temp_gpu${gpu_idx}" "${temp_int}°C exceeds max ${MAX_SAFE_TEMP}°C"
    else
      record_pass "temp_gpu${gpu_idx}" "${temp_int}°C"
    fi
    (( gpu_idx++ )) || true
  done < <(rocm-smi --showtemp 2>/dev/null | grep -oP "[0-9]+(?:\.[0-9]+)?(?=c)" || true)

  if [[ "$gpu_idx" -eq 0 ]]; then
    record_warn "temperature_readings" "No temperature values parsed"
  fi

  # Throttle state — count lines containing throttle, strip whitespace
  local throttle_count
  throttle_count=$(rocm-smi --showrasinfo all 2>/dev/null | \
                   grep -i "throttle\|thermal_throttle" | wc -l | tr -d '[:space:]')
  throttle_count=$(_int "$throttle_count")
  if [[ "$throttle_count" -eq 0 ]]; then
    record_pass "no_thermal_throttle" "No throttle events detected"
  else
    record_warn "no_thermal_throttle" "$throttle_count throttle event(s)"
  fi

  # Power draw
  local total_power=0 power_gpu_idx=0
  while read -r pwr; do
    local pwr_int
    pwr_int=$(_int "${pwr%%.*}")
    [[ "$pwr_int" =~ ^[0-9]+$ ]] || continue
    total_power=$(( total_power + pwr_int ))
    if [[ "$pwr_int" -gt "$MAX_SAFE_POWER_W" ]]; then
      record_warn "power_gpu${power_gpu_idx}" "${pwr_int}W exceeds expected max ${MAX_SAFE_POWER_W}W"
    else
      record_pass "power_gpu${power_gpu_idx}" "${pwr_int}W"
    fi
    (( power_gpu_idx++ )) || true
  done < <(rocm-smi --showpower 2>/dev/null | grep -oP "[0-9]+(?:\.[0-9]+)?(?=\s*W)" || true)

  [[ "$power_gpu_idx" -gt 0 ]] && \
    info "Total GPU power draw: ${total_power}W across $power_gpu_idx GPU(s)"

  # Clock frequencies — zero MHz is suspicious
  local clk_issues
  clk_issues=$(rocm-smi --showclocks 2>/dev/null | grep -i "Mhz" | grep " 0Mhz\b" | wc -l | tr -d '[:space:]')
  clk_issues=$(_int "$clk_issues")
  if [[ "$clk_issues" -eq 0 ]]; then
    record_pass "clock_frequencies" "All clocks non-zero"
  else
    record_warn "clock_frequencies" "$clk_issues zero-clock reading(s)"
  fi
}
