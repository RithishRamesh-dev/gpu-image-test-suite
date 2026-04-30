#!/usr/bin/env bash
# Module 04 — Observability Stack

test_observability() {
  section "04 · OBSERVABILITY STACK"

  # ── AMD Metrics Exporter ─────────────────────────────────────────────────
  if systemctl list-units --type=service 2>/dev/null | grep -q "amd-metrics-exporter"; then
    local exporter_status
    exporter_status=$(systemctl is-active amd-metrics-exporter 2>/dev/null || echo "inactive")
    if [[ "$exporter_status" == "active" ]]; then
      record_pass "amd_metrics_exporter_service" "Service is active"
    else
      record_fail "amd_metrics_exporter_service" "Service is $exporter_status"
    fi

    # Metrics endpoint responds
    if curl -sf http://127.0.0.1:5000/metrics &>/dev/null; then
      record_pass "metrics_endpoint_reachable" "http://127.0.0.1:5000/metrics responds"

      local metrics_body
      metrics_body=$(curl -s http://127.0.0.1:5000/metrics 2>/dev/null)

      # GPU metric families present
      for field in gpu temperature power memory utilization; do
        if echo "$metrics_body" | grep -qi "$field"; then
          record_pass "metrics_field_${field}"
        else
          record_fail "metrics_field_${field}" "Field '$field' missing from /metrics"
        fi
      done

      # Capture for diff-ing
      echo "$metrics_body" > "$RESULTS_DIR/metrics_snapshot.txt"

      # Cross-check GPU util vs rocm-smi
      local exporter_gpu_count
      exporter_gpu_count=$(echo "$metrics_body" | grep -oP 'gpu_id="\K[0-9]+' | sort -un | wc -l || echo "0")
      local smi_gpu_count
      smi_gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo "0")
      if [[ "$exporter_gpu_count" -eq "$smi_gpu_count" ]]; then
        record_pass "metrics_gpu_count_match" "Exporter reports $exporter_gpu_count GPU(s) = rocm-smi"
      else
        record_warn "metrics_gpu_count_match" "Exporter: $exporter_gpu_count, rocm-smi: $smi_gpu_count"
      fi

    else
      record_fail "metrics_endpoint_reachable" "Cannot reach http://127.0.0.1:5000/metrics"
    fi
  else
    record_skip "amd_metrics_exporter_service" "amd-metrics-exporter not installed"
    record_skip "metrics_endpoint_reachable"   "amd-metrics-exporter not installed"
    for field in gpu temperature power memory utilization; do
      record_skip "metrics_field_${field}"
    done
    record_skip "metrics_gpu_count_match"
  fi

  # ── DigitalOcean Agent ────────────────────────────────────────────────────
  if systemctl list-units --type=service 2>/dev/null | grep -q "do-agent"; then
    local do_status
    do_status=$(systemctl is-active do-agent 2>/dev/null || echo "inactive")
    if [[ "$do_status" == "active" ]]; then
      record_pass "do_agent_service" "do-agent is active"
    else
      record_fail "do_agent_service" "do-agent is $do_status"
    fi
  else
    record_skip "do_agent_service" "do-agent not installed"
  fi

  # ── Prometheus node exporter (common supplementary tool) ─────────────────
  if curl -sf http://127.0.0.1:9100/metrics &>/dev/null; then
    record_pass "node_exporter" "Prometheus node_exporter running"
  else
    record_skip "node_exporter" "Not running (optional)"
  fi

  # ── dcgm / alternative monitoring ────────────────────────────────────────
  if cmd_exists dcgmi; then
    record_pass "dcgm_available" "DCGM found (AMD DCGM)"
  else
    record_skip "dcgm_available" "DCGM not installed (optional)"
  fi

  # ── Syslog GPU-related entries ────────────────────────────────────────────
  local syslog_gpu_errors
  syslog_gpu_errors=$(journalctl -p err --since="1 hour ago" 2>/dev/null | grep -ciE "amdgpu|kfd|rocm" || echo "0")
  if [[ "$syslog_gpu_errors" -eq 0 ]]; then
    record_pass "no_syslog_gpu_errors" "No recent GPU-related errors in journal"
  else
    record_warn "no_syslog_gpu_errors" "$syslog_gpu_errors GPU-related error(s) in recent journal"
  fi
}
