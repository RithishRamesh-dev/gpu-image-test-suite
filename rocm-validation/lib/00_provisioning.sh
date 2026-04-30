#!/usr/bin/env bash
# Module 00 — Droplet Provisioning

test_provisioning() {
  section "00 · PROVISIONING"

  # cloud-init completion
  if cmd_exists cloud-init; then
    local ci_status
    ci_status=$(cloud-init status 2>/dev/null | awk '{print $NF}' || echo "unknown")
    if [[ "$ci_status" == "done" ]]; then
      record_pass "cloud_init_status" "status=done"
    else
      record_fail "cloud_init_status" "status=$ci_status (expected done)"
    fi

    # Check for cloud-init errors in journal
    local ci_errors
    ci_errors=$(journalctl -u cloud-init --no-pager 2>/dev/null | grep -cE "ERROR|CRITICAL" || echo "0")
    if [[ "$ci_errors" -eq 0 ]]; then
      record_pass "cloud_init_errors" "0 errors in journal"
    else
      record_fail "cloud_init_errors" "$ci_errors error(s) found in cloud-init journal"
    fi
  else
    record_skip "cloud_init_status" "cloud-init not installed"
    record_skip "cloud_init_errors" "cloud-init not installed"
  fi

  # GPU devices present at boot time
  if ls /dev/dri/card* &>/dev/null; then
    local gpu_count
    gpu_count=$(ls /dev/dri/card* 2>/dev/null | wc -l)
    record_pass "gpu_devices_at_boot" "$gpu_count DRI device(s) present"
  else
    record_fail "gpu_devices_at_boot" "No /dev/dri/card* devices found"
  fi

  # KFD device (required by ROCm)
  if [[ -c /dev/kfd ]]; then
    record_pass "kfd_device_present" "/dev/kfd exists"
  else
    record_fail "kfd_device_present" "/dev/kfd missing — ROCm will not function"
  fi

  # Uptime sanity (catch infinite reboot loops)
  local uptime_sec
  uptime_sec=$(awk '{print int($1)}' /proc/uptime)
  if [[ "$uptime_sec" -gt 30 ]]; then
    record_pass "uptime_sanity" "${uptime_sec}s uptime"
  else
    record_warn "uptime_sanity" "Very short uptime (${uptime_sec}s) — possible reboot loop"
  fi
}
