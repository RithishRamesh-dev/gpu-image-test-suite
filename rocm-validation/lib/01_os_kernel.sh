#!/usr/bin/env bash
# Module 01 — OS, Kernel, Driver Baseline

test_os_kernel() {
  section "01 · OS + KERNEL + DRIVER BASELINE"

  # OS release capture
  capture "os_release" cat /etc/os-release
  local os_id os_ver
  os_id=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}" || echo "unknown")
  os_ver=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-unknown}" || echo "unknown")
  record_pass "os_detected" "$os_id $os_ver"

  if [[ "$os_id" == "ubuntu" && ("$os_ver" == "22.04" || "$os_ver" == "24.04") ]]; then
    record_pass "os_supported" "Ubuntu $os_ver is supported"
  else
    record_warn "os_supported" "$os_id $os_ver — verify ROCm 7.2 support"
  fi

  # Kernel version
  local kernel
  kernel=$(uname -r 2>/dev/null || echo "unknown")
  capture "kernel_version" uname -r
  record_pass "kernel_detected" "$kernel"

  local kmaj kmin
  kmaj=$(echo "$kernel" | cut -d. -f1)
  kmin=$(echo "$kernel" | cut -d. -f2)
  if [[ "$kmaj" =~ ^[0-9]+$ && "$kmin" =~ ^[0-9]+$ ]]; then
    if [[ "$kmaj" -gt 5 || ("$kmaj" -eq 5 && "$kmin" -ge 15) ]]; then
      record_pass "kernel_version_ok" "$kernel ≥ 5.15"
    else
      record_fail "kernel_version_ok" "$kernel is below minimum 5.15"
    fi
  else
    record_warn "kernel_version_ok" "Could not parse kernel version: $kernel"
  fi

  # amdgpu module loaded
  if lsmod 2>/dev/null | grep -q "^amdgpu"; then
    record_pass "amdgpu_module_loaded"
  else
    record_fail "amdgpu_module_loaded" "amdgpu not in lsmod"
  fi

  # dmesg checks — grep returning 1 (no match) must not kill the script
  capture "dmesg_amdgpu" bash -c "dmesg -T 2>/dev/null | grep -iE 'amdgpu|kfd|iommu' | tail -n 100 || true"

  local gpu_resets
  gpu_resets=$(dmesg 2>/dev/null | grep -ciE "gpu reset|amdgpu.*reset" || echo "0")
  if [[ "${gpu_resets:-0}" -eq 0 ]]; then
    record_pass "no_gpu_resets" "0 GPU reset events in dmesg"
  else
    record_fail "no_gpu_resets" "$gpu_resets GPU reset event(s) detected"
  fi

  local iommu_faults
  iommu_faults=$(dmesg 2>/dev/null | grep -ciE "iommu.*fault|dmar.*fault" || echo "0")
  if [[ "${iommu_faults:-0}" -eq 0 ]]; then
    record_pass "no_iommu_faults" "0 IOMMU faults"
  else
    record_warn "no_iommu_faults" "$iommu_faults IOMMU fault(s) — may be benign, investigate"
  fi

  local vram_errors
  vram_errors=$(dmesg 2>/dev/null | grep -ciE "vram.*error|ecc.*error|uncorrected" || echo "0")
  if [[ "${vram_errors:-0}" -eq 0 ]]; then
    record_pass "no_vram_errors" "0 VRAM/ECC errors"
  else
    record_fail "no_vram_errors" "$vram_errors VRAM/ECC error(s) — hardware issue"
  fi

  local oom_events
  oom_events=$(dmesg 2>/dev/null | grep -ci "oom.*kill\|out of memory" || echo "0")
  if [[ "${oom_events:-0}" -eq 0 ]]; then
    record_pass "no_oom_events"
  else
    record_warn "no_oom_events" "$oom_events OOM event(s) in dmesg"
  fi

  # SELinux — getenforce may not exist; treat absence as "not enforcing"
  local selinux_mode
  selinux_mode=$(getenforce 2>/dev/null || echo "Disabled")
  if [[ "$selinux_mode" == "Enforcing" ]]; then
    record_warn "selinux_mode" "SELinux=Enforcing — may block GPU device access"
  else
    record_pass "selinux_mode" "SELinux: $selinux_mode"
  fi

  # Huge pages
  local hugepages
  hugepages=$(grep -E "^HugePages_Total" /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
  if [[ "${hugepages:-0}" -gt 0 ]]; then
    record_pass "hugepages_available" "$hugepages huge pages configured"
  else
    record_warn "hugepages_available" "No huge pages — consider enabling for inference perf"
  fi
}
