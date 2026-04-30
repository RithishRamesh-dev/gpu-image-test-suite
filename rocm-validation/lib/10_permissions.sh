#!/usr/bin/env bash
# Module 10 — Permissions & Isolation

test_permissions() {
  section "10 · PERMISSIONS & ISOLATION"

  if [[ -c /dev/kfd ]]; then
    local kfd_perms kfd_group
    kfd_perms=$(stat -c '%a' /dev/kfd 2>/dev/null || echo "000")
    kfd_group=$(stat -c '%G' /dev/kfd 2>/dev/null || echo "unknown")
    if [[ "$kfd_perms" == "660" || "$kfd_perms" == "666" ]]; then
      record_pass "kfd_permissions" "Mode $kfd_perms group=$kfd_group"
    else
      record_warn "kfd_permissions" "Mode $kfd_perms (expected 660) group=$kfd_group"
    fi

    local current_groups
    current_groups=$(groups 2>/dev/null || id -Gn 2>/dev/null || echo "")
    if echo "$current_groups" | grep -qE "render|video|kfd"; then
      record_pass "user_in_gpu_group" \
        "User in: $(echo "$current_groups" | tr ' ' '\n' | grep -E 'render|video|kfd' | tr '\n' ' ')"
    else
      record_warn "user_in_gpu_group" "User not in render/video/kfd group"
    fi
  else
    record_fail "kfd_permissions" "/dev/kfd not found"
    record_skip "user_in_gpu_group" "/dev/kfd not found"
  fi

  # /dev/dri
  local dri_files
  dri_files=$(ls /dev/dri/ 2>/dev/null | wc -l | tr -d '[:space:]')
  dri_files=$(_int "$dri_files")
  if [[ "$dri_files" -gt 0 ]]; then
    local dri_perms
    dri_perms=$(stat -c '%a' /dev/dri/renderD128 2>/dev/null || \
                stat -c '%a' /dev/dri/card0 2>/dev/null || echo "unknown")
    if [[ "$dri_perms" == "660" || "$dri_perms" == "666" ]]; then
      record_pass "dri_permissions" "/dev/dri/* mode $dri_perms"
    else
      record_warn "dri_permissions" "/dev/dri/* mode $dri_perms"
    fi
  else
    record_fail "dri_permissions" "/dev/dri empty"
  fi

  # udev rules
  if find /etc/udev/rules.d /lib/udev/rules.d 2>/dev/null \
      -name "*.rules" | xargs grep -l "rocm\|amdgpu\|kfd" 2>/dev/null | grep -q .; then
    record_pass "udev_rules_present" "ROCm udev rules found"
  else
    record_warn "udev_rules_present" "No ROCm udev rules found"
  fi

  # World-writable ROCm binaries
  local world_writable
  world_writable=$(find /opt/rocm/bin 2>/dev/null -perm -o+w | wc -l | tr -d '[:space:]')
  world_writable=$(_int "$world_writable")
  if [[ "$world_writable" -eq 0 ]]; then
    record_pass "rocm_bin_not_world_writable" "No world-writable ROCm binaries"
  else
    record_fail "rocm_bin_not_world_writable" "$world_writable world-writable file(s) in /opt/rocm/bin"
  fi

  # Setuid binaries
  local setuid_count
  setuid_count=$(find /opt/rocm 2>/dev/null -perm /4000 | wc -l | tr -d '[:space:]')
  setuid_count=$(_int "$setuid_count")
  if [[ "$setuid_count" -eq 0 ]]; then
    record_pass "no_setuid_rocm" "No setuid binaries in /opt/rocm"
  else
    record_warn "no_setuid_rocm" "$setuid_count setuid binary/ies in /opt/rocm"
  fi
}
