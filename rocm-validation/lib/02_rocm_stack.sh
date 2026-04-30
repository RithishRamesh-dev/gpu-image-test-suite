#!/usr/bin/env bash
# Module 02 — ROCm Stack Integrity

test_rocm_stack() {
  section "02 · ROCM STACK INTEGRITY"

  if cmd_exists rocminfo; then
    record_pass "rocminfo_found" "$(which rocminfo)"
  else
    record_fail "rocminfo_found" "rocminfo not in PATH — ROCm not installed?"
    return
  fi

  # ldd broken libs — sum across all lines
  local broken_libs
  broken_libs=$(ldd "$(which rocminfo)" 2>/dev/null | grep "not found" | wc -l | tr -d '[:space:]' || echo "0")
  broken_libs="${broken_libs//[^0-9]/}"   # strip any whitespace
  if [[ "${broken_libs:-0}" -eq 0 ]]; then
    record_pass "rocminfo_libs_ok" "All shared libraries resolved"
  else
    record_fail "rocminfo_libs_ok" "$broken_libs missing shared libraries"
  fi

  capture "rocminfo" rocminfo

  # Version detection — try multiple sources
  local detected_ver=""
  # 1. /opt/rocm/.info/version
  if [[ -z "$detected_ver" ]]; then
    detected_ver=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | tr -d '[:space:]' || true)
  fi
  # 2. any versioned rocm dir symlink target
  if [[ -z "$detected_ver" ]]; then
    detected_ver=$(readlink -f /opt/rocm 2>/dev/null | grep -oP "rocm-\K[0-9]+\.[0-9]+\.[0-9]+" || true)
  fi
  # 3. amd-smi
  if [[ -z "$detected_ver" ]] && cmd_exists amd-smi; then
    detected_ver=$(amd-smi version 2>/dev/null | grep -oP "(?i)ROCm.*?:\s*\K[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
  fi
  # 4. rocminfo text
  if [[ -z "$detected_ver" ]]; then
    detected_ver=$(rocminfo 2>/dev/null | grep -oP "ROCm Runtime Version:\s*\K[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
  fi

  # Compare major.minor only — patch releases (7.2.0 vs 7.2.1) are acceptable
  local expected_mm detected_mm
  expected_mm=$(echo "$ROCM_EXPECTED_VERSION" | cut -d. -f1,2)
  detected_mm=$(echo "$detected_ver" | cut -d. -f1,2)

  if [[ "$detected_ver" == "$ROCM_EXPECTED_VERSION" ]]; then
    record_pass "rocm_version" "Detected $detected_ver"
  elif [[ -n "$detected_mm" && "$detected_mm" == "$expected_mm" ]]; then
    record_pass "rocm_version" "Detected $detected_ver (patch differs from $ROCM_EXPECTED_VERSION — acceptable)"
  elif [[ -n "$detected_ver" ]]; then
    record_fail "rocm_version" "Detected $detected_ver, expected $ROCM_EXPECTED_VERSION"
  else
    record_warn "rocm_version" "Could not detect ROCm version — check manually"
  fi

  if cmd_exists hipcc; then
    local hip_ver
    hip_ver=$(hipcc --version 2>&1 | grep -oP "HIP version:\s*\K[0-9]+\.[0-9]+\.[0-9]+" || true)
    capture "hipcc_version" hipcc --version
    record_pass "hipcc_found" "HIP version: ${hip_ver:-unknown}"

    local rocm_major rocm_minor hip_major hip_minor
    IFS='.' read -r rocm_major rocm_minor _ <<< "$ROCM_EXPECTED_VERSION"
    IFS='.' read -r hip_major hip_minor _ <<< "${hip_ver:-0.0.0}"
    if [[ "$rocm_major.$rocm_minor" == "$hip_major.$hip_minor" ]]; then
      record_pass "hip_version_aligned" "HIP $hip_ver aligns with ROCm $ROCM_EXPECTED_VERSION"
    else
      record_warn "hip_version_aligned" "HIP $hip_ver may not match ROCm $ROCM_EXPECTED_VERSION"
    fi
  else
    record_fail "hipcc_found" "hipcc not in PATH"
  fi

  if cmd_exists rocm-smi; then
    capture "rocm_smi" rocm-smi
    record_pass "rocm_smi_found"
  else
    record_fail "rocm_smi_found" "rocm-smi not in PATH"
  fi

  if cmd_exists amd-smi; then
    capture "amd_smi_version" amd-smi version
    record_pass "amd_smi_found"
  else
    record_warn "amd_smi_found" "amd-smi not found (optional but recommended)"
  fi

  local agent_count
  agent_count=$(rocminfo 2>/dev/null | grep -c "^  Name:" | tr -d '[:space:]' || echo "0")
  agent_count="${agent_count//[^0-9]/}"
  if [[ "${agent_count:-0}" -gt 0 ]]; then
    record_pass "rocminfo_agents" "$agent_count agent(s) listed"
  else
    record_fail "rocminfo_agents" "No agents found in rocminfo output"
  fi

  # Library checks — search broader paths
  for lib in libhip_hcc.so libamdhip64.so librocblas.so; do
    if find /opt/rocm /usr/lib /usr/local/lib 2>/dev/null -name "${lib}*" -print -quit 2>/dev/null | grep -q .; then
      record_pass "lib_${lib%%.*}" "$lib found"
    else
      record_warn "lib_${lib%%.*}" "$lib not found in standard paths"
    fi
  done

  # No mixed ROCm version packages
  local old_rocm_files
  old_rocm_files=$(dpkg -l 2>/dev/null | grep -i rocm | awk '{print $3}' | \
    grep -oP "^[0-9]+\.[0-9]+" | sort -u | \
    grep -v "^$(echo "$ROCM_EXPECTED_VERSION" | cut -d. -f1,2)$" || true)
  if [[ -z "$old_rocm_files" ]]; then
    record_pass "no_mixed_rocm_versions" "No conflicting ROCm version strings found"
  else
    record_warn "no_mixed_rocm_versions" "Possible version mix: $old_rocm_files"
  fi

  if [[ -d /opt/rocm ]]; then
    local rocm_link_target
    rocm_link_target=$(readlink -f /opt/rocm 2>/dev/null || echo "/opt/rocm")
    record_pass "rocm_dir_exists" "$rocm_link_target"
  else
    record_fail "rocm_dir_exists" "/opt/rocm not found"
  fi
}
