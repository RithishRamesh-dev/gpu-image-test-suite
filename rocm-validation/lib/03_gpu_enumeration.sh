#!/usr/bin/env bash
# Module 03 — GPU Enumeration & Topology

test_gpu_enumeration() {
  section "03 · GPU ENUMERATION & TOPOLOGY"

  capture "rocm_smi_showhw" bash -c "rocm-smi --showhw 2>/dev/null || true"
  capture "rocm_smi_showtopo" bash -c "rocm-smi --showtopo 2>/dev/null || true"
  capture "rocm_smi_showmeminfo" bash -c "rocm-smi --showmeminfo vram 2>/dev/null || true"
  capture "lspci_amd" bash -c "lspci 2>/dev/null | grep -i 'amd\|radeon\|advanced micro' || true"

  # Count GPUs via rocminfo
  local gpu_count
  gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type:.*GPU" || echo "0")
  if [[ "$gpu_count" -gt 0 ]]; then
    record_pass "gpu_count" "$gpu_count GPU(s) detected via rocminfo"
  else
    record_fail "gpu_count" "No GPUs found via rocminfo"
    return
  fi

  # Verify count matches TP_SIZE expectation (if set to a real number)
  if [[ "$TP_SIZE" -gt 0 && "$gpu_count" -ne "$TP_SIZE" ]]; then
    record_warn "gpu_count_matches_tp" "Found $gpu_count GPUs, TP_SIZE=$TP_SIZE"
  else
    record_pass "gpu_count_matches_tp" "$gpu_count GPU(s) matches TP_SIZE=$TP_SIZE"
  fi

  # No UNKNOWN devices
  local unknown_count
  unknown_count=$(rocm-smi --showhw 2>/dev/null | grep -ci "UNKNOWN" || echo "0")
  if [[ "$unknown_count" -eq 0 ]]; then
    record_pass "no_unknown_devices" "No UNKNOWN GPU entries"
  else
    record_fail "no_unknown_devices" "$unknown_count UNKNOWN device(s) — driver/firmware issue"
  fi

  # PCIe width — check for unexpected x8/x4 links (expect x16)
  if cmd_exists lspci; then
    local narrow_links
    narrow_links=$(lspci -vvv 2>/dev/null | grep -A30 "AMD\|Radeon" | grep -cE "LnkSta:.*Width x[1248][^6]" || echo "0")
    if [[ "$narrow_links" -eq 0 ]]; then
      record_pass "pcie_link_width" "All GPU PCIe links appear full-width"
    else
      record_warn "pcie_link_width" "$narrow_links GPU(s) on narrow PCIe link (check topology)"
    fi
  else
    record_skip "pcie_link_width" "lspci not available"
  fi

  # NUMA locality check — all GPUs should have NUMA affinity
  local numa_info
  numa_info=$(rocm-smi --showtopo 2>/dev/null | grep -i "numa" || true)
  if [[ -n "$numa_info" ]]; then
    record_pass "numa_affinity" "NUMA info present in topology"
  else
    record_warn "numa_affinity" "No NUMA info in rocm-smi --showtopo"
  fi

  # VRAM per GPU — detect underreported VRAM (common firmware issue)
  while IFS= read -r line; do
    local gpu_id vram_bytes vram_gb
    gpu_id=$(echo "$line" | grep -oP "GPU\[?\K[0-9]+" || true)
    vram_bytes=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 "GPU\[$gpu_id\]" | grep -oP "VRAM Total Memory.*:\s*\K[0-9]+" || true)
    if [[ -n "$vram_bytes" ]]; then
      vram_gb=$(( vram_bytes / 1024 / 1024 / 1024 ))
      if [[ "$vram_gb" -gt 0 ]]; then
        record_pass "vram_gpu${gpu_id:-?}" "${vram_gb} GB VRAM"
      else
        record_fail "vram_gpu${gpu_id:-?}" "VRAM appears 0 GB — reporting issue"
      fi
    fi
  done < <(rocm-smi --showid 2>/dev/null | grep "GPU\[" || true)

  # Peer-to-peer access matrix
  local p2p_capable
  p2p_capable=$(rocm-smi --showtopo 2>/dev/null | grep -ci "P2P\|peer" || echo "0")
  if [[ "$p2p_capable" -gt 0 && "$gpu_count" -gt 1 ]]; then
    record_pass "p2p_access" "P2P access info present"
  elif [[ "$gpu_count" -eq 1 ]]; then
    record_skip "p2p_access" "Single GPU — P2P N/A"
  else
    record_warn "p2p_access" "P2P info not found — may impact NCCL/RCCL performance"
  fi

  # GPU temperature at enumeration (should not be pre-throttled)
  local max_temp=0
  while read -r temp_val; do
    temp_int="${temp_val%%.*}"
    if [[ "$temp_int" =~ ^[0-9]+$ && "$temp_int" -gt "$max_temp" ]]; then
      max_temp="$temp_int"
    fi
  done < <(rocm-smi --showtemp 2>/dev/null | grep -oP "[0-9]+(\.[0-9]+)?(?=c)" || true)

  if [[ "$max_temp" -eq 0 ]]; then
    record_warn "baseline_temp" "Could not read GPU temperatures"
  elif [[ "$max_temp" -lt 60 ]]; then
    record_pass "baseline_temp" "Max GPU temp at boot: ${max_temp}°C (healthy)"
  else
    record_warn "baseline_temp" "GPU already at ${max_temp}°C at boot — check cooling"
  fi
}
