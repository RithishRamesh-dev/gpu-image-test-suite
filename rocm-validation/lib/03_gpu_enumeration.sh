#!/usr/bin/env bash
# Module 03 — GPU Enumeration & Topology

test_gpu_enumeration() {
  section "03 · GPU ENUMERATION & TOPOLOGY"

  capture "rocm_smi_showhw"    bash -c "rocm-smi --showhw    2>/dev/null || true"
  capture "rocm_smi_showtopo"  bash -c "rocm-smi --showtopo  2>/dev/null || true"
  capture "rocm_smi_showmeminfo" bash -c "rocm-smi --showmeminfo vram 2>/dev/null || true"
  capture "lspci_amd"          bash -c "lspci 2>/dev/null | grep -iE 'amd|radeon|advanced micro' || true"

  # GPU count — sum all lines (safe across multi-GPU output)
  local gpu_count
  gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type:.*GPU" | awk '{s+=$1} END{print s+0}')
  if [[ "$gpu_count" -gt 0 ]]; then
    record_pass "gpu_count" "$gpu_count GPU(s) detected via rocminfo"
  else
    record_fail "gpu_count" "No GPUs found via rocminfo"
    return
  fi

  if [[ "$TP_SIZE" -gt 0 && "$gpu_count" -ne "$TP_SIZE" ]]; then
    record_warn "gpu_count_matches_tp" "Found $gpu_count GPUs, TP_SIZE=$TP_SIZE"
  else
    record_pass "gpu_count_matches_tp" "$gpu_count GPU(s) matches TP_SIZE=$TP_SIZE"
  fi

  # UNKNOWN devices
  local unknown_count
  unknown_count=$(rocm-smi --showhw 2>/dev/null | grep -ci "UNKNOWN" | awk '{s+=$1} END{print s+0}')
  if [[ "$unknown_count" -eq 0 ]]; then
    record_pass "no_unknown_devices" "No UNKNOWN GPU entries"
  else
    record_fail "no_unknown_devices" "$unknown_count UNKNOWN device(s) — driver/firmware issue"
  fi

  # PCIe width
  if cmd_exists lspci; then
    local narrow_links
    narrow_links=$(lspci -vvv 2>/dev/null | grep -A30 -E "AMD|Radeon" | \
                   grep "LnkSta:" | grep -cv "Width x16" || echo "0")
    narrow_links="${narrow_links//[^0-9]/}"
    if [[ "${narrow_links:-0}" -eq 0 ]]; then
      record_pass "pcie_link_width" "All GPU PCIe links appear full-width (x16)"
    else
      record_warn "pcie_link_width" "$narrow_links GPU(s) on non-x16 PCIe link"
    fi
  else
    record_skip "pcie_link_width" "lspci not available"
  fi

  # NUMA locality
  local numa_info
  numa_info=$(rocm-smi --showtopo 2>/dev/null | grep -i "numa" || true)
  if [[ -n "$numa_info" ]]; then
    record_pass "numa_affinity" "NUMA info present in topology"
  else
    record_warn "numa_affinity" "No NUMA info in rocm-smi --showtopo"
  fi

  # Per-GPU VRAM — use Python to avoid bash integer overflow on large byte values
  while IFS=: read -r idx status detail; do
    if [[ "$status" == "PASS" ]]; then
      record_pass "vram_gpu${idx}" "$detail"
    else
      record_fail "vram_gpu${idx}" "$detail"
    fi
  done < <(python3 2>/dev/null <<'PYEOF'
import subprocess, re
out = subprocess.run(["rocm-smi","--showmeminfo","vram"],
    capture_output=True, text=True).stdout
totals = re.findall(r"VRAM Total Memory.*?:\s*(\d+)", out)
for i, t in enumerate(totals):
    gb = int(t) // (1024**3)
    status = "PASS" if gb > 0 else "FAIL"
    detail = f"{gb} GB VRAM" if gb > 0 else "VRAM appears 0 GB"
    print(f"{i}:{status}:{detail}")
PYEOF
)

  # P2P access
  local p2p_capable
  p2p_capable=$(rocm-smi --showtopo 2>/dev/null | grep -ci "P2P\|peer" | awk '{s+=$1} END{print s+0}')
  if [[ "$p2p_capable" -gt 0 && "$gpu_count" -gt 1 ]]; then
    record_pass "p2p_access" "P2P access info present"
  elif [[ "$gpu_count" -eq 1 ]]; then
    record_skip "p2p_access" "Single GPU — P2P N/A"
  else
    record_warn "p2p_access" "P2P info not found — may impact NCCL/RCCL performance"
  fi

  # Baseline temperature — use awk to find max, avoiding bash float comparison
  local max_temp
  max_temp=$(rocm-smi --showtemp 2>/dev/null | \
             grep -oP "[0-9]+(?:\.[0-9]+)?(?=c)" | \
             awk 'BEGIN{m=0} {v=int($1); if(v>m) m=v} END{print m}')
  max_temp="${max_temp:-0}"
  if [[ "$max_temp" -eq 0 ]]; then
    record_warn "baseline_temp" "Could not read GPU temperatures"
  elif [[ "$max_temp" -lt 60 ]]; then
    record_pass "baseline_temp" "Max GPU temp: ${max_temp}°C (healthy)"
  else
    record_warn "baseline_temp" "GPU already at ${max_temp}°C — check cooling"
  fi
}
