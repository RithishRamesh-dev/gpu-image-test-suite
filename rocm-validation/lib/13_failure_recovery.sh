#!/usr/bin/env bash
# Module 13 — Failure & Recovery

test_failure_recovery() {
  section "13 · FAILURE & RECOVERY"

  # ── Kernel module unload/reload (non-destructive on running system) ───────
  info "Testing amdgpu module reload (requires no active GPU users)..."
  local active_procs
  active_procs=$(lsof /dev/kfd /dev/dri/renderD* 2>/dev/null | wc -l || echo "0")
  if [[ "$active_procs" -gt 0 ]]; then
    record_skip "module_reload" "$active_procs processes using GPU — skipping reload"
  else
    # Try reload if nothing is using GPU
    if modprobe -r amdgpu 2>/dev/null && modprobe amdgpu 2>/dev/null; then
      sleep 5
      if ls /dev/dri/card* &>/dev/null; then
        record_pass "module_reload" "amdgpu module reloaded, devices reappeared"
      else
        record_fail "module_reload" "GPU devices missing after module reload"
      fi
    else
      record_warn "module_reload" "modprobe reload failed (may be expected in containers)"
    fi
  fi

  # ── Docker daemon restart recovery ────────────────────────────────────────
  if cmd_exists docker && cmd_exists systemctl; then
    info "Restarting Docker daemon..."
    if systemctl restart docker 2>/dev/null; then
      sleep 5
      if docker info &>/dev/null; then
        record_pass "docker_daemon_restart" "Docker daemon restarted successfully"
        # GPU still accessible after restart
        if docker run --rm --device=/dev/kfd --device=/dev/dri \
            rocm/dev-ubuntu-24.04:latest rocminfo 2>/dev/null | grep -qi "GPU"; then
          record_pass "gpu_accessible_post_docker_restart"
        else
          record_warn "gpu_accessible_post_docker_restart" "GPU not visible in container after Docker restart"
        fi
      else
        record_fail "docker_daemon_restart" "Docker daemon failed to restart"
      fi
    else
      record_skip "docker_daemon_restart" "systemctl restart not permitted"
    fi
  else
    record_skip "docker_daemon_restart" "Docker or systemctl not available"
  fi

  # ── Metrics exporter recovery ─────────────────────────────────────────────
  if systemctl list-units 2>/dev/null | grep -q "amd-metrics-exporter"; then
    if systemctl restart amd-metrics-exporter 2>/dev/null; then
      sleep 3
      if systemctl is-active amd-metrics-exporter &>/dev/null; then
        record_pass "metrics_exporter_restart"
      else
        record_fail "metrics_exporter_restart" "amd-metrics-exporter failed to restart"
      fi
    else
      record_skip "metrics_exporter_restart" "Cannot restart amd-metrics-exporter"
    fi
  else
    record_skip "metrics_exporter_restart" "amd-metrics-exporter not installed"
  fi

  # ── OOM recovery simulation (safe: allocate then free) ────────────────────
  info "Testing VRAM allocation and release..."
  local vram_before vram_after
  vram_before=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP "VRAM Used.*:\s*\K[0-9]+" | head -1 || echo "0")

  # Use a Python script to allocate/free GPU memory
  python3 - 2>/dev/null <<'PYEOF' || true
import subprocess, sys
try:
    import torch
    if torch.cuda.is_available() or hasattr(torch, 'hip'):
        x = torch.ones(1024, 1024, 1024, device='cuda')  # ~4 GB
        del x
        torch.cuda.empty_cache()
        print("VRAM alloc/free: OK")
    else:
        print("PyTorch GPU not available")
except Exception as e:
    print(f"Skipped: {e}")
PYEOF

  vram_after=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP "VRAM Used.*:\s*\K[0-9]+" | head -1 || echo "0")
  local leak_bytes=$(( vram_after - vram_before ))
  if [[ "${leak_bytes#-}" -lt 100000000 ]]; then
    record_pass "vram_alloc_free" "VRAM returned to baseline (delta: ${leak_bytes} bytes)"
  else
    record_warn "vram_alloc_free" "VRAM delta post alloc/free: ${leak_bytes} bytes"
  fi

  # ── Post-recovery final GPU check ─────────────────────────────────────────
  local final_gpu_count
  final_gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type.*GPU" || echo "0")
  if [[ "$final_gpu_count" -gt 0 ]]; then
    record_pass "post_recovery_gpu_count" "$final_gpu_count GPU(s) available after all recovery tests"
  else
    record_fail "post_recovery_gpu_count" "No GPUs after recovery tests — critical failure"
  fi
}
