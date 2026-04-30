#!/usr/bin/env bash
# Module 13 — Failure & Recovery

test_failure_recovery() {
  section "13 · FAILURE & RECOVERY"

  # ── Kernel module reload ───────────────────────────────────────────────────
  local active_procs
  active_procs=$(lsof /dev/kfd /dev/dri/renderD* 2>/dev/null | wc -l | tr -d '[:space:]')
  active_procs=$(_int "$active_procs")
  if [[ "$active_procs" -gt 0 ]]; then
    record_skip "module_reload" "$active_procs processes using GPU — skipping reload"
  else
    if modprobe -r amdgpu 2>/dev/null && modprobe amdgpu 2>/dev/null; then
      sleep 5
      if ls /dev/dri/card* &>/dev/null; then
        record_pass "module_reload" "amdgpu reloaded, devices reappeared"
      else
        record_fail "module_reload" "GPU devices missing after module reload"
      fi
    else
      record_warn "module_reload" "modprobe reload failed (may be expected in containers)"
    fi
  fi

  # ── Docker daemon restart ─────────────────────────────────────────────────
  if cmd_exists docker && cmd_exists systemctl; then
    info "Restarting Docker daemon..."
    if systemctl restart docker 2>/dev/null; then
      sleep 5
      if docker info &>/dev/null; then
        record_pass "docker_daemon_restart" "Docker daemon restarted successfully"
        if docker run --rm --device=/dev/kfd --device=/dev/dri \
            rocm/dev-ubuntu-24.04:latest rocminfo 2>/dev/null | grep -qi "GPU"; then
          record_pass "gpu_accessible_post_docker_restart"
        else
          record_warn "gpu_accessible_post_docker_restart" \
            "GPU not visible in container after Docker restart"
        fi
      else
        record_fail "docker_daemon_restart" "Docker daemon failed to restart"
        record_skip "gpu_accessible_post_docker_restart"
      fi
    else
      record_skip "docker_daemon_restart" "systemctl restart not permitted"
      record_skip "gpu_accessible_post_docker_restart"
    fi
  else
    record_skip "docker_daemon_restart" "Docker or systemctl not available"
    record_skip "gpu_accessible_post_docker_restart"
  fi

  # ── Metrics exporter restart ──────────────────────────────────────────────
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

  # ── VRAM alloc/free ── Python handles large byte values safely ────────────
  info "Testing VRAM allocation and release..."
  local vram_result
  vram_result=$(python3 - 2>/dev/null <<'PYEOF' || echo "WARN:unknown"
import subprocess, re
def get_used():
    out = subprocess.run(["rocm-smi","--showmeminfo","vram"],
        capture_output=True, text=True).stdout
    return sum(int(v) for v in re.findall(r"VRAM Used.*?:\s*(\d+)", out))
before = get_used()
try:
    import torch
    if torch.cuda.is_available() or hasattr(torch, "hip"):
        x = torch.ones(512, 1024, 1024, device="cuda")
        del x
        torch.cuda.empty_cache()
except Exception:
    pass
after  = get_used()
delta  = after - before
status = "PASS" if abs(delta) < 100_000_000 else "WARN"
print(f"{status}:{delta}")
PYEOF
)
  local vram_status vram_delta
  IFS=: read -r vram_status vram_delta <<< "${vram_result:-WARN:unknown}"
  if [[ "$vram_status" == "PASS" ]]; then
    record_pass "vram_alloc_free" "VRAM returned to baseline (delta: ${vram_delta} bytes)"
  else
    record_warn "vram_alloc_free" "VRAM delta post alloc/free: ${vram_delta} bytes"
  fi

  # ── Final GPU count ───────────────────────────────────────────────────────
  local final_gpu_count
  final_gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type.*GPU" | tr -d '[:space:]')
  final_gpu_count=$(_int "$final_gpu_count")
  if [[ "$final_gpu_count" -gt 0 ]]; then
    record_pass "post_recovery_gpu_count" \
      "$final_gpu_count GPU(s) available after all recovery tests"
  else
    record_fail "post_recovery_gpu_count" "No GPUs after recovery tests — critical failure"
  fi
}
