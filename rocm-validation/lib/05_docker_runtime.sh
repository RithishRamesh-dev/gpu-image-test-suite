#!/usr/bin/env bash
# Module 05 — Docker + GPU Container Runtime
# Auto-installs Docker Engine if not present before running any tests.

# =============================================================================
# _install_docker
#   Installs Docker Engine using the official get.docker.com convenience script.
#   Returns 0 on success, 1 on failure.
# =============================================================================
_install_docker() {
  info "Docker not found — installing Docker Engine via get.docker.com..."

  # Require curl
  if ! cmd_exists curl; then
    info "curl not found, installing..."
    apt-get update -qq 2>&1 | tee -a "$LOG_FILE" || true
    apt-get install -y -qq curl 2>&1 | tee -a "$LOG_FILE" || true
  fi

  if ! cmd_exists curl; then
    warn "Could not install curl — cannot proceed with Docker installation"
    return 1
  fi

  local installer="/tmp/get-docker-$$.sh"

  info "Downloading Docker installer..."
  if ! curl -fsSL https://get.docker.com -o "$installer" 2>&1 | tee -a "$LOG_FILE"; then
    warn "Failed to download Docker installer from https://get.docker.com"
    rm -f "$installer"
    return 1
  fi

  info "Running Docker installer (this may take a minute)..."
  if ! sh "$installer" 2>&1 | tee -a "$LOG_FILE"; then
    warn "Docker installer script exited with an error"
    rm -f "$installer"
    return 1
  fi
  rm -f "$installer"

  # Start and enable Docker
  info "Starting Docker service..."
  systemctl start docker 2>&1 | tee -a "$LOG_FILE" || true

  info "Enabling Docker to start on boot..."
  systemctl enable docker 2>&1 | tee -a "$LOG_FILE" || true

  # Wait up to 15s for daemon to be ready
  local attempts=0
  while [[ $attempts -lt 15 ]]; do
    if docker info &>/dev/null; then
      info "Docker daemon is ready"
      return 0
    fi
    sleep 1
    (( attempts++ )) || true
  done

  warn "Docker installed but daemon did not respond within 15s"
  return 1
}

# =============================================================================
# test_docker_runtime
# =============================================================================
test_docker_runtime() {
  section "05 · DOCKER + GPU RUNTIME"

  # ── Installation ─────────────────────────────────────────────────────────
  if ! cmd_exists docker; then
    if _install_docker; then
      record_pass "docker_install" "Docker Engine installed successfully"
    else
      record_fail "docker_install" "Docker installation failed — see log for details"
      for t in docker_daemon docker_service_active docker_service_enabled \
                docker_hello_world docker_gpu_container \
                docker_rocm_version_in_container docker_device_permissions \
                docker_ipc_mode docker_storage_driver docker_disk_space; do
        record_skip "$t" "Docker installation failed"
      done
      return
    fi
  else
    record_pass "docker_install" "Docker already installed: $(docker --version 2>/dev/null)"
  fi

  # ── Daemon health ─────────────────────────────────────────────────────────
  capture "docker_info" bash -c "docker info 2>/dev/null || true"

  if docker info &>/dev/null; then
    record_pass "docker_daemon" "Daemon responding — $(docker --version 2>/dev/null)"
  else
    record_fail "docker_daemon" "docker info failed — daemon not running"
    systemctl start docker 2>/dev/null || true
    sleep 3
    if ! docker info &>/dev/null; then
      for t in docker_service_active docker_service_enabled docker_hello_world \
                docker_gpu_container docker_rocm_version_in_container \
                docker_device_permissions docker_ipc_mode \
                docker_storage_driver docker_disk_space; do
        record_skip "$t" "Docker daemon not running"
      done
      return
    fi
    record_pass "docker_daemon_recovered" "Daemon started after retry"
  fi

  # ── Systemd status ────────────────────────────────────────────────────────
  capture "docker_systemd_status" bash -c "systemctl status docker 2>/dev/null || true"

  local docker_active
  docker_active=$(systemctl is-active docker 2>/dev/null || echo "unknown")
  if [[ "$docker_active" == "active" ]]; then
    record_pass "docker_service_active" "systemd unit: active"
  else
    record_warn "docker_service_active" "systemd unit: $docker_active"
  fi

  local docker_enabled
  docker_enabled=$(systemctl is-enabled docker 2>/dev/null || echo "unknown")
  if [[ "$docker_enabled" == "enabled" ]]; then
    record_pass "docker_service_enabled" "Docker enabled on boot"
  else
    record_warn "docker_service_enabled" "Docker not enabled on boot ($docker_enabled)"
  fi

  # ── hello-world baseline ──────────────────────────────────────────────────
  if docker run --rm hello-world &>/dev/null; then
    record_pass "docker_hello_world" "Basic container execution works"
  else
    record_fail "docker_hello_world" "hello-world container failed"
  fi

  # ── GPU device passthrough ────────────────────────────────────────────────
  local gpu_test_out
  gpu_test_out=$(docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    rocm/dev-ubuntu-24.04:latest \
    rocminfo 2>&1 | head -60 || true)

  if echo "$gpu_test_out" | grep -qi "Device Type.*GPU"; then
    record_pass "docker_gpu_container" "GPU visible inside container"
  else
    record_fail "docker_gpu_container" "GPU not detected inside container"
  fi

  # ── ROCm version inside container ─────────────────────────────────────────
  local container_rocm_ver
  container_rocm_ver=$(docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    rocm/dev-ubuntu-24.04:latest \
    bash -c "cat /opt/rocm/.info/version 2>/dev/null || \
             rocminfo 2>/dev/null | grep -oP 'ROCm Runtime Version:\s*\K[0-9.]+' | head -1" \
    2>/dev/null | tr -d '[:space:]' || true)

  local expected_mm detected_mm
  expected_mm=$(echo "$ROCM_EXPECTED_VERSION" | cut -d. -f1,2)
  detected_mm=$(echo "$container_rocm_ver"    | cut -d. -f1,2)

  if [[ "$container_rocm_ver" == "$ROCM_EXPECTED_VERSION" ]]; then
    record_pass "docker_rocm_version_in_container" "Container ROCm = $container_rocm_ver"
  elif [[ -n "$detected_mm" && "$detected_mm" == "$expected_mm" ]]; then
    record_pass "docker_rocm_version_in_container" \
      "Container ROCm $container_rocm_ver (patch differs — acceptable)"
  else
    record_warn "docker_rocm_version_in_container" \
      "Container ROCm '${container_rocm_ver:-unknown}', host expected $ROCM_EXPECTED_VERSION"
  fi

  # ── /dev/kfd readable inside container ───────────────────────────────────
  if docker run --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      rocm/dev-ubuntu-24.04:latest \
      test -r /dev/kfd 2>/dev/null; then
    record_pass "docker_device_permissions" "/dev/kfd readable in container"
  else
    record_fail "docker_device_permissions" \
      "/dev/kfd not readable in container — check group/udev rules"
  fi

  # ── IPC host mode ─────────────────────────────────────────────────────────
  local ipc_test
  ipc_test=$(docker run --rm --ipc=host ubuntu:22.04 echo "ipc_ok" 2>/dev/null || echo "failed")
  if [[ "$ipc_test" == "ipc_ok" ]]; then
    record_pass "docker_ipc_mode" "--ipc=host works (required for multi-GPU NCCL)"
  else
    record_warn "docker_ipc_mode" "--ipc=host may be restricted"
  fi

  # ── Storage driver ────────────────────────────────────────────────────────
  local storage_driver
  storage_driver=$(docker info --format '{{.Driver}}' 2>/dev/null || echo "unknown")
  if [[ "$storage_driver" == "overlay2" ]]; then
    record_pass "docker_storage_driver" "overlay2"
  else
    record_warn "docker_storage_driver" "Driver is '$storage_driver' (overlay2 recommended)"
  fi

  # ── Disk space ────────────────────────────────────────────────────────────
  local free_gb
  free_gb=$(df /var/lib/docker 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || echo "0")
  free_gb=$(_int "$free_gb")
  if [[ "$free_gb" -ge 200 ]]; then
    record_pass "docker_disk_space" "${free_gb} GB free for Docker images"
  elif [[ "$free_gb" -ge 50 ]]; then
    record_warn "docker_disk_space" "Only ${free_gb} GB free — large models may not fit"
  else
    record_fail "docker_disk_space" "Only ${free_gb} GB free — insufficient for LLM images"
  fi
}
