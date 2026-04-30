#!/usr/bin/env bash
# Module 05 — Docker + GPU Container Runtime

test_docker_runtime() {
  section "05 · DOCKER + GPU RUNTIME"

  if ! cmd_exists docker; then
    record_fail "docker_installed" "Docker not found"
    for t in docker_daemon docker_hello_world docker_gpu_container docker_rocm_version_in_container \
              docker_device_permissions docker_ipc_mode; do
      record_skip "$t" "Docker not installed"
    done
    return
  fi
  record_pass "docker_installed" "$(docker --version)"

  # Daemon responsive
  if docker info &>/dev/null; then
    record_pass "docker_daemon" "Docker daemon responding"
  else
    record_fail "docker_daemon" "docker info failed"
    return
  fi

  # hello-world baseline
  if docker run --rm hello-world &>/dev/null; then
    record_pass "docker_hello_world"
  else
    record_fail "docker_hello_world" "hello-world container failed"
  fi

  # GPU device passthrough
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

  # ROCm version inside container matches host
  local container_rocm_ver
  container_rocm_ver=$(docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    rocm/dev-ubuntu-24.04:latest \
    bash -c "cat /opt/rocm/.info/version 2>/dev/null || rocminfo 2>/dev/null | grep -oP 'ROCm Runtime Version:\s*\K[0-9.]+' | head -1" 2>/dev/null || true)

  if [[ "$container_rocm_ver" == "$ROCM_EXPECTED_VERSION" ]]; then
    record_pass "docker_rocm_version_in_container" "Container ROCm = $container_rocm_ver"
  else
    record_warn "docker_rocm_version_in_container" "Container ROCm = '${container_rocm_ver:-unknown}', host expected $ROCM_EXPECTED_VERSION"
  fi

  # /dev/kfd readable inside container
  if docker run --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      rocm/dev-ubuntu-24.04:latest \
      test -r /dev/kfd 2>/dev/null; then
    record_pass "docker_device_permissions" "/dev/kfd readable in container"
  else
    record_fail "docker_device_permissions" "/dev/kfd not readable in container — check group/udev rules"
  fi

  # IPC host mode (needed for multi-GPU NCCL in containers)
  local ipc_test
  ipc_test=$(docker run --rm --ipc=host ubuntu:22.04 echo "ipc_ok" 2>/dev/null || echo "failed")
  if [[ "$ipc_test" == "ipc_ok" ]]; then
    record_pass "docker_ipc_mode" "--ipc=host works"
  else
    record_warn "docker_ipc_mode" "--ipc=host may be restricted"
  fi

  # Docker storage driver
  local storage_driver
  storage_driver=$(docker info --format '{{.Driver}}' 2>/dev/null || echo "unknown")
  if [[ "$storage_driver" == "overlay2" ]]; then
    record_pass "docker_storage_driver" "overlay2"
  else
    record_warn "docker_storage_driver" "Driver is '$storage_driver' (overlay2 recommended)"
  fi

  # Available disk for pulling large images
  local free_gb
  free_gb=$(df /var/lib/docker 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || echo "0")
  if [[ "$free_gb" -ge 200 ]]; then
    record_pass "docker_disk_space" "${free_gb} GB free for Docker images"
  elif [[ "$free_gb" -ge 50 ]]; then
    record_warn "docker_disk_space" "Only ${free_gb} GB free — large models may not fit"
  else
    record_fail "docker_disk_space" "Only ${free_gb} GB free — insufficient for LLM images"
  fi
}
