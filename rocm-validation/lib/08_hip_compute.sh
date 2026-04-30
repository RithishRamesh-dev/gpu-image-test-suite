#!/usr/bin/env bash
# Module 08 — HIP / Compute Validation

test_hip_compute() {
  section "08 · HIP COMPUTE VALIDATION"

  local samples_base="/opt/rocm/samples/bin"

  if [[ ! -d "$samples_base" ]]; then
    # Try building from source
    if [[ -d /opt/rocm/share/hip/samples ]]; then
      info "Building ROCm samples..."
      pushd /opt/rocm/share/hip/samples >/dev/null
      make 2>&1 | tee -a "$LOG_FILE" || true
      popd >/dev/null
    else
      record_skip "hip_samples" "ROCm samples not found at $samples_base or source"
      # Fallback: compile a minimal HIP program
      _test_hip_minimal
      return
    fi
  fi

  # Run all found sample binaries
  local pass_count=0 fail_count=0
  while IFS= read -r binary; do
    local name
    name=$(basename "$binary")
    local output
    if output=$("$binary" 2>&1); then
      ((pass_count++))
      capture "hip_sample_${name}" echo "$output"
    else
      ((fail_count++))
      record_fail "hip_sample_${name}" "Non-zero exit"
    fi
  done < <(find "$samples_base" -type f -executable 2>/dev/null | head -30)

  if [[ "$fail_count" -eq 0 && "$pass_count" -gt 0 ]]; then
    record_pass "hip_samples" "$pass_count samples passed"
  elif [[ "$fail_count" -gt 0 ]]; then
    record_fail "hip_samples" "$fail_count/$((pass_count+fail_count)) samples failed"
  else
    record_skip "hip_samples" "No runnable sample binaries found"
    _test_hip_minimal
  fi

  # rocBLAS SGEMM test
  if cmd_exists rocblas-bench; then
    local sgemm_out
    sgemm_out=$(rocblas-bench -f gemm -r f32_r --transpA N --transpB N -m 4096 -n 4096 -k 4096 2>&1 | tail -5 || true)
    capture "rocblas_sgemm" echo "$sgemm_out"
    if echo "$sgemm_out" | grep -qi "gflops\|performance"; then
      record_pass "rocblas_sgemm" "rocBLAS SGEMM completed"
    else
      record_warn "rocblas_sgemm" "Unexpected rocBLAS output"
    fi
  else
    record_skip "rocblas_sgemm" "rocblas-bench not found"
  fi

  # rocFFT smoke test
  if find /opt/rocm/bin -name "rocfft_rider" 2>/dev/null | grep -q .; then
    local fft_bin
    fft_bin=$(find /opt/rocm/bin -name "rocfft_rider" | head -1)
    if "$fft_bin" -t 0 -x 4096 2>/dev/null | grep -qi "pass\|result"; then
      record_pass "rocfft_smoke"
    else
      record_warn "rocfft_smoke" "rocFFT output unexpected"
    fi
  else
    record_skip "rocfft_smoke" "rocfft_rider not found"
  fi
}

_test_hip_minimal() {
  info "Compiling minimal HIP vector-add..."
  local src="/tmp/hip_validate_$$.cpp"
  local bin="/tmp/hip_validate_$$"
  cat > "$src" << 'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void vadd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int N = 1024;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N*4); hipMalloc(&d_b, N*4); hipMalloc(&d_c, N*4);
    vadd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("HIP error: %s\n", hipGetErrorString(err)); return 1; }
    printf("HIP vector add: OK\n");
    return 0;
}
EOF
  if hipcc "$src" -o "$bin" 2>/dev/null && "$bin" 2>/dev/null | grep -q "OK"; then
    record_pass "hip_minimal_compile_run" "Vector-add kernel compiled and ran successfully"
  else
    record_fail "hip_minimal_compile_run" "hipcc compile/run failed"
  fi
  rm -f "$src" "$bin"
}
