#!/usr/bin/env bash
# =============================================================================
# Module 15 — Targeted GPU Micro-Benchmarks
#
# Tests:
#   cuda_matmul_fp16  — FP16 GEMM / matrix core utilization
#   cuda_matmul_fp32  — FP32 GEMM / CUDA core utilization
#   memory_stress     — VRAM fill + HBM sustained bandwidth + ECC counters
#   pcie_stress       — Host↔GPU DMA bandwidth, link gen/width, replay counters
#   thermal_stress    — Max-compute thermals, power vs TDP, throttle flags
#
# Metrics captured per test:
#   ECC counters, SM occupancy, matrix/FP32 pipeline activity,
#   memory bandwidth, VRAM utilization, SM clock, core/mem temperature,
#   power draw, throttle flags, PCIe throughput, link health
# =============================================================================

# ── Thresholds (all overridable via env) ──────────────────────────────────────
MB_GEMM_FP16_MIN_TFLOPS="${MB_GEMM_FP16_MIN_TFLOPS:-100}"   # FP16 matrix TFLOPS lower bound
MB_GEMM_FP32_MIN_TFLOPS="${MB_GEMM_FP32_MIN_TFLOPS:-20}"    # FP32 TFLOPS lower bound
MB_HBM_BW_MIN_GBPS="${MB_HBM_BW_MIN_GBPS:-800}"             # HBM bandwidth lower bound (GB/s)
MB_PCIE_BW_MIN_GBPS="${MB_PCIE_BW_MIN_GBPS:-20}"            # PCIe H2D bandwidth lower bound (GB/s)
MB_THERMAL_MAX_TEMP="${MB_THERMAL_MAX_TEMP:-90}"             # Thermal stress max temp (°C)
MB_GEMM_DURATION="${MB_GEMM_DURATION:-30}"                   # Seconds to sustain GEMM loops
MB_MEM_DURATION="${MB_MEM_DURATION:-60}"                     # Seconds for memory stress
MB_PCIE_ITER="${MB_PCIE_ITER:-200}"                          # PCIe DMA iteration count
MB_THERMAL_DURATION="${MB_THERMAL_DURATION:-120}"            # Thermal stress duration (seconds)

# ── Temp dir for compiled kernels ─────────────────────────────────────────────
_MB_TMP="/tmp/rocm_mb_$$"
mkdir -p "$_MB_TMP"

# =============================================================================
# Helper: compile a HIP program, return binary path or empty string on failure
# =============================================================================
_mb_compile() {
  local name="$1" src="$2"
  local out="$_MB_TMP/${name}"
  local src_file="$_MB_TMP/${name}.cpp"
  printf '%s' "$src" > "$src_file"
  if hipcc "$src_file" -o "$out" -O3 -lrocblas 2>>"$LOG_FILE"; then
    echo "$out"
  else
    # Try without rocblas in case it's not needed
    if hipcc "$src_file" -o "$out" -O3 2>>"$LOG_FILE"; then
      echo "$out"
    else
      echo ""
    fi
  fi
}

# =============================================================================
# Helper: snapshot all per-GPU metrics via rocm-smi into a JSON-like record
# =============================================================================
_mb_snapshot_metrics() {
  local label="$1"
  local out="$RESULTS_DIR/mb_metrics_${label}.txt"
  {
    echo "=== Metrics snapshot: $label @ $(date) ==="
    rocm-smi --showtemp   2>/dev/null || true
    rocm-smi --showpower  2>/dev/null || true
    rocm-smi --showclocks 2>/dev/null || true
    rocm-smi --showmeminfo vram 2>/dev/null || true
    rocm-smi --showuse    2>/dev/null || true
    rocm-smi --showrasinfo all 2>/dev/null || true
  } | tee "$out" >> "$LOG_FILE"
}

# =============================================================================
# Helper: read a single scalar from rocm-smi output
# =============================================================================
_mb_max_temp()  { rocm-smi --showtemp  2>/dev/null | grep -oP "[0-9]+(?=c)"        | sort -rn | head -1 || echo "0"; }
_mb_max_power() { rocm-smi --showpower 2>/dev/null | grep -oP "[0-9]+(?=\s*W)"     | sort -rn | head -1 || echo "0"; }
_mb_sm_clock()  { rocm-smi --showclocks 2>/dev/null | grep -iP "sclk|shader"       | grep -oP "[0-9]+"  | sort -rn | head -1 || echo "0"; }
_mb_vram_used_pct() {
  python3 - 2>/dev/null <<'PYEOF' || echo "0"
import subprocess, re
out = subprocess.run(["rocm-smi","--showmeminfo","vram"],
    capture_output=True, text=True).stdout
used  = sum(int(x) for x in re.findall(r"VRAM Used Memory.*?:\s*(\d+)", out))
total = sum(int(x) for x in re.findall(r"VRAM Total Memory.*?:\s*(\d+)", out))
print(round(used*100/total, 1) if total else 0)
PYEOF
}
_mb_throttle_active() {
  rocm-smi --showrasinfo all 2>/dev/null |     grep -iE "throttle.*true|thermal.*throttle" | wc -l | tr -d '[:space:]' || echo "0"
}
_mb_ecc_errors() {
  rocm-smi --showrasinfo all 2>/dev/null |     grep -oP "(?i)uncorrected.*?:\s*\K[0-9]+" |     awk '{s+=$1} END{print s+0}' || echo "0"
}

# =============================================================================
# 15.1  FP16 GEMM — matrix core utilization + SM occupancy
# =============================================================================
_mb_gemm_fp16_src() { cat << 'HIPSRC'
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_HIP(x) do { \
    hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %s\n",hipGetErrorString(e));exit(1);} \
} while(0)
#define CHECK_RB(x)  do { \
    rocblas_status s=(x); if(s!=rocblas_status_success){fprintf(stderr,"rocBLAS error %d\n",(int)s);exit(1);} \
} while(0)

int main(int argc, char** argv) {
    const int M=8192, N=8192, K=8192;
    const int ITERS = argc>1 ? atoi(argv[1]) : 50;

    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));

    size_t bytes = (size_t)M*K * sizeof(rocblas_half)
                 + (size_t)K*N * sizeof(rocblas_half)
                 + (size_t)M*N * sizeof(rocblas_half);

    rocblas_half *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, (size_t)M*K*sizeof(rocblas_half)));
    CHECK_HIP(hipMalloc(&dB, (size_t)K*N*sizeof(rocblas_half)));
    CHECK_HIP(hipMalloc(&dC, (size_t)M*N*sizeof(rocblas_half)));
    CHECK_HIP(hipMemset(dA, 1, (size_t)M*K*sizeof(rocblas_half)));
    CHECK_HIP(hipMemset(dB, 1, (size_t)K*N*sizeof(rocblas_half)));

    rocblas_half alpha_h = rocblas_float_to_half(1.0f);
    rocblas_half beta_h  = rocblas_float_to_half(0.0f);

    // Warm-up
    CHECK_RB(rocblas_hgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K, &alpha_h, dA, M, dB, K, &beta_h, dC, M));
    CHECK_HIP(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        CHECK_RB(rocblas_hgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            M, N, K, &alpha_h, dA, M, dB, K, &beta_h, dC, M));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1-t0).count();
    double tflops  = 2.0 * M * N * K * ITERS / elapsed / 1e12;
    double gbs     = bytes * ITERS / elapsed / 1e9;

    printf("FP16_GEMM M=%d N=%d K=%d iters=%d\n", M, N, K, ITERS);
    printf("FP16_TFLOPS=%.2f\n", tflops);
    printf("FP16_ARITHMETIC_INTENSITY=%.2f\n", (2.0*M*N*K) / bytes);
    printf("FP16_ELAPSED_S=%.3f\n", elapsed);
    printf("RESULT=PASS\n");

    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
HIPSRC
}

_run_matmul_fp16() {
  local gpu_idx="${1:-0}"
  info "Compiling FP16 GEMM kernel (GPU $gpu_idx)..."
  local src
  src=$(_mb_gemm_fp16_src)
  local bin
  bin=$(_mb_compile "gemm_fp16_g${gpu_idx}" "$src")
  if [[ -z "$bin" ]]; then
    record_fail "matmul_fp16_compile_gpu${gpu_idx}" "hipcc failed — check rocBLAS install"
    return
  fi
  record_pass "matmul_fp16_compile_gpu${gpu_idx}"

  # Snapshot metrics before
  _mb_snapshot_metrics "fp16_before_gpu${gpu_idx}"

  # Run with monitoring in background
  local metrics_pid outfile="$RESULTS_DIR/mb_fp16_gpu${gpu_idx}.txt"
  {
    local end=$(( $(date +%s) + MB_GEMM_DURATION ))
    while [[ $(date +%s) -lt $end ]]; do
      rocm-smi --showuse --showclocks 2>/dev/null
      sleep 5
    done
  } > "$RESULTS_DIR/mb_fp16_monitor_gpu${gpu_idx}.txt" &
  metrics_pid=$!

  HIP_VISIBLE_DEVICES="$gpu_idx" "$bin" 20 2>&1 | tee "$outfile" >> "$LOG_FILE"
  kill "$metrics_pid" 2>/dev/null || true; wait "$metrics_pid" 2>/dev/null || true

  _mb_snapshot_metrics "fp16_after_gpu${gpu_idx}"

  # Parse results
  if grep -q "RESULT=PASS" "$outfile"; then
    local tflops
    tflops=$(grep "FP16_TFLOPS=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    local arith_intensity
    arith_intensity=$(grep "FP16_ARITHMETIC_INTENSITY=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")

    if awk "BEGIN{exit !($tflops >= $MB_GEMM_FP16_MIN_TFLOPS)}"; then
      record_pass "matmul_fp16_throughput_gpu${gpu_idx}" \
        "${tflops} TFLOPS (threshold: ≥${MB_GEMM_FP16_MIN_TFLOPS}) arith_intensity=${arith_intensity}"
    else
      record_fail "matmul_fp16_throughput_gpu${gpu_idx}" \
        "${tflops} TFLOPS below threshold ${MB_GEMM_FP16_MIN_TFLOPS}"
    fi

    # SM clock during run
    local sm_clk
    sm_clk=$(grep -oP "sclk.*?(\d+)Mhz" "$RESULTS_DIR/mb_fp16_monitor_gpu${gpu_idx}.txt" 2>/dev/null | \
             grep -oP "[0-9]+(?=Mhz)" | sort -rn | head -1 || echo "0")
    if [[ "$sm_clk" -gt 0 ]]; then
      record_pass "matmul_fp16_sm_clock_gpu${gpu_idx}" "Peak SM clock: ${sm_clk} MHz"
    else
      record_warn "matmul_fp16_sm_clock_gpu${gpu_idx}" "Could not read SM clock during FP16 run"
    fi

    # Check matrix-core pipeline active (GPU utilization > 80% implies pipeline engaged)
    local gpu_util
    gpu_util=$(grep -oP "GPU use.*?:\s*\K[0-9]+" "$RESULTS_DIR/mb_fp16_monitor_gpu${gpu_idx}.txt" 2>/dev/null | \
               sort -rn | head -1 || echo "0")
    if [[ "$gpu_util" -ge 80 ]]; then
      record_pass "matmul_fp16_pipeline_active_gpu${gpu_idx}" "GPU utilization ${gpu_util}% (matrix cores engaged)"
    else
      record_warn "matmul_fp16_pipeline_active_gpu${gpu_idx}" "GPU utilization ${gpu_util}% — matrix cores may be underutilized"
    fi
  else
    record_fail "matmul_fp16_run_gpu${gpu_idx}" "Kernel did not complete cleanly"
  fi

  # ECC after FP16
  local ecc
  ecc=$(_mb_ecc_errors)
  if [[ "$ecc" -eq 0 ]]; then
    record_pass "matmul_fp16_ecc_gpu${gpu_idx}" "0 uncorrected ECC errors"
  else
    record_fail "matmul_fp16_ecc_gpu${gpu_idx}" "$ecc uncorrected ECC error(s) after FP16 GEMM"
  fi
}

# =============================================================================
# 15.2  FP32 GEMM — CUDA core utilization + SM multiprocessor occupancy
# =============================================================================
_mb_gemm_fp32_src() { cat << 'HIPSRC'
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_HIP(x) do { \
    hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %s\n",hipGetErrorString(e));exit(1);} \
} while(0)
#define CHECK_RB(x)  do { \
    rocblas_status s=(x); if(s!=rocblas_status_success){fprintf(stderr,"rocBLAS error %d\n",(int)s);exit(1);} \
} while(0)

int main(int argc, char** argv) {
    const int M=8192, N=8192, K=8192;
    const int ITERS = argc>1 ? atoi(argv[1]) : 30;

    rocblas_handle handle;
    CHECK_RB(rocblas_create_handle(&handle));

    float *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, (size_t)M*K*sizeof(float)));
    CHECK_HIP(hipMalloc(&dB, (size_t)K*N*sizeof(float)));
    CHECK_HIP(hipMalloc(&dC, (size_t)M*N*sizeof(float)));
    CHECK_HIP(hipMemset(dA, 0, (size_t)M*K*sizeof(float)));
    CHECK_HIP(hipMemset(dB, 0, (size_t)K*N*sizeof(float)));

    const float alpha=1.0f, beta=0.0f;

    // Warm-up
    CHECK_RB(rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    CHECK_HIP(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        CHECK_RB(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1-t0).count();
    double tflops  = 2.0 * M * N * K * ITERS / elapsed / 1e12;
    size_t bytes   = ((size_t)M*K + (size_t)K*N + (size_t)M*N) * sizeof(float);
    double arith_intensity = (2.0*M*N*K) / bytes;

    printf("FP32_GEMM M=%d N=%d K=%d iters=%d\n", M, N, K, ITERS);
    printf("FP32_TFLOPS=%.2f\n", tflops);
    printf("FP32_ARITHMETIC_INTENSITY=%.2f\n", arith_intensity);
    printf("FP32_ELAPSED_S=%.3f\n", elapsed);

    // SM occupancy estimation via wavefront count proxy
    int wf_count = 0;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
        wf_count = prop.maxThreadsPerMultiProcessor / prop.warpSize * prop.multiProcessorCount;
        printf("FP32_MAX_WAVEFRONTS=%d\n", wf_count);
        printf("FP32_CU_COUNT=%d\n", prop.multiProcessorCount);
        printf("FP32_DEVICE=%s\n", prop.name);
    }
    printf("RESULT=PASS\n");

    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
HIPSRC
}

_run_matmul_fp32() {
  local gpu_idx="${1:-0}"
  info "Compiling FP32 GEMM kernel (GPU $gpu_idx)..."
  local src
  src=$(_mb_gemm_fp32_src)
  local bin
  bin=$(_mb_compile "gemm_fp32_g${gpu_idx}" "$src")
  if [[ -z "$bin" ]]; then
    record_fail "matmul_fp32_compile_gpu${gpu_idx}" "hipcc failed"
    return
  fi
  record_pass "matmul_fp32_compile_gpu${gpu_idx}"

  _mb_snapshot_metrics "fp32_before_gpu${gpu_idx}"

  local outfile="$RESULTS_DIR/mb_fp32_gpu${gpu_idx}.txt"
  local monitor_file="$RESULTS_DIR/mb_fp32_monitor_gpu${gpu_idx}.txt"
  {
    local end=$(( $(date +%s) + MB_GEMM_DURATION ))
    while [[ $(date +%s) -lt $end ]]; do
      rocm-smi --showuse --showclocks --showpower 2>/dev/null
      sleep 3
    done
  } > "$monitor_file" &
  local metrics_pid=$!

  HIP_VISIBLE_DEVICES="$gpu_idx" "$bin" 15 2>&1 | tee "$outfile" >> "$LOG_FILE"
  kill "$metrics_pid" 2>/dev/null || true; wait "$metrics_pid" 2>/dev/null || true

  _mb_snapshot_metrics "fp32_after_gpu${gpu_idx}"

  if grep -q "RESULT=PASS" "$outfile"; then
    local tflops
    tflops=$(grep "FP32_TFLOPS=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    local cu_count
    cu_count=$(grep "FP32_CU_COUNT=" "$outfile" | grep -oP "[0-9]+" || echo "N/A")
    local device_name
    device_name=$(grep "FP32_DEVICE=" "$outfile" | cut -d= -f2 || echo "unknown")

    if awk "BEGIN{exit !($tflops >= $MB_GEMM_FP32_MIN_TFLOPS)}"; then
      record_pass "matmul_fp32_throughput_gpu${gpu_idx}" \
        "${tflops} TFLOPS (threshold: ≥${MB_GEMM_FP32_MIN_TFLOPS}) CUs=${cu_count} device=${device_name}"
    else
      record_fail "matmul_fp32_throughput_gpu${gpu_idx}" \
        "${tflops} TFLOPS below threshold ${MB_GEMM_FP32_MIN_TFLOPS}"
    fi

    # CU count sanity — should be > 0
    if [[ "$cu_count" != "N/A" && "$cu_count" -gt 0 ]]; then
      record_pass "matmul_fp32_cu_count_gpu${gpu_idx}" "$cu_count Compute Units detected"
    else
      record_warn "matmul_fp32_cu_count_gpu${gpu_idx}" "Could not read CU count"
    fi

    # SM multiprocessor occupancy via utilization during run
    local util
    util=$(grep -oP "GPU use.*?:\s*\K[0-9]+" "$monitor_file" 2>/dev/null | sort -rn | head -1 || echo "0")
    if [[ "$util" -ge 70 ]]; then
      record_pass "matmul_fp32_sm_occupancy_gpu${gpu_idx}" "GPU utilization ${util}% (FP32 SMs occupied)"
    else
      record_warn "matmul_fp32_sm_occupancy_gpu${gpu_idx}" "GPU utilization ${util}% — low SM occupancy for SGEMM"
    fi

    # FP32 pipeline power signal — power should rise significantly
    local peak_power
    peak_power=$(grep -oP "[0-9]+(?=\s*W)" "$monitor_file" 2>/dev/null | sort -rn | head -1 || echo "0")
    if [[ "$peak_power" -gt 50 ]]; then
      record_pass "matmul_fp32_power_signal_gpu${gpu_idx}" "Peak power ${peak_power}W (FP32 pipeline active)"
    else
      record_warn "matmul_fp32_power_signal_gpu${gpu_idx}" "Peak power ${peak_power}W — FP32 pipeline may be idle"
    fi
  else
    record_fail "matmul_fp32_run_gpu${gpu_idx}" "Kernel did not complete cleanly"
  fi

  # ECC after FP32
  local ecc
  ecc=$(_mb_ecc_errors)
  if [[ "$ecc" -eq 0 ]]; then
    record_pass "matmul_fp32_ecc_gpu${gpu_idx}" "0 uncorrected ECC errors"
  else
    record_fail "matmul_fp32_ecc_gpu${gpu_idx}" "$ecc uncorrected ECC error(s)"
  fi
}

# =============================================================================
# 15.3  Memory Stress — VRAM fill + HBM sustained RW + ECC counters
# =============================================================================
_mb_memory_stress_src() { cat << 'HIPSRC'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

#define CHECK_HIP(x) do { \
    hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));exit(1);} \
} while(0)

// Streaming read-write kernel — saturates HBM bandwidth
__global__ void stream_copy(float* __restrict__ dst,
                            const float* __restrict__ src,
                            size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i] + 1.0f;
}

// Fill kernel — ensures every VRAM byte is written
__global__ void fill_vram(float* buf, float val, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) buf[i] = val;
}

int main(int argc, char** argv) {
    int duration_s = argc>1 ? atoi(argv[1]) : 30;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));

    // Allocate 90% of available VRAM
    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t alloc = (size_t)(free_mem * 0.90);
    size_t n = alloc / sizeof(float) / 2;  // split into src+dst

    printf("MEM_TOTAL_BYTES=%zu\n",  total_mem);
    printf("MEM_ALLOC_BYTES=%zu\n",  alloc);
    printf("MEM_DEVICE=%s\n",        prop.name);
    printf("MEM_HBM_BUSES=%d\n",     prop.memoryBusWidth);

    float *d_src, *d_dst;
    CHECK_HIP(hipMalloc(&d_src, n * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_dst, n * sizeof(float)));

    // Phase 1: Fill all VRAM
    int blocks = (prop.multiProcessorCount * 64);
    int threads = 256;
    fill_vram<<<blocks, threads>>>(d_src, 1.0f, n);
    fill_vram<<<blocks, threads>>>(d_dst, 0.0f, n);
    CHECK_HIP(hipDeviceSynchronize());
    printf("MEM_VRAM_FILL=PASS\n");

    // Phase 2: Sustained copy loop measuring bandwidth
    long long total_bytes = 0;
    int iters = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end   = t_start + std::chrono::seconds(duration_s);

    while (std::chrono::high_resolution_clock::now() < t_end) {
        stream_copy<<<blocks, threads>>>(d_dst, d_src, n);
        CHECK_HIP(hipDeviceSynchronize());
        total_bytes += (long long)n * sizeof(float) * 2;  // read + write
        iters++;
    }

    auto t_actual = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_actual - t_start).count();
    double bw_gbps = (double)total_bytes / elapsed / 1e9;

    printf("MEM_BANDWIDTH_GBPS=%.2f\n", bw_gbps);
    printf("MEM_ITERS=%d\n",            iters);
    printf("MEM_ELAPSED_S=%.2f\n",      elapsed);
    printf("MEM_BYTES_MOVED=%lld\n",    total_bytes);

    // Verify data integrity (sample check)
    float* h_sample = new float[1024];
    CHECK_HIP(hipMemcpy(h_sample, d_dst, 1024*sizeof(float), hipMemcpyDeviceToHost));
    int corrupt = 0;
    for (int i=0; i<1024; i++) if (h_sample[i] < 0.9f || h_sample[i] > 3.1f) corrupt++;
    printf("MEM_INTEGRITY_ERRORS=%d\n", corrupt);
    delete[] h_sample;

    printf("RESULT=PASS\n");

    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}
HIPSRC
}

_run_memory_stress() {
  local gpu_idx="${1:-0}"
  info "Compiling memory stress kernel (GPU $gpu_idx)..."
  local src
  src=$(_mb_memory_stress_src)
  local bin
  bin=$(_mb_compile "mem_stress_g${gpu_idx}" "$src")
  if [[ -z "$bin" ]]; then
    record_fail "memory_stress_compile_gpu${gpu_idx}" "hipcc failed"
    return
  fi
  record_pass "memory_stress_compile_gpu${gpu_idx}"

  # ECC baseline before run
  local ecc_before
  ecc_before=$(_mb_ecc_errors)
  _mb_snapshot_metrics "mem_before_gpu${gpu_idx}"

  local outfile="$RESULTS_DIR/mb_mem_gpu${gpu_idx}.txt"
  local monitor_file="$RESULTS_DIR/mb_mem_monitor_gpu${gpu_idx}.txt"

  # Concurrent metrics sampling
  {
    local end=$(( $(date +%s) + MB_MEM_DURATION + 10 ))
    while [[ $(date +%s) -lt $end ]]; do
      rocm-smi --showmeminfo vram --showclocks --showtemp 2>/dev/null
      sleep 3
    done
  } > "$monitor_file" &
  local monitor_pid=$!

  HIP_VISIBLE_DEVICES="$gpu_idx" "$bin" "$MB_MEM_DURATION" 2>&1 | tee "$outfile" >> "$LOG_FILE"
  kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true

  _mb_snapshot_metrics "mem_after_gpu${gpu_idx}"

  if grep -q "RESULT=PASS" "$outfile"; then
    # Bandwidth check
    local bw
    bw=$(grep "MEM_BANDWIDTH_GBPS=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    if awk "BEGIN{exit !($bw >= $MB_HBM_BW_MIN_GBPS)}"; then
      record_pass "memory_stress_bandwidth_gpu${gpu_idx}" "${bw} GB/s (threshold: ≥${MB_HBM_BW_MIN_GBPS})"
    else
      record_fail "memory_stress_bandwidth_gpu${gpu_idx}" "${bw} GB/s below threshold ${MB_HBM_BW_MIN_GBPS}"
    fi

    # VRAM fill check
    if grep -q "MEM_VRAM_FILL=PASS" "$outfile"; then
      local total_bytes alloc_bytes
      total_bytes=$(grep "MEM_TOTAL_BYTES=" "$outfile" | grep -oP "[0-9]+")
      alloc_bytes=$(grep "MEM_ALLOC_BYTES="  "$outfile" | grep -oP "[0-9]+")
      local alloc_gb=$(( ${alloc_bytes:-0} / 1024 / 1024 / 1024 ))
      record_pass "memory_stress_vram_fill_gpu${gpu_idx}" "Filled ${alloc_gb} GB VRAM (90% of total)"
    else
      record_fail "memory_stress_vram_fill_gpu${gpu_idx}" "VRAM fill phase failed"
    fi

    # Data integrity
    local integrity_errors
    integrity_errors=$(grep "MEM_INTEGRITY_ERRORS=" "$outfile" | grep -oP "[0-9]+" || echo "0")
    if [[ "$integrity_errors" -eq 0 ]]; then
      record_pass "memory_stress_integrity_gpu${gpu_idx}" "0 data integrity errors"
    else
      record_fail "memory_stress_integrity_gpu${gpu_idx}" "$integrity_errors data corruption error(s)"
    fi

    # VRAM utilization during run (from monitor)
    local peak_vram_pct
    peak_vram_pct=$(python3 - "$monitor_file" 2>/dev/null <<'PYEOF' || echo "0"
import sys, re
with open(sys.argv[1]) as f: txt = f.read()
used  = [int(x) for x in re.findall(r"VRAM Used Memory.*?:\s*(\d+)", txt)]
total = [int(x) for x in re.findall(r"VRAM Total Memory.*?:\s*(\d+)", txt)]
if total and used:
    peak = max(u/t*100 for u,t in zip(used, total))
    print(round(peak, 1))
else:
    print(0)
PYEOF
)
    if awk "BEGIN{exit !($peak_vram_pct >= 80)}"; then
      record_pass "memory_stress_vram_utilization_gpu${gpu_idx}" "Peak VRAM utilization: ${peak_vram_pct}%"
    else
      record_warn "memory_stress_vram_utilization_gpu${gpu_idx}" "Peak VRAM utilization: ${peak_vram_pct}% (expected ≥80%)"
    fi

    # Memory clock during run
    local mem_clk
    mem_clk=$(grep -oP "mclk.*?(\d+)Mhz" "$monitor_file" 2>/dev/null | grep -oP "[0-9]+(?=Mhz)" | sort -rn | head -1 || echo "0")
    if [[ "$mem_clk" -gt 0 ]]; then
      record_pass "memory_stress_mem_clock_gpu${gpu_idx}" "Memory clock: ${mem_clk} MHz"
    else
      record_warn "memory_stress_mem_clock_gpu${gpu_idx}" "Could not read memory clock"
    fi
  else
    record_fail "memory_stress_run_gpu${gpu_idx}" "Kernel did not complete"
  fi

  # ECC delta
  local ecc_after
  ecc_after=$(_mb_ecc_errors)
  local ecc_delta=$(( ecc_after - ecc_before ))
  if [[ "$ecc_delta" -eq 0 ]]; then
    record_pass "memory_stress_ecc_gpu${gpu_idx}" "0 new ECC errors during memory stress"
  else
    record_fail "memory_stress_ecc_gpu${gpu_idx}" "$ecc_delta new uncorrected ECC error(s) — possible VRAM fault"
  fi
}

# =============================================================================
# 15.4  PCIe Stress — H2D/D2H DMA bandwidth + link health + replay counters
# =============================================================================
_mb_pcie_stress_src() { cat << 'HIPSRC'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

#define CHECK_HIP(x) do { \
    hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));exit(1);} \
} while(0)

int main(int argc, char** argv) {
    const int ITERS = argc>1 ? atoi(argv[1]) : 100;
    const size_t BUF_BYTES = 512ULL * 1024 * 1024;  // 512 MB per transfer

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    printf("PCIE_DEVICE=%s\n",      prop.name);
    printf("PCIE_BUS_WIDTH=%d\n",   prop.pciBusID);

    // Pinned host memory for maximum PCIe throughput
    float *h_buf, *d_buf;
    CHECK_HIP(hipHostMalloc(&h_buf, BUF_BYTES, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_buf, BUF_BYTES));
    memset(h_buf, 1, BUF_BYTES);

    // H2D: host → device
    CHECK_HIP(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<ITERS; i++) {
        CHECK_HIP(hipMemcpy(d_buf, h_buf, BUF_BYTES, hipMemcpyHostToDevice));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double h2d_bw = (double)BUF_BYTES * ITERS /
                    std::chrono::duration<double>(t1-t0).count() / 1e9;

    // D2H: device → host
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<ITERS; i++) {
        CHECK_HIP(hipMemcpy(h_buf, d_buf, BUF_BYTES, hipMemcpyDeviceToHost));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto t3 = std::chrono::high_resolution_clock::now();
    double d2h_bw = (double)BUF_BYTES * ITERS /
                    std::chrono::duration<double>(t3-t2).count() / 1e9;

    // Bidirectional (async)
    hipStream_t s1, s2;
    CHECK_HIP(hipStreamCreate(&s1));
    CHECK_HIP(hipStreamCreate(&s2));
    float *h_buf2, *d_buf2;
    CHECK_HIP(hipHostMalloc(&h_buf2, BUF_BYTES, hipHostMallocDefault));
    CHECK_HIP(hipMalloc(&d_buf2, BUF_BYTES));

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<ITERS; i++) {
        CHECK_HIP(hipMemcpyAsync(d_buf,  h_buf,  BUF_BYTES, hipMemcpyHostToDevice,   s1));
        CHECK_HIP(hipMemcpyAsync(h_buf2, d_buf2, BUF_BYTES, hipMemcpyDeviceToHost,   s2));
    }
    CHECK_HIP(hipStreamSynchronize(s1));
    CHECK_HIP(hipStreamSynchronize(s2));
    auto t5 = std::chrono::high_resolution_clock::now();
    double bidir_bw = (double)BUF_BYTES * ITERS * 2 /
                      std::chrono::duration<double>(t5-t4).count() / 1e9;

    printf("PCIE_H2D_GBPS=%.2f\n",   h2d_bw);
    printf("PCIE_D2H_GBPS=%.2f\n",   d2h_bw);
    printf("PCIE_BIDIR_GBPS=%.2f\n", bidir_bw);
    printf("PCIE_BUF_MB=%zu\n",      BUF_BYTES / 1024 / 1024);
    printf("PCIE_ITERS=%d\n",        ITERS);
    printf("RESULT=PASS\n");

    hipStreamDestroy(s1); hipStreamDestroy(s2);
    hipFree(d_buf); hipFree(d_buf2);
    hipHostFree(h_buf); hipHostFree(h_buf2);
    return 0;
}
HIPSRC
}

_run_pcie_stress() {
  local gpu_idx="${1:-0}"
  info "Compiling PCIe stress kernel (GPU $gpu_idx)..."
  local src
  src=$(_mb_pcie_stress_src)
  local bin
  bin=$(_mb_compile "pcie_stress_g${gpu_idx}" "$src")
  if [[ -z "$bin" ]]; then
    record_fail "pcie_stress_compile_gpu${gpu_idx}" "hipcc failed"
    return
  fi
  record_pass "pcie_stress_compile_gpu${gpu_idx}"

  # Capture PCIe link state BEFORE transfers
  local lspci_out="$RESULTS_DIR/mb_pcie_lspci_gpu${gpu_idx}.txt"
  lspci -vvv 2>/dev/null | grep -A 40 -E "AMD|Radeon|Advanced Micro" | \
    grep -E "LnkSta:|LnkCap:|Rply" | head -40 > "$lspci_out" || true

  local outfile="$RESULTS_DIR/mb_pcie_gpu${gpu_idx}.txt"
  HIP_VISIBLE_DEVICES="$gpu_idx" "$bin" "$MB_PCIE_ITER" 2>&1 | tee "$outfile" >> "$LOG_FILE"

  if grep -q "RESULT=PASS" "$outfile"; then
    local h2d d2h bidir
    h2d=$(  grep "PCIE_H2D_GBPS="   "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    d2h=$(  grep "PCIE_D2H_GBPS="   "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    bidir=$(grep "PCIE_BIDIR_GBPS=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")

    # H2D bandwidth
    if awk "BEGIN{exit !($h2d >= $MB_PCIE_BW_MIN_GBPS)}"; then
      record_pass "pcie_stress_h2d_bw_gpu${gpu_idx}" "${h2d} GB/s (threshold: ≥${MB_PCIE_BW_MIN_GBPS})"
    else
      record_fail "pcie_stress_h2d_bw_gpu${gpu_idx}" "${h2d} GB/s below threshold ${MB_PCIE_BW_MIN_GBPS}"
    fi

    # D2H bandwidth
    if awk "BEGIN{exit !($d2h >= $MB_PCIE_BW_MIN_GBPS)}"; then
      record_pass "pcie_stress_d2h_bw_gpu${gpu_idx}" "${d2h} GB/s"
    else
      record_fail "pcie_stress_d2h_bw_gpu${gpu_idx}" "${d2h} GB/s below threshold ${MB_PCIE_BW_MIN_GBPS}"
    fi

    # Bidirectional
    record_pass "pcie_stress_bidir_bw_gpu${gpu_idx}" "${bidir} GB/s bidirectional"

    # PCIe symmetry — H2D vs D2H should be within 20% of each other
    if awk "BEGIN{
        r = ($h2d > $d2h) ? $d2h/$h2d : $h2d/$d2h;
        exit !(r < 0.80)
    }"; then
      record_warn "pcie_stress_symmetry_gpu${gpu_idx}" "H2D=${h2d} vs D2H=${d2h} GB/s — >20% asymmetry"
    else
      record_pass "pcie_stress_symmetry_gpu${gpu_idx}" "H2D/D2H within 20% (${h2d} / ${d2h} GB/s)"
    fi
  else
    record_fail "pcie_stress_run_gpu${gpu_idx}" "PCIe stress kernel failed"
  fi

  # PCIe link gen / width from lspci
  if [[ -s "$lspci_out" ]]; then
    local link_speed link_width
    link_speed=$(grep -oP "LnkSta:.*Speed \K[0-9.]+GT/s" "$lspci_out" | head -1 || true)
    link_width=$(grep -oP "LnkSta:.*Width \Kx[0-9]+" "$lspci_out" | head -1 || true)
    if [[ -n "$link_speed" && -n "$link_width" ]]; then
      record_pass "pcie_stress_link_state_gpu${gpu_idx}" "Speed=${link_speed} Width=${link_width}"
    else
      record_warn "pcie_stress_link_state_gpu${gpu_idx}" "Could not parse LnkSta — check lspci output"
    fi

    # PCIe replay counter — non-zero indicates link errors
    local replay_count
    replay_count=$(grep -oP "Replay#:\K[0-9]+" "$lspci_out" | awk '{s+=$1} END{print s+0}')
    if [[ "$replay_count" -eq 0 ]]; then
      record_pass "pcie_stress_replay_counter_gpu${gpu_idx}" "0 PCIe replays"
    else
      record_fail "pcie_stress_replay_counter_gpu${gpu_idx}" "$replay_count PCIe replay(s) — link instability"
    fi

    # Check for Gen4/Gen5 (expected for MI300X-class)
    if echo "$link_speed" | grep -qE "16GT|32GT"; then
      record_pass "pcie_stress_link_gen_gpu${gpu_idx}" "PCIe Gen4/Gen5 confirmed ($link_speed)"
    else
      record_warn "pcie_stress_link_gen_gpu${gpu_idx}" "PCIe speed $link_speed — expected Gen4 (16GT/s) or Gen5 (32GT/s)"
    fi
  else
    record_skip "pcie_stress_link_state_gpu${gpu_idx}" "lspci output unavailable"
    record_skip "pcie_stress_replay_counter_gpu${gpu_idx}" "lspci output unavailable"
    record_skip "pcie_stress_link_gen_gpu${gpu_idx}"
  fi

  # PCIe bandwidth from rocm-smi (if available)
  if rocm-smi --showpciebw &>/dev/null 2>&1; then
    local smi_pcie
    smi_pcie=$(rocm-smi --showpciebw 2>/dev/null | grep -oP "[0-9]+(?=\s*MB/s)" | head -1 || echo "0")
    if [[ "$smi_pcie" -gt 0 ]]; then
      local smi_pcie_gbps
      smi_pcie_gbps=$(echo "scale=1; $smi_pcie / 1000" | bc 2>/dev/null || echo "0")
      record_pass "pcie_stress_smi_throughput_gpu${gpu_idx}" "rocm-smi reports ${smi_pcie_gbps} GB/s"
    fi
  fi
}

# =============================================================================
# 15.5  Thermal Stress — max compute, temp/power vs TDP, throttle detection
# =============================================================================
_mb_thermal_stress_src() { cat << 'HIPSRC'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_HIP(x) do { \
    hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));exit(1);} \
} while(0)

// Maximum-power kernel: FMA-heavy, register-blocked to maximise ALU utilisation
__global__ void __launch_bounds__(256, 4)
fma_stress(float* out, float a, float b, int n_reps) {
    float acc0 = a, acc1 = b, acc2 = a+1, acc3 = b+1;
    float acc4 = a+2, acc5 = b+2, acc6 = a+3, acc7 = b+3;
    #pragma unroll 64
    for (int i = 0; i < n_reps; i++) {
        acc0 = acc0 * a + acc1;
        acc1 = acc1 * b + acc2;
        acc2 = acc2 * a + acc3;
        acc3 = acc3 * b + acc4;
        acc4 = acc4 * a + acc5;
        acc5 = acc5 * b + acc6;
        acc6 = acc6 * a + acc7;
        acc7 = acc7 * b + acc0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = acc0 + acc1 + acc2 + acc3;
}

int main(int argc, char** argv) {
    int duration_s = argc>1 ? atoi(argv[1]) : 60;

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    int blocks  = prop.multiProcessorCount * 16;
    int threads = 256;

    float* d_out;
    CHECK_HIP(hipMalloc(&d_out, sizeof(float)));

    printf("THERMAL_DEVICE=%s\n", prop.name);
    printf("THERMAL_CU_COUNT=%d\n", prop.multiProcessorCount);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end   = t_start + std::chrono::seconds(duration_s);
    long long kernel_launches = 0;

    while (std::chrono::high_resolution_clock::now() < t_end) {
        fma_stress<<<blocks, threads>>>(d_out, 1.00001f, 0.99999f, 8192);
        CHECK_HIP(hipDeviceSynchronize());
        kernel_launches++;
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    // 8 FMAs × 2 ops × unroll 64 × 8192 reps × blocks × threads
    double flops = 8.0 * 2.0 * 64 * 8192 * kernel_launches * blocks * threads;
    printf("THERMAL_TFLOPS=%.2f\n", flops / elapsed / 1e12);
    printf("THERMAL_LAUNCHES=%lld\n", kernel_launches);
    printf("THERMAL_ELAPSED_S=%.1f\n", elapsed);
    printf("RESULT=PASS\n");

    hipFree(d_out);
    return 0;
}
HIPSRC
}

_run_thermal_stress() {
  local gpu_idx="${1:-0}"
  info "Compiling thermal stress kernel (GPU $gpu_idx, duration=${MB_THERMAL_DURATION}s)..."
  local src
  src=$(_mb_thermal_stress_src)
  local bin
  bin=$(_mb_compile "thermal_stress_g${gpu_idx}" "$src")
  if [[ -z "$bin" ]]; then
    record_fail "thermal_stress_compile_gpu${gpu_idx}" "hipcc failed"
    return
  fi
  record_pass "thermal_stress_compile_gpu${gpu_idx}"

  # Baseline readings before stress
  local temp_before power_before
  temp_before=$(_mb_max_temp)
  power_before=$(_mb_max_power)
  _mb_snapshot_metrics "thermal_before_gpu${gpu_idx}"

  # Detailed monitoring file: temp + power + clocks every 2s
  local monitor_file="$RESULTS_DIR/mb_thermal_monitor_gpu${gpu_idx}.txt"
  {
    local end=$(( $(date +%s) + MB_THERMAL_DURATION + 15 ))
    while [[ $(date +%s) -lt $end ]]; do
      echo "TS=$(date +%s)"
      rocm-smi --showtemp --showpower --showclocks --showrasinfo all 2>/dev/null
      sleep 2
    done
  } > "$monitor_file" &
  local monitor_pid=$!

  local outfile="$RESULTS_DIR/mb_thermal_gpu${gpu_idx}.txt"
  HIP_VISIBLE_DEVICES="$gpu_idx" "$bin" "$MB_THERMAL_DURATION" 2>&1 | tee "$outfile" >> "$LOG_FILE"
  kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true

  _mb_snapshot_metrics "thermal_after_gpu${gpu_idx}"

  if grep -q "RESULT=PASS" "$outfile"; then
    local tflops
    tflops=$(grep "THERMAL_TFLOPS=" "$outfile" | grep -oP "[0-9]+\.[0-9]+")
    record_pass "thermal_stress_compute_sustained_gpu${gpu_idx}" "${tflops} TFLOPS sustained over ${MB_THERMAL_DURATION}s"
  else
    record_fail "thermal_stress_run_gpu${gpu_idx}" "Kernel did not complete"
    kill "$monitor_pid" 2>/dev/null || true
    return
  fi

  # Parse monitor log
  python3 - "$monitor_file" "$RESULTS_DIR/mb_thermal_parsed_gpu${gpu_idx}.json" 2>/dev/null <<'PYEOF' || true
import re, json, sys

with open(sys.argv[1]) as f:
    txt = f.read()

temps  = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)c\b", txt)]
powers = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*W\b", txt)]
sclks  = [float(x) for x in re.findall(r"sclk.*?(\d+)Mhz", txt, re.I)]
throttle_events = len(re.findall(r"(?i)throttle.*?true|power.*?limit.*?active", txt))

data = {
    "peak_temp_c":      max(temps)  if temps  else None,
    "min_temp_c":       min(temps)  if temps  else None,
    "peak_power_w":     max(powers) if powers else None,
    "min_power_w":      min(powers) if powers else None,
    "avg_power_w":      sum(powers)/len(powers) if powers else None,
    "peak_sclk_mhz":    max(sclks)  if sclks  else None,
    "min_sclk_mhz":     min(sclks)  if sclks  else None,
    "sclk_variance":    (max(sclks)-min(sclks)) if len(sclks)>1 else 0,
    "throttle_events":  throttle_events,
    "temp_samples":     len(temps),
}
with open(sys.argv[2], "w") as f:
    json.dump(data, f, indent=2)
print(json.dumps(data, indent=2))
PYEOF

  # Read parsed values
  local parsed_json="$RESULTS_DIR/mb_thermal_parsed_gpu${gpu_idx}.json"
  if [[ -f "$parsed_json" ]]; then
    local peak_temp min_sclk peak_sclk sclk_var throttle_events avg_power peak_power
    peak_temp=$(      python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('peak_temp_c') or 0)" 2>/dev/null || echo "0")
    peak_sclk=$(      python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('peak_sclk_mhz') or 0)" 2>/dev/null || echo "0")
    min_sclk=$(       python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('min_sclk_mhz') or 0)" 2>/dev/null || echo "0")
    sclk_var=$(       python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('sclk_variance') or 0)" 2>/dev/null || echo "0")
    throttle_events=$(python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('throttle_events') or 0)" 2>/dev/null || echo "0")
    avg_power=$(      python3 -c "import json; d=json.load(open('$parsed_json')); print(round(d.get('avg_power_w') or 0, 1))" 2>/dev/null || echo "0")
    peak_power=$(     python3 -c "import json; d=json.load(open('$parsed_json')); print(d.get('peak_power_w') or 0)" 2>/dev/null || echo "0")

    # Temperature under load
    if awk "BEGIN{exit !($peak_temp > 0 && $peak_temp <= $MB_THERMAL_MAX_TEMP)}"; then
      record_pass "thermal_stress_temperature_gpu${gpu_idx}" \
        "Peak ${peak_temp}°C ≤ ${MB_THERMAL_MAX_TEMP}°C (baseline was ${temp_before}°C)"
    elif awk "BEGIN{exit !($peak_temp > $MB_THERMAL_MAX_TEMP)}"; then
      record_fail "thermal_stress_temperature_gpu${gpu_idx}" \
        "Peak ${peak_temp}°C exceeds limit ${MB_THERMAL_MAX_TEMP}°C"
    else
      record_warn "thermal_stress_temperature_gpu${gpu_idx}" "Could not read peak temperature"
    fi

    # Temp/power correlation — temp should rise with power (basic sanity)
    if awk "BEGIN{exit !($peak_temp > $temp_before && $avg_power > $power_before)}"; then
      record_pass "thermal_stress_temp_power_correlation_gpu${gpu_idx}" \
        "Temp ${temp_before}→${peak_temp}°C, avg power ${avg_power}W correlate correctly"
    else
      record_warn "thermal_stress_temp_power_correlation_gpu${gpu_idx}" \
        "Temp ${temp_before}→${peak_temp}°C, power ${power_before}→${avg_power}W — check sensor"
    fi

    # Throttle flags
    if [[ "$throttle_events" -eq 0 ]]; then
      record_pass "thermal_stress_no_throttle_gpu${gpu_idx}" "No throttle events during ${MB_THERMAL_DURATION}s run"
    else
      record_warn "thermal_stress_throttle_gpu${gpu_idx}" \
        "$throttle_events throttle event(s) — GPU hit power/thermal limit"
    fi

    # SM clock stability (sclk variance < 200 MHz indicates no runaway throttling)
    if awk "BEGIN{exit !($sclk_var < 200)}"; then
      record_pass "thermal_stress_sclk_stable_gpu${gpu_idx}" \
        "SM clock variance ${sclk_var} MHz (peak=${peak_sclk} min=${min_sclk} MHz)"
    else
      record_warn "thermal_stress_sclk_stable_gpu${gpu_idx}" \
        "SM clock variance ${sclk_var} MHz — significant clock throttling (${min_sclk}→${peak_sclk} MHz)"
    fi

    # Power vs TDP — check power draw is in a sensible range (>50W means compute active)
    if awk "BEGIN{exit !($avg_power >= 50)}"; then
      record_pass "thermal_stress_power_draw_gpu${gpu_idx}" \
        "Average power draw ${avg_power}W (peak ${peak_power}W)"
    else
      record_warn "thermal_stress_power_draw_gpu${gpu_idx}" \
        "Average power ${avg_power}W — GPU may not be under full load"
    fi
  else
    record_warn "thermal_stress_metrics_parsed_gpu${gpu_idx}" "Could not parse monitoring data"
  fi

  # ECC post-thermal
  local ecc
  ecc=$(_mb_ecc_errors)
  if [[ "$ecc" -eq 0 ]]; then
    record_pass "thermal_stress_ecc_gpu${gpu_idx}" "0 uncorrected ECC errors after thermal stress"
  else
    record_fail "thermal_stress_ecc_gpu${gpu_idx}" "$ecc uncorrected ECC error(s) — thermal damage possible"
  fi
}

# =============================================================================
# 15 — Entry point
# =============================================================================
test_gpu_microbenchmarks() {
  section "15 · TARGETED GPU MICRO-BENCHMARKS"

  if ! cmd_exists hipcc; then
    for t in matmul_fp16 matmul_fp32 memory_stress pcie_stress thermal_stress; do
      record_skip "${t}_suite" "hipcc not found — cannot compile kernels"
    done
    rm -rf "$_MB_TMP"
    return
  fi

  # Determine which GPUs to test
  local gpu_count
  gpu_count=$(rocminfo 2>/dev/null | grep -c "Device Type.*GPU" | awk '{s+=$1} END{print s+0}' || echo "1")
  # Run on GPU 0 (and GPU N-1 for multi-GPU setups to catch per-slot differences)
  local gpus_to_test=(0)
  if [[ "$gpu_count" -gt 1 ]]; then
    gpus_to_test+=($(( gpu_count - 1 )))
  fi

  info "Micro-benchmark GPUs: ${gpus_to_test[*]} (of $gpu_count total)"

  for gpu in "${gpus_to_test[@]}"; do
    info "━━━ GPU $gpu ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 15.1 FP16 GEMM
    info "--- 15.1: FP16 GEMM (matrix core / tensor core utilization)"
    _run_matmul_fp16 "$gpu"

    # 15.2 FP32 GEMM
    info "--- 15.2: FP32 GEMM (CU / SM occupancy)"
    _run_matmul_fp32 "$gpu"

    # 15.3 Memory stress
    info "--- 15.3: Memory stress (HBM bandwidth + ECC + VRAM utilization)"
    _run_memory_stress "$gpu"

    # 15.4 PCIe stress
    info "--- 15.4: PCIe stress (H2D/D2H bandwidth + link health)"
    _run_pcie_stress "$gpu"

    # 15.5 Thermal stress
    info "--- 15.5: Thermal stress (max compute, power vs TDP, throttle)"
    _run_thermal_stress "$gpu"
  done

  # ── Summary CSV ─────────────────────────────────────────────────────────
  {
    echo "gpu,test,metric,value,unit"
    for gpu in "${gpus_to_test[@]}"; do
      # FP16
      local fp16_tflops
      fp16_tflops=$(grep "FP16_TFLOPS=" "$RESULTS_DIR/mb_fp16_gpu${gpu}.txt" 2>/dev/null | grep -oP "[0-9.]+")
      [[ -n "$fp16_tflops" ]] && echo "$gpu,matmul_fp16,throughput,$fp16_tflops,TFLOPS"

      # FP32
      local fp32_tflops
      fp32_tflops=$(grep "FP32_TFLOPS=" "$RESULTS_DIR/mb_fp32_gpu${gpu}.txt" 2>/dev/null | grep -oP "[0-9.]+")
      [[ -n "$fp32_tflops" ]] && echo "$gpu,matmul_fp32,throughput,$fp32_tflops,TFLOPS"

      # Memory bandwidth
      local mem_bw
      mem_bw=$(grep "MEM_BANDWIDTH_GBPS=" "$RESULTS_DIR/mb_mem_gpu${gpu}.txt" 2>/dev/null | grep -oP "[0-9.]+")
      [[ -n "$mem_bw" ]] && echo "$gpu,memory_stress,bandwidth,$mem_bw,GB/s"

      # PCIe H2D
      local pcie_h2d
      pcie_h2d=$(grep "PCIE_H2D_GBPS=" "$RESULTS_DIR/mb_pcie_gpu${gpu}.txt" 2>/dev/null | grep -oP "[0-9.]+")
      [[ -n "$pcie_h2d" ]] && echo "$gpu,pcie_stress,h2d_bandwidth,$pcie_h2d,GB/s"

      # PCIe D2H
      local pcie_d2h
      pcie_d2h=$(grep "PCIE_D2H_GBPS=" "$RESULTS_DIR/mb_pcie_gpu${gpu}.txt" 2>/dev/null | grep -oP "[0-9.]+")
      [[ -n "$pcie_d2h" ]] && echo "$gpu,pcie_stress,d2h_bandwidth,$pcie_d2h,GB/s"

      # Thermal peak
      if [[ -f "$RESULTS_DIR/mb_thermal_parsed_gpu${gpu}.json" ]]; then
        local peak_temp avg_pwr
        peak_temp=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/mb_thermal_parsed_gpu${gpu}.json')); print(d.get('peak_temp_c',''))" 2>/dev/null)
        avg_pwr=$(  python3 -c "import json; d=json.load(open('$RESULTS_DIR/mb_thermal_parsed_gpu${gpu}.json')); print(d.get('avg_power_w',''))"  2>/dev/null)
        [[ -n "$peak_temp" ]] && echo "$gpu,thermal_stress,peak_temp,$peak_temp,C"
        [[ -n "$avg_pwr"   ]] && echo "$gpu,thermal_stress,avg_power,$avg_pwr,W"
      fi
    done
  } > "$RESULTS_DIR/mb_benchmark_summary.csv"
  info "Micro-benchmark summary: $RESULTS_DIR/mb_benchmark_summary.csv"

  rm -rf "$_MB_TMP"
}
