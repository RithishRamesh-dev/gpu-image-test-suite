# ROCm 7.2.0 GPU Droplet — End-to-End Validation Suite

A comprehensive, modular test suite for validating AMD GPU droplets running
ROCm 7.2.0 across correctness, performance, observability, and production
readiness for inference workloads (vLLM).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Test Modules](#test-modules)
   - [00 · Provisioning](#00--provisioning)
   - [01 · OS + Kernel + Driver](#01--os--kernel--driver-baseline)
   - [02 · ROCm Stack Integrity](#02--rocm-stack-integrity)
   - [03 · GPU Enumeration & Topology](#03--gpu-enumeration--topology)
   - [04 · Observability Stack](#04--observability-stack)
   - [05 · Docker + GPU Runtime](#05--docker--gpu-runtime)
   - [06 · vLLM Inference](#06--vllm-inference)
   - [07 · RCCL Multi-GPU Communication](#07--rccl-multi-gpu-communication)
   - [08 · HIP / Compute Validation](#08--hip--compute-validation)
   - [09 · Power + Thermal](#09--power--thermal)
   - [10 · Permissions & Isolation](#10--permissions--isolation)
   - [11 · Package Consistency](#11--package-consistency)
   - [12 · Stress & Longevity](#12--stress--longevity)
   - [13 · Failure & Recovery](#13--failure--recovery)
   - [14 · Regression Comparison Matrix](#14--regression-comparison-matrix)
   - [15 · Targeted GPU Micro-Benchmarks](#15--targeted-gpu-micro-benchmarks)
6. [Output & Artifacts](#output--artifacts)
7. [Regression Testing Workflow](#regression-testing-workflow)
8. [Pass / Fail Criteria](#pass--fail-criteria)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)
11. [Adding New Tests](#adding-new-tests)

---

## Quick Start

```bash
# Full suite (all modules)
sudo bash run_tests.sh

# Skip time-intensive sections
SKIP_VLLM=true SKIP_STRESS=true sudo bash run_tests.sh

# Fast smoke test (provisioning + ROCm stack + GPU enum only)
SKIP_VLLM=true SKIP_RCCL=true SKIP_STRESS=true sudo bash run_tests.sh

# With regression baseline
BASELINE_RESULTS_DIR=/previous/run/results sudo bash run_tests.sh

# Full production validation with vLLM
HF_TOKEN=hf_xxx TP_SIZE=8 MODEL=deepseek-ai/DeepSeek-V3-0324 sudo bash run_tests.sh
```

Results land in `results/YYYYMMDD_HHMMSS/` automatically.

---

## Prerequisites

### Required

| Dependency    | Purpose                                | Install                                       |
|---------------|----------------------------------------|-----------------------------------------------|
| `bash ≥ 5`   | Test runner shell                      | Pre-installed on Ubuntu 22.04+                |
| `python3`     | JSON output, benchmark parsing         | `apt-get install python3`                     |
| `rocm-smi`    | GPU metrics                            | Installed with ROCm                           |
| `rocminfo`    | GPU enumeration                        | Installed with ROCm                           |
| `hipcc`       | HIP compilation                        | Installed with ROCm                           |
| `docker`      | Container + vLLM tests                 | [docs.docker.com/engine/install](https://docs.docker.com/engine/install/) |

### Recommended

| Dependency         | Purpose                                  | Install                                       |
|--------------------|------------------------------------------|-----------------------------------------------|
| `stress-ng`        | GPU stress testing (Module 12)           | `apt-get install stress-ng`                   |
| `lspci`            | PCIe topology checks                     | `apt-get install pciutils`                    |
| `amd-smi`          | Unified AMD management tool              | Included in ROCm 7.x                          |
| `mpirun`           | RCCL MPI mode test                       | `apt-get install openmpi-bin`                 |
| `rccl-tests`       | RCCL collective benchmarks               | See [Module 07](#07--rccl-multi-gpu-communication) |

### Optional (vLLM tests)

| Requirement        | Purpose                                  |
|--------------------|------------------------------------------|
| HuggingFace token  | Downloading gated models                 |
| ≥ 200 GB disk      | LLM model weights + Docker layers        |
| ≥ 8 GPUs           | Full tensor-parallel validation          |

---

## Architecture

```
rocm-validation/
├── run_tests.sh              # Entry point — orchestrates all modules
├── lib/
│   ├── 00_provisioning.sh    # Cloud-init, device presence
│   ├── 01_os_kernel.sh       # Kernel, dmesg, IOMMU, hugepages
│   ├── 02_rocm_stack.sh      # ROCm version, hipcc, library linkage
│   ├── 03_gpu_enumeration.sh # rocminfo, topology, VRAM, P2P
│   ├── 04_observability.sh   # Metrics exporter, do-agent, journald
│   ├── 05_docker_runtime.sh  # Docker daemon, GPU passthrough
│   ├── 06_vllm.sh            # vLLM server, completions, benchmarks
│   ├── 07_rccl.sh            # All-reduce, all-gather, reduce-scatter
│   ├── 08_hip_compute.sh     # HIP samples, rocBLAS SGEMM, rocFFT
│   ├── 09_power_thermal.sh   # Temperature, power draw, clock freq
│   ├── 10_permissions.sh     # /dev/kfd, /dev/dri, udev, setuid
│   ├── 11_package_consistency.sh  # dpkg version coherence
│   ├── 12_stress_longevity.sh    # stress-ng, longevity vLLM run
│   ├── 13_failure_recovery.sh   # Module reload, Docker restart
│   └── 14_regression_matrix.sh  # Cross-run comparison
│   └── 15_gpu_microbenchmarks.sh # FP16/FP32 GEMM, memory, PCIe, thermal
├── results/
│   └── YYYYMMDD_HHMMSS/
│       ├── test_run.log            # Full run log
│       ├── summary.json            # Machine-readable results
│       ├── regression_metrics.json # Numeric metrics for diffing
│       ├── regression_delta.json   # Diff vs baseline
│       ├── vllm_benchmark_summary.csv
│       ├── rccl_bandwidth_summary.csv
│       └── *.txt                   # Per-command captured output
└── README.md
```

Each module is independently `source`-able and can be run in isolation for
targeted debugging.

---

## Configuration

All configuration is via environment variables — no files to edit.

| Variable                   | Default                            | Description                                         |
|----------------------------|------------------------------------|-----------------------------------------------------|
| `ROCM_EXPECTED_VERSION`    | `7.2.0`                           | Expected ROCm version string                        |
| `BASELINE_VERSION`         | `7.0.2`                           | Label for the baseline in regression output         |
| `HF_TOKEN`                 | *(empty)*                         | HuggingFace API token for gated model download      |
| `MODEL`                    | `deepseek-ai/DeepSeek-V3-0324`   | vLLM model to serve                                 |
| `TP_SIZE`                  | `8`                               | Tensor parallel size (= number of GPUs)             |
| `VLLM_HOST`                | `127.0.0.1`                       | vLLM server host                                    |
| `VLLM_PORT`                | `8000`                            | vLLM server port                                    |
| `SKIP_VLLM`                | `false`                           | Skip vLLM module entirely                           |
| `SKIP_RCCL`                | `false`                           | Skip RCCL module entirely                           |
| `SKIP_STRESS`              | `false`                           | Skip stress/longevity module                        |
| `STRESS_DURATION`          | `600`                             | GPU stress test duration (seconds)                  |
| `VLLM_LONGEVITY_DURATION`  | `1800`                            | vLLM longevity test duration (seconds)              |
| `MAX_SAFE_TEMP`            | `85`                              | Max acceptable GPU temperature °C                   |
| `MAX_SAFE_POWER_W`         | `500`                             | Max acceptable GPU power draw (Watts)               |
| `RCCL_MIN_BW_GBPS`         | `100`                             | Minimum acceptable RCCL bus bandwidth (GB/s)        |
| `RCCL_TESTS_DIR`           | `/opt/rccl-tests/build`           | Path to rccl-tests binaries                         |
| `BASELINE_RESULTS_DIR`     | *(empty)*                         | Path to a previous run's `results/` dir for diffing |
| `FAIL_FAST`                | `false`                           | Abort on first failure                              |
| `SKIP_MICROBENCH`          | `false`                           | Skip module 15 micro-benchmarks                     |
| `MB_GEMM_FP16_MIN_TFLOPS`  | `100`                             | Minimum acceptable FP16 GEMM TFLOPS                 |
| `MB_GEMM_FP32_MIN_TFLOPS`  | `20`                              | Minimum acceptable FP32 GEMM TFLOPS                 |
| `MB_HBM_BW_MIN_GBPS`       | `800`                             | Minimum HBM bandwidth (GB/s)                        |
| `MB_PCIE_BW_MIN_GBPS`      | `20`                              | Minimum PCIe H2D/D2H bandwidth (GB/s)               |
| `MB_THERMAL_MAX_TEMP`      | `90`                              | Max GPU temperature during thermal stress (°C)      |
| `MB_GEMM_DURATION`         | `30`                              | Seconds to sustain GEMM loops for pipeline checks   |
| `MB_MEM_DURATION`          | `60`                              | Seconds for HBM bandwidth stress run                |
| `MB_PCIE_ITER`             | `200`                             | DMA transfer iteration count for PCIe test          |
| `MB_THERMAL_DURATION`      | `120`                             | Seconds for thermal stress kernel                   |

---

## Test Modules

### 00 · Provisioning

Validates the droplet booted cleanly.

| Check                   | What it tests                                              |
|-------------------------|------------------------------------------------------------|
| `cloud_init_status`     | `cloud-init status` must be `done`                        |
| `cloud_init_errors`     | Zero `ERROR`/`CRITICAL` lines in cloud-init journal       |
| `gpu_devices_at_boot`   | `/dev/dri/card*` present immediately after boot           |
| `kfd_device_present`    | `/dev/kfd` exists (required by all ROCm tools)            |
| `uptime_sanity`         | Uptime > 30s (detects reboot loops)                       |

---

### 01 · OS + Kernel + Driver Baseline

| Check                    | What it tests                                                  |
|--------------------------|----------------------------------------------------------------|
| `os_detected`            | OS name and version readable                                   |
| `os_supported`           | Ubuntu 22.04 or 24.04 (ROCm 7.2 supported platforms)          |
| `kernel_detected`        | `uname -r` readable                                            |
| `kernel_version_ok`      | Kernel ≥ 5.15 (ROCm 7.2 minimum)                              |
| `amdgpu_module_loaded`   | `amdgpu` present in `lsmod`                                   |
| `no_gpu_resets`          | Zero GPU reset events in dmesg                                 |
| `no_iommu_faults`        | Zero IOMMU faults in dmesg                                     |
| `no_vram_errors`         | Zero VRAM/ECC errors in dmesg                                  |
| `no_oom_events`          | No OOM-killer events                                           |
| `selinux_mode`           | SELinux not in Enforcing mode                                  |
| `hugepages_available`    | Huge pages configured (performance advisory)                   |

---

### 02 · ROCm Stack Integrity

The most critical module — catches broken or mis-versioned installs.

| Check                       | What it tests                                                |
|-----------------------------|--------------------------------------------------------------|
| `rocminfo_found`            | `rocminfo` in PATH                                           |
| `rocminfo_libs_ok`          | `ldd` shows no missing `.so` files                           |
| `rocm_version`              | Detected version == `ROCM_EXPECTED_VERSION`                  |
| `hipcc_found`               | `hipcc` in PATH and HIP version readable                     |
| `hip_version_aligned`       | HIP major.minor == ROCm major.minor                          |
| `rocm_smi_found`            | `rocm-smi` functional                                        |
| `amd_smi_found`             | `amd-smi` present (advisory)                                 |
| `rocminfo_agents`           | At least one Agent listed in `rocminfo`                      |
| `lib_libamdhip64`           | Core HIP runtime library present                             |
| `lib_librocblas`            | rocBLAS library present                                      |
| `no_mixed_rocm_versions`    | No `.so` files embedding a different ROCm version string     |
| `rocm_dir_exists`           | `/opt/rocm` directory exists                                 |

---

### 03 · GPU Enumeration & Topology

| Check                     | What it tests                                               |
|---------------------------|-------------------------------------------------------------|
| `gpu_count`               | ≥ 1 GPU detected via `rocminfo`                            |
| `gpu_count_matches_tp`    | GPU count matches `TP_SIZE` setting                         |
| `no_unknown_devices`      | No "UNKNOWN" entries in `rocm-smi --showhw`               |
| `pcie_link_width`         | All GPU PCIe links at expected width                        |
| `numa_affinity`           | NUMA topology info available                                |
| `vram_gpu{N}`             | Per-GPU: VRAM > 0 GB                                       |
| `p2p_access`              | P2P/xGMI info present for multi-GPU topologies              |
| `baseline_temp`           | GPU temperatures sane at idle (< 60°C)                     |

---

### 04 · Observability Stack

| Check                         | What it tests                                             |
|-------------------------------|-----------------------------------------------------------|
| `amd_metrics_exporter_service`| Service is `active`                                       |
| `metrics_endpoint_reachable`  | `http://127.0.0.1:5000/metrics` responds                 |
| `metrics_field_{gpu,temp,…}`  | Key metric families present in output                     |
| `metrics_gpu_count_match`     | Exporter reports same GPU count as `rocm-smi`            |
| `do_agent_service`            | DigitalOcean agent is running                             |
| `node_exporter`               | Prometheus node_exporter running (optional)               |
| `no_syslog_gpu_errors`        | No GPU-related errors in recent journal                   |

---

### 05 · Docker + GPU Runtime

| Check                          | What it tests                                            |
|--------------------------------|----------------------------------------------------------|
| `docker_installed`             | Docker binary in PATH                                    |
| `docker_daemon`                | `docker info` responds                                   |
| `docker_hello_world`           | Basic container execution works                          |
| `docker_gpu_container`         | GPU visible inside container via `/dev/kfd` + `/dev/dri` |
| `docker_rocm_version_in_container` | Container ROCm matches host                          |
| `docker_device_permissions`    | `/dev/kfd` readable inside container                     |
| `docker_ipc_mode`              | `--ipc=host` permitted (needed for multi-GPU NCCL)       |
| `docker_storage_driver`        | overlay2 in use                                          |
| `docker_disk_space`            | ≥ 200 GB free for model images                           |

---

### 06 · vLLM Inference

The most comprehensive real-world validation. Tests escalate from simple to stress.

#### Sub-tests

| Sub-test                       | Description                                                        |
|--------------------------------|--------------------------------------------------------------------|
| `vllm_image_pull`              | `rocm/vllm:latest` pulls successfully                              |
| `vllm_rocm_version`            | Image ROCm version matches expected                                |
| `vllm_startup_tp1`             | Server starts with TP=1 (single-GPU baseline)                      |
| `vllm_startup_tp{N}`           | Server starts at full `TP_SIZE`                                    |
| `vllm_completion_functional`   | `/v1/completions` returns valid JSON with content                  |
| `vllm_chat_endpoint`           | `/v1/chat/completions` works                                       |
| `vllm_models_endpoint`         | `/v1/models` responds                                              |
| `vllm_streaming`               | Streaming responses work                                           |
| `vllm_bench_tp1_baseline`      | Throughput benchmark at TP=1                                       |
| `vllm_bench_tp{N}_standard`    | Standard benchmark: 5600 in / 140 out / 64 concurrency / 500 req  |
| `vllm_bench_tp{N}_longctx`     | Long-context: 16000 in / 256 out                                   |
| `vllm_bench_tp{N}_highconc`    | High-concurrency: 128 concurrent requests                          |
| `vllm_scaling_tp{1,2,4,8}`     | Throughput at each TP level — checks linear scaling                |
| `vllm_longevity`               | 30-min continuous load — ≥ 95% request success rate               |

#### Metrics captured per benchmark

| Metric              | Stored in                                  |
|---------------------|--------------------------------------------|
| Request throughput  | `vllm_bench_<label>_throughput.txt`       |
| All benchmark JSON  | `vllm_bench_<label>.txt`                  |
| Summary CSV         | `vllm_benchmark_summary.csv`              |

---

### 07 · RCCL Multi-GPU Communication

Requires ≥ 2 GPUs. Automatically attempts to build `rccl-tests` if missing.

| Test                         | Operation                                      | Threshold             |
|------------------------------|------------------------------------------------|-----------------------|
| `rccl_all_reduce`            | AllReduce 16 GB                               | ≥ `RCCL_MIN_BW_GBPS` |
| `rccl_all_gather`            | AllGather 16 GB                               | ≥ `RCCL_MIN_BW_GBPS` |
| `rccl_reduce_scatter`        | ReduceScatter 16 GB                           | ≥ `RCCL_MIN_BW_GBPS` |
| `rccl_broadcast`             | Broadcast 16 GB                               | advisory              |
| `rccl_reduce`                | Reduce 16 GB                                  | advisory              |
| `rccl_all_to_all`            | AllToAll 1–8 GB sweep                         | advisory              |
| `rccl_all_reduce_fp16/bf16/fp32` | Data-type coverage                        | advisory              |
| `rccl_mpi_available`         | MPI multi-process mode (if `mpirun` present)  | advisory              |
| `rccl_library_found`         | `librccl.so` located                          | required              |

Bandwidth summary saved to `rccl_bandwidth_summary.csv`.

---

### 08 · HIP / Compute Validation

| Check                        | What it tests                                               |
|------------------------------|-------------------------------------------------------------|
| `hip_samples`                | All ROCm sample binaries exit 0                            |
| `hip_minimal_compile_run`    | Fallback: compile + run a vector-add kernel via `hipcc`    |
| `rocblas_sgemm`              | 4096×4096 SGEMM via `rocblas-bench`                       |
| `rocfft_smoke`               | FFT correctness via `rocfft_rider`                         |

---

### 09 · Power + Thermal

| Check                    | What it tests                                              |
|--------------------------|------------------------------------------------------------|
| `temp_gpu{N}`            | Per-GPU idle temperature < `MAX_SAFE_TEMP` (default 85°C) |
| `no_thermal_throttle`    | No throttle events in RAS info                            |
| `power_gpu{N}`           | Per-GPU power draw ≤ `MAX_SAFE_POWER_W` (default 500 W)  |
| `clock_frequencies`      | All GPU clocks non-zero                                    |

---

### 10 · Permissions & Isolation

| Check                        | What it tests                                             |
|------------------------------|-----------------------------------------------------------|
| `kfd_permissions`            | `/dev/kfd` mode 660 or 666                               |
| `user_in_gpu_group`          | Current user in `render`/`video`/`kfd` group             |
| `dri_permissions`            | `/dev/dri/*` accessible                                  |
| `udev_rules_present`         | ROCm udev rules installed                                |
| `rocm_bin_not_world_writable`| No world-writable files in `/opt/rocm/bin`               |
| `no_setuid_rocm`             | No setuid binaries in `/opt/rocm`                        |

---

### 11 · Package Consistency

| Check                          | What it tests                                           |
|--------------------------------|---------------------------------------------------------|
| `no_mixed_rocm_packages`       | All installed ROCm packages on same major.minor version |
| `rocm_package_versions_correct`| No packages on unexpected ROCm branch                  |
| `rocm_package_count`           | Count of installed ROCm packages (informational)        |
| `no_broken_packages`           | No half-installed (`iH`/`iF`) ROCm packages            |
| `pkg_{rocm-dev,hip-base,...}`  | Key packages present                                    |
| `rocm_version_file`            | `/opt/rocm/.info/version` readable                      |
| `pytorch_rocm`                 | PyTorch with ROCm/HIP available (optional)              |

---

### 12 · Stress & Longevity

| Check                      | What it tests                                                |
|----------------------------|--------------------------------------------------------------|
| `stress_temperature`       | Peak temp ≤ 90°C during `STRESS_DURATION`-second GPU stress |
| `stress_no_gpu_resets`     | Zero GPU resets after stress                                 |
| `stress_gpus_survive`      | All GPUs still enumerated after stress                       |
| `rocm_bandwidth_test`      | Memory bandwidth consistent                                  |
| `vllm_longevity`           | ≥ 95% success rate over `VLLM_LONGEVITY_DURATION` seconds   |
| `no_vram_leak`             | VRAM returns to baseline after workload (< 500 MB delta)     |

---

### 13 · Failure & Recovery

| Check                              | What it tests                                          |
|------------------------------------|--------------------------------------------------------|
| `module_reload`                    | `amdgpu` module unload/reload restores devices        |
| `docker_daemon_restart`            | Docker daemon restart does not break GPU access        |
| `gpu_accessible_post_docker_restart` | GPU visible in containers after Docker restart      |
| `metrics_exporter_restart`         | amd-metrics-exporter restarts cleanly                  |
| `vram_alloc_free`                  | VRAM alloc + explicit free returns to baseline         |
| `post_recovery_gpu_count`          | All GPUs present after all recovery tests              |

---

### 14 · Regression Comparison Matrix

Generates a `regression_metrics.json` snapshot of:

- GPU count
- ROCm version string
- RCCL all-reduce bandwidth (GB/s)
- vLLM throughput (req/s)
- Memory bandwidth (GB/s)
- Peak temperature (°C)

When `BASELINE_RESULTS_DIR` is set, produces a `regression_delta.json` and
records PASS/FAIL/WARN per metric against configurable thresholds:

| Metric                   | Threshold                                |
|--------------------------|------------------------------------------|
| RCCL bandwidth           | Current ≥ 95% of baseline               |
| vLLM throughput          | Current ≥ 95% of baseline               |
| Memory bandwidth         | Current ≥ 95% of baseline               |
| Peak temperature         | Current ≤ 110% of baseline              |

---

### 15 · Targeted GPU Micro-Benchmarks

Compiled HIP kernels that stress individual hardware pipelines in isolation.
Each test self-compiles via `hipcc` at runtime — no pre-built binaries needed.
Tests run on GPU 0, and also on GPU N-1 for multi-GPU droplets (catches
per-slot hardware differences).

#### 15.1 — `cuda_matmul_fp16`: FP16 GEMM / Matrix Core Utilization

Runs a sustained 8192×8192×8192 `rocblas_hgemm` loop, targeting the matrix
(tensor) cores.

| Check | Metric | Pass criterion |
|---|---|---|
| `matmul_fp16_compile_gpu{N}` | hipcc compilation | exit 0 |
| `matmul_fp16_throughput_gpu{N}` | TFLOPS | ≥ `MB_GEMM_FP16_MIN_TFLOPS` (default 100) |
| `matmul_fp16_pipeline_active_gpu{N}` | GPU utilization % | ≥ 80% (matrix cores engaged) |
| `matmul_fp16_sm_clock_gpu{N}` | SM clock during run | > 0 MHz |
| `matmul_fp16_ecc_gpu{N}` | ECC uncorrected errors | 0 |

Additional output: arithmetic intensity (FLOPs/byte), elapsed time.

#### 15.2 — `cuda_matmul_fp32`: FP32 GEMM / SM Occupancy

Runs a sustained 8192×8192×8192 `rocblas_sgemm` loop, targeting FP32 CUDA
cores. Also queries `hipDeviceProp_t` for CU count and maximum wavefront
capacity to estimate theoretical occupancy.

| Check | Metric | Pass criterion |
|---|---|---|
| `matmul_fp32_compile_gpu{N}` | hipcc compilation | exit 0 |
| `matmul_fp32_throughput_gpu{N}` | TFLOPS | ≥ `MB_GEMM_FP32_MIN_TFLOPS` (default 20) |
| `matmul_fp32_cu_count_gpu{N}` | CU count from device props | > 0 |
| `matmul_fp32_sm_occupancy_gpu{N}` | GPU utilization % | ≥ 70% |
| `matmul_fp32_power_signal_gpu{N}` | Peak power during run | > 50 W |
| `matmul_fp32_ecc_gpu{N}` | ECC uncorrected errors | 0 |

#### 15.3 — `memory_stress`: VRAM Fill + HBM Sustained Bandwidth + ECC

**Phase 1 (VRAM fill):** fills 90% of available VRAM with a parallel write
kernel — every memory cell is written before Phase 2 begins.

**Phase 2 (streaming RW):** a streaming `copy` kernel (`dst[i] = src[i] + 1`)
runs for `MB_MEM_DURATION` seconds, saturating both HBM read and write buses.
Reports total bytes moved and achieved bandwidth.

**Phase 3 (integrity check):** 1024-element host-side sample verifies no silent
data corruption occurred.

| Check | Metric | Pass criterion |
|---|---|---|
| `memory_stress_compile_gpu{N}` | hipcc compilation | exit 0 |
| `memory_stress_bandwidth_gpu{N}` | HBM bandwidth | ≥ `MB_HBM_BW_MIN_GBPS` (default 800 GB/s) |
| `memory_stress_vram_fill_gpu{N}` | Full VRAM write | PASS |
| `memory_stress_integrity_gpu{N}` | Data corruption check | 0 errors |
| `memory_stress_vram_utilization_gpu{N}` | Peak VRAM used % | ≥ 80% |
| `memory_stress_mem_clock_gpu{N}` | Memory clock during run | > 0 MHz |
| `memory_stress_ecc_gpu{N}` | New uncorrected ECC errors | 0 (FAIL if > 0 — indicates VRAM fault) |

#### 15.4 — `pcie_stress`: Host↔GPU DMA Bandwidth + Link Health

Three transfer modes using 512 MB pinned-memory buffers:
- **H2D** (`hipMemcpy` host→device, `MB_PCIE_ITER` iterations)
- **D2H** (`hipMemcpy` device→host, `MB_PCIE_ITER` iterations)
- **Bidirectional** (async H2D + D2H simultaneously on separate streams)

Also reads PCIe link state and replay counters from `lspci -vvv`.

| Check | Metric | Pass criterion |
|---|---|---|
| `pcie_stress_compile_gpu{N}` | hipcc compilation | exit 0 |
| `pcie_stress_h2d_bw_gpu{N}` | H2D bandwidth | ≥ `MB_PCIE_BW_MIN_GBPS` (default 20 GB/s) |
| `pcie_stress_d2h_bw_gpu{N}` | D2H bandwidth | ≥ `MB_PCIE_BW_MIN_GBPS` |
| `pcie_stress_bidir_bw_gpu{N}` | Bidirectional bandwidth | informational |
| `pcie_stress_symmetry_gpu{N}` | H2D vs D2H ratio | within 20% of each other |
| `pcie_stress_link_state_gpu{N}` | PCIe speed + width | informational (e.g. 16GT/s x16) |
| `pcie_stress_link_gen_gpu{N}` | PCIe generation | WARN if < Gen4 |
| `pcie_stress_replay_counter_gpu{N}` | PCIe replay count | 0 (FAIL if > 0 — link errors) |
| `pcie_stress_smi_throughput_gpu{N}` | rocm-smi PCIe BW | informational |

#### 15.5 — `thermal_stress`: Max-Compute Thermals, Power vs TDP, Throttle Flags

A register-blocked FMA kernel (`acc = acc * a + b`, 8 independent accumulators,
64× unrolled, 8192 inner reps) runs for `MB_THERMAL_DURATION` seconds at
maximum SM occupancy. Temperature, power, SM clock, and throttle flags are
sampled every 2 seconds throughout.

| Check | Metric | Pass criterion |
|---|---|---|
| `thermal_stress_compile_gpu{N}` | hipcc compilation | exit 0 |
| `thermal_stress_compute_sustained_gpu{N}` | TFLOPS over full duration | informational |
| `thermal_stress_temperature_gpu{N}` | Peak temperature | ≤ `MB_THERMAL_MAX_TEMP` (default 90°C) |
| `thermal_stress_temp_power_correlation_gpu{N}` | Temp rises with power | both increase vs idle baseline |
| `thermal_stress_no_throttle_gpu{N}` | Throttle events | 0 (WARN if > 0) |
| `thermal_stress_sclk_stable_gpu{N}` | SM clock variance | < 200 MHz (no runaway throttling) |
| `thermal_stress_power_draw_gpu{N}` | Average power draw | > 50 W (compute active) |
| `thermal_stress_ecc_gpu{N}` | ECC uncorrected errors | 0 (FAIL if > 0 — thermal damage risk) |

Per-run JSON with peak/min/avg temp, power, clock, and throttle event count
saved to `mb_thermal_parsed_gpu{N}.json`.

#### Micro-benchmark Summary CSV

All numeric results aggregated to `mb_benchmark_summary.csv`:

```
gpu,test,metric,value,unit
0,matmul_fp16,throughput,312.4,TFLOPS
0,matmul_fp32,throughput,82.1,TFLOPS
0,memory_stress,bandwidth,1247.3,GB/s
0,pcie_stress,h2d_bandwidth,24.8,GB/s
0,pcie_stress,d2h_bandwidth,23.9,GB/s
0,thermal_stress,peak_temp,78.0,C
0,thermal_stress,avg_power,485.2,W
```

#### Metrics coverage matrix

| Hardware unit | FP16 GEMM | FP32 GEMM | Memory stress | PCIe stress | Thermal stress |
|---|:---:|:---:|:---:|:---:|:---:|
| ECC counters | ✓ | ✓ | ✓ | — | ✓ |
| SM occupancy / GPU util | ✓ | ✓ | — | — | (implicit) |
| Matrix/tensor pipeline | ✓ | — | — | — | — |
| FP32 pipeline activity | — | ✓ | — | — | ✓ |
| HBM bandwidth | — | — | ✓ | — | — |
| VRAM utilization | — | — | ✓ | — | — |
| Data integrity | — | — | ✓ | — | — |
| Memory clock | — | — | ✓ | — | — |
| SM (shader) clock | ✓ | — | — | — | ✓ |
| Core temperature | ✓ | ✓ | ✓ | — | ✓ |
| Power draw | ✓ | ✓ | — | — | ✓ |
| Throttle flags | — | — | — | — | ✓ |
| Temp/power correlation | — | — | — | — | ✓ |
| PCIe H2D/D2H throughput | — | — | — | ✓ | — |
| PCIe link gen/width | — | — | — | ✓ | — |
| PCIe replay counters | — | — | — | ✓ | — |

Every run creates a timestamped directory under `results/`:

```
results/20250120_143022/
├── test_run.log                       # Full verbose log of all output
├── summary.json                       # Machine-readable pass/fail/warn/skip per test
├── regression_metrics.json            # Numeric metrics for future comparison
├── regression_delta.json              # Diff vs BASELINE_RESULTS_DIR (if set)
├── vllm_benchmark_summary.csv         # Throughput per benchmark config
├── rccl_bandwidth_summary.csv         # Bandwidth per collective operation
├── metrics_snapshot.txt               # Full /metrics endpoint at test time
├── os_release.txt                     # /etc/os-release
├── rocminfo.txt                       # Full rocminfo output
├── rocm_smi.txt                       # rocm-smi output
├── rocm_smi_showtopo.txt              # Topology output
├── dmesg_amdgpu.txt                   # amdgpu-related dmesg entries
├── rocm_packages.txt                  # Installed ROCm package list
├── vllm_bench_*.txt                   # Raw benchmark output per run
└── rccl_*.txt                         # Raw RCCL test output per operation
```

### `summary.json` schema

```json
{
  "timestamp": "2025-01-20T14:30:22+00:00",
  "rocm_expected": "7.2.0",
  "baseline": "7.0.2",
  "host": "gpu-droplet-01",
  "totals": { "pass": 82, "fail": 0, "warn": 3, "skip": 5 },
  "tests": {
    "rocm_version": { "status": "PASS", "detail": "Detected 7.2.0 == expected 7.2.0" },
    "vllm_completion_functional": { "status": "PASS", "detail": "Valid completion returned" }
  }
}
```

---

## Regression Testing Workflow

```bash
# 1. Run baseline (ROCm 7.0.2) and keep results
ROCM_EXPECTED_VERSION=7.0.2 bash run_tests.sh
# Results saved to e.g. results/20250110_120000/

# 2. Upgrade droplet to ROCm 7.2.0

# 3. Run new suite with baseline comparison
ROCM_EXPECTED_VERSION=7.2.0 \
BASELINE_RESULTS_DIR=results/20250110_120000 \
bash run_tests.sh
```

The regression delta table will print inline and save to `regression_delta.json`.

---

## Pass / Fail Criteria

### ✅ Production Ready

- `FAIL` count = 0
- ROCm version correct (`rocm_version` = PASS)
- All GPUs enumerated (`gpu_count` = PASS)
- vLLM completes functional test (`vllm_completion_functional` = PASS)
- RCCL completes without hangs (`rccl_all_reduce` = PASS)
- Metrics pipeline working (`metrics_endpoint_reachable` = PASS)
- No crashes under stress (`stress_gpus_survive` = PASS)
- No performance regression (`regression_*` all PASS or SKIP)

### ⚠️ Conditional Pass

Any `WARN` items should be reviewed. Common acceptable warnings:

- `hugepages_available` — advisory performance suggestion
- `vllm_rocm_version` — if using a bleeding-edge container
- `docker_disk_space` — if not planning to run large models
- `rccl_mpi_available` — if MPI is not part of deployment

### ❌ Blocking Failures

Any single `FAIL` in these checks requires resolution before production:

- `rocm_version`, `kfd_device_present`, `amdgpu_module_loaded`
- `gpu_count`, `no_gpu_resets`, `no_vram_errors`
- `vllm_startup_tp{N}`, `rccl_all_reduce`
- `no_broken_packages`, `no_mixed_rocm_packages`

---

## CI/CD Integration

### GitHub Actions example

```yaml
name: ROCm GPU Validation
on:
  workflow_dispatch:
    inputs:
      droplet_ip:
        description: 'GPU Droplet IP'
        required: true

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Copy suite to droplet
        run: |
          rsync -r rocm-validation/ root@${{ inputs.droplet_ip }}:/root/rocm-validation/

      - name: Run smoke test
        run: |
          ssh root@${{ inputs.droplet_ip }} \
            "SKIP_VLLM=true SKIP_STRESS=true bash /root/rocm-validation/run_tests.sh"

      - name: Fetch results
        if: always()
        run: |
          scp -r root@${{ inputs.droplet_ip }}:/root/rocm-validation/results ./validation-results

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: validation-results/

      - name: Check for failures
        run: |
          python3 -c "
          import json, glob, sys
          files = sorted(glob.glob('validation-results/*/summary.json'))
          if not files: sys.exit(1)
          data = json.load(open(files[-1]))
          if data['totals']['fail'] > 0:
              print(f\"FAILED: {data['totals']['fail']} check(s)\")
              sys.exit(1)
          print(f\"PASSED: {data['totals']['pass']} checks\")
          "
```

### Exit codes

| Code | Meaning                             |
|------|-------------------------------------|
| `0`  | All checks passed                   |
| `1`  | One or more `FAIL` checks           |

---

## Troubleshooting

### "rocminfo: No such file or directory"

ROCm is not installed or `/opt/rocm/bin` is not in `PATH`.

```bash
export PATH="/opt/rocm/bin:$PATH"
# or
apt-get install rocminfo
```

### "GPU not visible inside container"

```bash
# Check device permissions
ls -la /dev/kfd /dev/dri/
# Add user to render group
usermod -aG render,video $(whoami)
# Verify udev rules
ls /etc/udev/rules.d/ | grep rocm
```

### vLLM server times out

Check if the HuggingFace model download is still running:

```bash
docker logs vllm-validation -f
```

Model weights for DeepSeek-V3 are ~685 GB. Ensure `VOLUME` points to a fast
NVMe mount with sufficient space.

### RCCL test hangs

```bash
# Enable debug output
NCCL_DEBUG=INFO RCCL_DEBUG=INFO ./build/all_reduce_perf -b 16G -e 16G -g 8
# Check xGMI/NVLink topology
rocm-smi --showtopo
```

### "mixed ROCm version" warning

```bash
dpkg -l | grep rocm | awk '{print $2, $3}' | sort
# Remove old packages
apt-get remove --purge rocm-*-6.* hip-*-6.*
apt-get autoremove
```

### Temperature check failing before stress

Could indicate inadequate cooling or a pre-existing workload:

```bash
rocm-smi --showtemp --showfan
# Check if another process is using the GPU
fuser /dev/dri/renderD*
```

---

## Adding New Tests

1. Create or edit a module file in `lib/`:

```bash
# lib/15_my_custom_test.sh
test_my_custom() {
  section "15 · MY CUSTOM TEST"

  if some_condition; then
    record_pass "my_check" "details here"
  else
    record_fail "my_check" "what went wrong"
  fi
}
```

2. Source it and call it from `run_tests.sh`:

```bash
source "$LIB_DIR/15_my_custom_test.sh"
# ...
test_my_custom
```

Available helper functions:

| Function        | Usage                                          |
|-----------------|------------------------------------------------|
| `record_pass`   | `record_pass "test_name" "optional detail"`    |
| `record_fail`   | `record_fail "test_name" "what failed"`        |
| `record_warn`   | `record_warn "test_name" "advisory message"`   |
| `record_skip`   | `record_skip "test_name" "reason"`             |
| `capture`       | `capture "output_name" command args`           |
| `cmd_exists`    | `cmd_exists rocminfo && ...`                   |
| `info/ok/fail/warn/skip` | Logging helpers with colour             |

---

## License

Apache 2.0. Contributions welcome — especially new test cases for ROCm
subsystems not yet covered (MIOpen, rocRAND, hipSPARSE, etc.).
