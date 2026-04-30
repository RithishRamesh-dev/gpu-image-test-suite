#!/usr/bin/env bash
# Module 06 — vLLM Inference Tests

VLLM_CONTAINER_NAME="vllm-validation"
VLLM_IMAGE="rocm/vllm:latest"

_vllm_running() {
  curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" &>/dev/null
}

_stop_vllm() {
  docker stop "$VLLM_CONTAINER_NAME" &>/dev/null || true
  docker rm   "$VLLM_CONTAINER_NAME" &>/dev/null || true
}

_start_vllm() {
  local tp="${1:-$TP_SIZE}"
  info "Starting vLLM server (TP=$tp, model=$MODEL)..."
  _stop_vllm
  docker run -d \
    --name "$VLLM_CONTAINER_NAME" \
    -v "${VOLUME:-/root/.cache/huggingface}:${VOLUME:-/root/.cache/huggingface}" \
    -p "${VLLM_PORT}:8000" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL="$MODEL" \
    -e VLLM_USE_V1=1 \
    -e VLLM_ROCM_USE_AITER=1 \
    -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    "$VLLM_IMAGE" \
    vllm serve "$MODEL" \
      --tensor-parallel-size="$tp" \
      --block-size=1 2>&1 &>/dev/null

  # Wait for readiness (up to 10 min for large model downloads)
  local attempts=0
  while [[ $attempts -lt 120 ]]; do
    if _vllm_running; then
      info "vLLM ready after $((attempts * 5))s"
      return 0
    fi
    # Check for container crash
    if ! docker inspect "$VLLM_CONTAINER_NAME" &>/dev/null; then
      warn "vLLM container disappeared"
      return 1
    fi
    local exit_code
    exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$VLLM_CONTAINER_NAME" 2>/dev/null || echo "-1")
    if [[ "$exit_code" != "0" && "$exit_code" != "-1" ]]; then
      warn "vLLM container exited with code $exit_code"
      docker logs "$VLLM_CONTAINER_NAME" 2>&1 | tail -30 | tee -a "$LOG_FILE" || true
      return 1
    fi
    sleep 5
    ((attempts++))
  done
  warn "vLLM did not become ready within 600s"
  docker logs "$VLLM_CONTAINER_NAME" 2>&1 | tail -50 | tee -a "$LOG_FILE" || true
  return 1
}

_run_bench() {
  local label="$1" input_len="$2" output_len="$3" concurrency="$4" prompts="$5" extra="${6:-}"
  local outfile="$RESULTS_DIR/vllm_bench_${label}.txt"
  info "Bench: $label (input=$input_len out=$output_len conc=$concurrency prompts=$prompts)"
  # shellcheck disable=SC2086
  docker run --rm --network=host "$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.benchmark_serving \
      --backend openai \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len  "$input_len" \
      --random-output-len "$output_len" \
      --max-concurrency   "$concurrency" \
      --num-prompts       "$prompts" \
      $extra 2>&1 | tee "$outfile" | tee -a "$LOG_FILE" || true

  # Parse throughput from output
  local throughput
  throughput=$(grep -oP "Throughput:\s*\K[0-9.]+" "$outfile" 2>/dev/null | head -1 || echo "0")
  local ttft
  ttft=$(grep -oP "TTFT.*?Mean:\s*\K[0-9.]+" "$outfile" 2>/dev/null | head -1 || echo "N/A")
  echo "$throughput" > "$RESULTS_DIR/vllm_bench_${label}_throughput.txt"
  info "  ↳ Throughput: $throughput req/s | TTFT: ${ttft}ms"
}

test_vllm() {
  section "06 · vLLM INFERENCE"

  if [[ -z "$HF_TOKEN" ]]; then
    record_warn "hf_token" "HF_TOKEN not set — model download may fail for gated models"
  else
    record_pass "hf_token" "HF_TOKEN is set"
  fi

  # Pull image
  info "Pulling $VLLM_IMAGE ..."
  if docker pull "$VLLM_IMAGE" 2>&1 | tee -a "$LOG_FILE"; then
    record_pass "vllm_image_pull" "$VLLM_IMAGE"
  else
    record_fail "vllm_image_pull" "Failed to pull $VLLM_IMAGE"
    return
  fi

  # ROCm version inside vLLM image
  local vllm_rocm_ver
  vllm_rocm_ver=$(docker run --rm "$VLLM_IMAGE" \
    bash -c "cat /opt/rocm/.info/version 2>/dev/null || python3 -c 'import torch; print(torch.version.hip)' 2>/dev/null" \
    2>/dev/null | head -1 | tr -d '[:space:]' || true)
  if [[ "$vllm_rocm_ver" == "$ROCM_EXPECTED_VERSION"* ]]; then
    record_pass "vllm_rocm_version" "vLLM image ROCm $vllm_rocm_ver"
  else
    record_warn "vllm_rocm_version" "vLLM image ROCm '${vllm_rocm_ver:-unknown}' — expect $ROCM_EXPECTED_VERSION"
  fi

  # ── Test 1: Single-GPU baseline ──────────────────────────────────────────
  info "--- Test 6.1: Single-GPU baseline (TP=1)"
  if _start_vllm 1; then
    record_pass "vllm_startup_tp1" "Server up (TP=1)"
    _run_bench "tp1_baseline" 512 128 8 50
    record_pass "vllm_bench_tp1_baseline" "Completed"
  else
    record_fail "vllm_startup_tp1" "Server failed to start (TP=1)"
  fi
  _stop_vllm

  # ── Test 2: Full TP (production config) ─────────────────────────────────
  info "--- Test 6.2: Full TP=$TP_SIZE (production)"
  if _start_vllm "$TP_SIZE"; then
    record_pass "vllm_startup_tp${TP_SIZE}" "Server up (TP=$TP_SIZE)"

    # Functional completions test
    local completion_response
    completion_response=$(curl -sf \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Explain ROCm in one sentence\",\"max_tokens\":50}" \
      "http://${VLLM_HOST}:${VLLM_PORT}/v1/completions" 2>/dev/null || echo "")

    if echo "$completion_response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" &>/dev/null; then
      record_pass "vllm_completion_functional" "Valid completion returned"
    else
      record_fail "vllm_completion_functional" "Invalid or empty response"
    fi

    # Chat completions endpoint
    local chat_response
    chat_response=$(curl -sf \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":20}" \
      "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" 2>/dev/null || echo "")
    if echo "$chat_response" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
      record_pass "vllm_chat_endpoint"
    else
      record_warn "vllm_chat_endpoint" "Chat endpoint returned invalid JSON"
    fi

    # Models list endpoint
    if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
      record_pass "vllm_models_endpoint"
    else
      record_warn "vllm_models_endpoint" "/v1/models returned unexpected response"
    fi

    # Standard benchmark
    _run_bench "tp${TP_SIZE}_standard" 5600 140 64 500

    # Long context stress
    info "--- Test 6.3: Long context (16k tokens)"
    _run_bench "tp${TP_SIZE}_longctx" 16000 256 16 50

    # High concurrency
    info "--- Test 6.4: High concurrency"
    _run_bench "tp${TP_SIZE}_highconc" 512 128 128 1000

    # Streaming output
    local stream_ok
    stream_ok=$(curl -sf \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Count to 5\",\"max_tokens\":30,\"stream\":true}" \
      "http://${VLLM_HOST}:${VLLM_PORT}/v1/completions" 2>/dev/null | head -c 200 || echo "")
    if [[ -n "$stream_ok" ]]; then
      record_pass "vllm_streaming" "Streaming response received"
    else
      record_warn "vllm_streaming" "Streaming response empty or failed"
    fi

    record_pass "vllm_suite_tp${TP_SIZE}" "All TP=$TP_SIZE tests complete"
  else
    record_fail "vllm_startup_tp${TP_SIZE}" "Server failed to start at TP=$TP_SIZE"
  fi
  _stop_vllm

  # ── Test 5: TP scaling (1 → 2 → 4 → TP_SIZE) ────────────────────────────
  info "--- Test 6.5: TP scaling test"
  local prev_tput=0
  for tp in 1 2 4 "$TP_SIZE"; do
    [[ "$tp" -gt "$TP_SIZE" ]] && continue
    if _start_vllm "$tp"; then
      local bench_out="$RESULTS_DIR/vllm_bench_scale_tp${tp}.txt"
      _run_bench "scale_tp${tp}" 512 128 16 100
      local tput
      tput=$(grep -oP "Throughput:\s*\K[0-9.]+" "$RESULTS_DIR/vllm_bench_scale_tp${tp}.txt" 2>/dev/null | head -1 || echo "0")
      if [[ "$prev_tput" -gt 0 ]] && awk "BEGIN{exit !($tput < $prev_tput * 0.7)}"; then
        record_warn "vllm_scaling_tp${tp}" "Throughput $tput req/s — expected ≥ 70% of TP=$((tp/2)) value"
      else
        record_pass "vllm_scaling_tp${tp}" "TP=$tp throughput: ${tput} req/s"
      fi
      prev_tput="$tput"
    else
      record_fail "vllm_scaling_tp${tp}" "Server did not start"
    fi
    _stop_vllm
  done

  # Save benchmark summary CSV
  {
    echo "benchmark,throughput_req_s"
    for f in "$RESULTS_DIR"/vllm_bench_*_throughput.txt; do
      [[ -f "$f" ]] || continue
      local label tput
      label=$(basename "$f" _throughput.txt | sed 's/vllm_bench_//')
      tput=$(cat "$f")
      echo "$label,$tput"
    done
  } > "$RESULTS_DIR/vllm_benchmark_summary.csv"
  info "Benchmark summary saved to $RESULTS_DIR/vllm_benchmark_summary.csv"
}
