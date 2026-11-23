#!/usr/bin/env bash
# STEP 5 — Serve the merged model with vLLM's OpenAI-compatible server.
# vLLM gives you continuous batching + PagedAttention + token streaming for free.
# Install on the GPU box:  pip install vllm
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-outputs/merged}"
SERVED_NAME="${SERVED_MODEL_NAME:-domain-llm}"
PORT="${VLLM_PORT:-8000}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --disable-log-requests
# TensorRT-LLM alternative: build an engine with trtllm-build and serve via
# Triton — same gateway sits in front either way (see GUIDE §serving).
