#!/usr/bin/env bash
# PRWhisperer — start PrefixPilot + RAG and run one load test (vLLM must be on :8000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export RAG_BASE_URL="${RAG_BASE_URL:-http://127.0.0.1:8080}"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export PREFIX_PILOT_URL="${PREFIX_PILOT_URL:-http://127.0.0.1:8001}"
export CHROMA_DIR="${CHROMA_DIR:-$ROOT/chroma_prwhisperer}"

if ! curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
  echo "vLLM not responding on :8000 — start it first (use --enable-prefix-caching as you prefer)." >&2
  exit 1
fi
echo "[ok] vLLM reachable on :8000"

if [ ! -d "$CHROMA_DIR" ] && [ -n "${GITHUB_REPO:-}" ]; then
  echo "Ingest: GITHUB_REPO=$GITHUB_REPO"
  python3 "$ROOT/ingest.py" --chroma-dir "$CHROMA_DIR" --max-prs 100 --reset || true
fi

if ! curl -sf "http://127.0.0.1:8001/health" >/dev/null 2>&1; then
  echo "Starting PrefixPilot on :8001…"
  nohup python3 -m uvicorn prefix_pilot:app --host 0.0.0.0 --port 8001 >>"$ROOT/prefix_pilot.log" 2>&1 &
  for _ in 1 2 3 4 5; do
    sleep 1
    curl -sf "http://127.0.0.1:8001/health" && break || true
  done
else
  echo "[ok] PrefixPilot already on :8001"
fi

if ! curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
  echo "Starting rag_service on :8080…"
  nohup python3 -m uvicorn rag_service:app --host 0.0.0.0 --port 8080 >>"$ROOT/rag_service.log" 2>&1 &
  for _ in 1 2 3 4 5; do
    sleep 1
    curl -sf "http://127.0.0.1:8080/health" && break || true
  done
else
  echo "[ok] RAG service already on :8080"
fi

echo "Run load test…"
python3 "$ROOT/loadtest.py" --concurrency 50 --paraphrase-ratio 0.35 --duration 120 --phase-gap 5
echo "Logs: $ROOT/prefix_pilot.log $ROOT/rag_service.log"
