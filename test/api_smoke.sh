#!/bin/bash
#
# Phase 4 step 3: start the OpenAI-compatible server, send one completion,
# shut it down.  Uses fp16 and a small model because the GPUs here are 6 GB
# Turing cards (no bf16 below sm_80).
#
# Usage: ./test/api_smoke.sh [model]
set -eu

MODEL="${1:-HuggingFaceTB/SmolLM2-360M-Instruct}"
PORT="${PORT:-8123}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(mktemp -t vllm-api-smoke.XXXXXX.log)"

export HF_HOME="$ROOT/cache"
export PYTHON_GIL=0

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "starting server on port $PORT (log: $LOG)"
"$ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype half --port "$PORT" \
    --max-model-len 1024 --gpu-memory-utilization 0.55 \
    > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited before becoming ready; last lines:"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "server did not become ready in time; last lines:"
    tail -30 "$LOG"
    exit 1
fi
echo "server is up"

RESPONSE=$(curl -sf "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\",
         \"max_tokens\": 16, \"temperature\": 0}")

echo "$RESPONSE"
TEXT=$("$ROOT/.venv/bin/python" -c '
import json, sys
body = json.load(sys.stdin)
print(body["choices"][0]["text"])
' <<< "$RESPONSE")

if [ -z "${TEXT// /}" ]; then
    echo "RESULT: FAIL — empty completion"
    exit 1
fi

echo "completion: ${TEXT}"
echo "RESULT: PASS"
