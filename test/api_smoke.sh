#!/bin/bash
#
# Phase 4 step 3: start the OpenAI-compatible server, send one completion,
# shut it down.  Runs against the host venv or inside a runtime image:
#
#   ./test/api_smoke.sh [model]
#   ./test/run_in_docker.sh api_smoke.sh
#
# Overridable: PYTHON, HF_HOME, PORT, DTYPE, MEM_UTIL.  Uses urllib, not
# curl, which some runtime images lack.
set -eu

MODEL="${1:-HuggingFaceTB/SmolLM2-360M-Instruct}"
PORT="${PORT:-8123}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(mktemp -t vllm-api-smoke.XXXXXX.log)"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"

export HF_HOME="${HF_HOME:-$ROOT/cache}"
export PYTHON_GIL=0

# fp16 on the 6 GB Turing cards (no bf16 below sm_80); bf16 on CPU.
# On CPU --gpu-memory-utilization is the fraction of host RAM to reserve;
# the 0.92 default refuses to start on a busy box.
if "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    DTYPE="${DTYPE:-half}"
    MEM_UTIL="${MEM_UTIL:-0.55}"
else
    DTYPE="${DTYPE:-bfloat16}"
    MEM_UTIL="${MEM_UTIL:-0.3}"
fi

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "starting server on port $PORT (dtype $DTYPE, mem-util $MEM_UTIL, log: $LOG)"
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --dtype "$DTYPE" --port "$PORT" \
    --max-model-len 1024 --gpu-memory-utilization "$MEM_UTIL" \
    > "$LOG" 2>&1 &
SERVER_PID=$!

healthy() {
    "$PYTHON" - "$PORT" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

READY=0
for _ in $(seq 1 180); do
    if healthy; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited before becoming ready; last lines:"
        tail -200 "$LOG"
        exit 1
    fi
    sleep 1
done

if [ "$READY" != 1 ]; then
    echo "server did not become ready in time; last lines:"
    tail -200 "$LOG"
    exit 1
fi
echo "server is up"

TEXT=$("$PYTHON" - "$PORT" "$MODEL" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    print(json.load(r)["choices"][0]["text"])
PY
)

if [ -z "${TEXT// /}" ]; then
    echo "RESULT: FAIL — empty completion"
    exit 1
fi

echo "completion: ${TEXT}"
echo "RESULT: PASS"
