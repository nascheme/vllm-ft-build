#!/bin/sh
#
# Run OpenAI compatible server inside Docker container (CPU build)

PORT=8889
IMAGE=vllm-freethreaded-cpu
MODEL=HuggingFaceTB/SmolLM2-360M-Instruct

cat <<EOF

****************************************************************************
*** Note that OpenAI compatible server listens on http://localhost:$PORT ***
****************************************************************************

Can test API using a command like:

curl http://localhost:$PORT/v1/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

EOF

# folder for HF_HOME
test -d cache || mkdir cache

# Upstream LD_PRELOADs tcmalloc and libiomp5 on x86_64.  Not done here by
# default (libiomp5 next to torch's libgomp means two OpenMP runtimes).  To try:
#
#   -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/app/.venv/lib/libiomp5.so
#
# VLLM_CPU_OMP_THREADS_BIND and VLLM_CPU_NUM_OF_RESERVED_CPU matter on NUMA boxes.

docker run \
    --rm \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    -p 127.0.0.1:$PORT:8000 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_CPU_KVCACHE_SPACE=8 \
    -e HF_HOME=/vllm-cache \
    -e PYTHON_GIL=0 \
    --mount=type=bind,src=`pwd`/cache,dst=/vllm-cache \
    --mount=type=bind,src=`pwd`/test,dst=/test \
    $IMAGE \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --enforce-eager
