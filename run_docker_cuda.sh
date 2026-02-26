#!/bin/sh
#
# Run OpenAI compatible server inside Docker container (CUDA build)

PORT=8889
IMAGE=vllm-freethreaded-cuda
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
test -e cache || mkdir cache

docker run \
    --rm \
    --gpus all \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    -p 127.0.0.1:$PORT:8000 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=/vllm-cache \
    -e PYTHON_GIL=0 \
    --mount=type=bind,src=`pwd`/cache,dst=/vllm-cache \
    --mount=type=bind,src=`pwd`/test,dst=/test \
    $IMAGE \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-model-len 4096 \
        --enforce-eager
