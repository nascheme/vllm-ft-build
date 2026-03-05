#!/bin/sh
#
# Run vllm-freethreaded-rocm container with docker.

PORT=8889
IMAGE=vllm-freethreaded-rocm
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

# You might need to use an environment variable like the following, depending
# on your GPU hardware.
#
#    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \

docker run \
    -it \
    --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
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
