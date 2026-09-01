#!/bin/sh
#
# Run one of the test/ scripts inside a built runtime image.
#
#   ./test/run_in_docker.sh verify_build.py
#   ./test/run_in_docker.sh threaded_stress.py repeat --iters 3
#   ./test/run_in_docker.sh api_smoke.sh
#
# IMAGE selects the backend (default: the CPU image); it also decides which
# devices get passed through:
#
#   IMAGE=vllm-freethreaded-cuda ./test/run_in_docker.sh verify_build.py
#   IMAGE=vllm-freethreaded-rocm ./test/run_in_docker.sh threaded_stress.py single
#
# test/ and git-repos.txt are mounted under /app so verify_build.py's ROOT
# resolves to the image's third_party/vllm.
set -eu

IMAGE="${IMAGE:-vllm-freethreaded-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <script> [args...]" >&2; exit 2; }

SCRIPT="$1"
shift

case "$SCRIPT" in
    *.sh) CMD="/app/test/$SCRIPT" ;;
    *)    CMD="python /app/test/$SCRIPT" ;;
esac

DEVICE_ARGS=""
case "$IMAGE" in
    *cuda*) DEVICE_ARGS="--gpus all" ;;
    *rocm*) DEVICE_ARGS="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render" ;;
esac

test -d "$ROOT/cache" || mkdir "$ROOT/cache"

# shellcheck disable=SC2086
exec docker run --rm -i \
    $DEVICE_ARGS \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    -e PYTHON_GIL=0 \
    -e HF_HOME=/vllm-cache \
    -e VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}" \
    --mount=type=bind,src="$ROOT/cache",dst=/vllm-cache \
    --mount=type=bind,src="$ROOT/test",dst=/app/test,readonly \
    --mount=type=bind,src="$ROOT/git-repos.txt",dst=/app/git-repos.txt,readonly \
    -w /app \
    "$IMAGE" \
    $CMD "$@"
