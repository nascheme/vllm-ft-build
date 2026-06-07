#!/bin/sh

set -eu

# Build and install vllm from source (and Triton too, when enabled via
# BUILD_TRITON=1 or by passing --triton).  Extra args are forwarded to
# build_uv.py, e.g.:
#   ./install-all.sh --arch=8.0
#   BUILD_TRITON=1 ./install-all.sh        # or: ./install-all.sh --triton
uv run ./build_uv.py "$@"
