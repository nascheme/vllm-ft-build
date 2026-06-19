#!/bin/sh
#
# Clone the source repos needed for the build.
#
# Triton is OPTIONAL: it is cloned (and later built from source with the
# free-threading patches in patches/triton/) only when enabled, via either:
#   BUILD_TRITON=1 ./clone-all.sh      # env var
#   ./clone-all.sh --triton            # flag
set -eu

triton="${BUILD_TRITON:-0}"
for arg in "$@"; do
    case "$arg" in
        --triton) triton=1 ;;
        --no-triton) triton=0 ;;
    esac
done

set -- --repo vllm --repo flash-attention --repo tokenizers
[ "$triton" = "1" ] && set -- "$@" --repo triton

uv run ./clone-repos.py "$@"
