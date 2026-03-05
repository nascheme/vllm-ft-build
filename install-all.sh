#!/bin/sh

set -eu

uv pip install third_party/harmony --no-build-isolation --no-deps

# Build and install vllm.  You can edit this command if you want to pass
# different options, e.g. --arch=8.0
uv run ./build_uv.py
