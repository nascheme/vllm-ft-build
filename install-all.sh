#!/bin/sh

set -eu

# Build and install vllm.  You can edit this command if you want to pass
# different options, e.g. --arch=8.0
uv run ./build_uv.py
