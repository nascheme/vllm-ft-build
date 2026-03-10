#!/bin/sh
#
# Create venv and install Python packages.
#
# Usage ./setup-venv.sh [ cpu | cuda | rocm ]

set -eu

dev=$1

case $dev in
    cpu)
        url=https://download.pytorch.org/whl/cpu
        ;;
    rocm)
        url=https://download.pytorch.org/whl/rocm7.0
        ;;
    cuda)
        url=https://download.pytorch.org/whl/cu128
        ;;
    *)
        echo "Unknown compute device $dev"
        exit 1
esac

uv venv --python=3.14t
uv pip install -r requirements/torch.txt --extra-index-url "$url"
uv pip install -r requirements/common.txt
