#!/bin/sh
#
# Create venv and install Python packages.
#
# Usage ./setup-venv.sh [ cpu | cuda | cuda-nightly | rocm ]

set -eu

dev=$1
pre=""
torch_req=requirements/torch.txt

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
    cuda-nightly)
        url=https://download.pytorch.org/whl/nightly/cu128
        pre="--pre"
        torch_req=requirements/torch-nightly.txt
        ;;
    *)
        echo "Unknown compute device $dev"
        exit 1
esac

uv venv --python=3.14t
uv pip install $pre -r "$torch_req" --extra-index-url "$url"
uv pip install -r requirements/common.txt
