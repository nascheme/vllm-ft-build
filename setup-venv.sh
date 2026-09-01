#!/bin/bash
#
# Create venv and install Python packages.
#
# Usage ./setup-venv.sh [-p <py_version>] cpu | cuda | cuda-nightly | rocm

set -eu

py_version=3.14t

args=$(getopt -o p: -- "$@")
eval set -- "$args"
while true; do
    case $1 in
        -p) py_version=$2; shift 2 ;;
        --) shift; break ;;
    esac
done

dev="${1:-cuda}"
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
        url=https://download.pytorch.org/whl/cu130
        ;;
    cuda-nightly)
        url=https://download.pytorch.org/whl/nightly/cu130
        pre="--pre"
        torch_req=requirements/torch-nightly.txt
        ;;
    *)
        echo "Unknown compute device $dev"
        exit 1
esac

uv venv --python="$py_version"
uv pip install $pre -r "$torch_req" --extra-index-url "$url"
uv pip install -r requirements/common.txt
