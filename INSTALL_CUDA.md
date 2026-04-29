# Install guide, host build, CUDA

See README.md for more details.  This is a short summary of how to do a host
build for CUDA.


## Prerequisites

- CUDA toolkit / developer files and a working compiler toolchain
- uv on your PATH.
- Optional but recommended: ccache.


## Install commands

```bash
./setup-venv.sh  # create uv venv
./clone-all.sh  # clone source repos
./install-all.sh  # install deps and install vllm from source
uv run python -c "import vllm; print(vllm.__version__)"
```
