# vllm-ft-build

*Running vLLM on the free-threaded version of Python is an EXPERIMENTAL
configuration and is not officially supported.  You may encounter bugs or
instability.  Also, this build does not support all of the models and options
that the offical vLLM package supports.  Use with caution.*

## Introduction

This repository contains scripts and helper tooling to build vLLM against the
free-threaded Python 3.14t runtime on Linux x86_64. Supported compute backends
are: CUDA, ROCm and CPU (note: host uv-based instructions currently only
cover CUDA).

Choose the path that matches your goal:

- I want a reproducible container image (recommended): see "Build - Docker"
- I want to build and run on the host (no Docker): see "Build - Host (uv)"

## Build

This section covers the two supported build workflows: building a Docker image
(using the provided build script) and building on the host using uv-managed
virtual environments.

### Docker image

A convenience script builds Docker images for the free-threaded Python
3.14t vLLM. The script accepts a --compute option to select the vLLM backend
and will look for a Dockerfile named Dockerfile.<compute> in the repository
root.

Supported compute values (default: cuda):

- cuda - uses Dockerfile.cuda
- rocm - uses Dockerfile.rocm
- cpu - uses Dockerfile.cpu

Usage examples:

```bash
# default (build CUDA image)
./build_docker.py

# explicit choices
./build_docker.py --compute=cuda
./build_docker.py --compute=rocm
./build_docker.py --compute=cpu
```

Notes:

- The script auto-detects system CPUs and RAM and sets MAX_JOBS and
  NVCC_THREADS for the build. You can override detection by setting
  MAX_JOBS/NVCC_THREADS in the environment when invoking Docker (or by
  passing your preferred values into build scripts that run inside the image).
- To override the CUDA architecture selection for CUDA builds set
  TORCH_CUDA_ARCH_LIST in the environment, for example:

```bash
TORCH_CUDA_ARCH_LIST=8.0 ./build_docker.py --compute=cuda
```

- The resulting image is tagged as vllm-freethreaded-<compute> (for example
  vllm-freethreaded-cuda).

To run the built image using the included helper script:

```bash
./run_docker_cuda.sh    # for CUDA
./run_docker_rocm.sh    # for ROCm
./run_docker_cpu.sh     # for CPU
```

### Host (uv) build - CUDA

These instructions reproduce the steps from Dockerfile.cuda but run on the
host. They assume you want the CUDA-enabled build (adjust as needed).

Prerequisites (host must provide):

- CUDA toolkit / developer files and a working compiler toolchain (gcc, g++,
  cmake). Ensure CUDA_HOME or CUDA_PATH points to your CUDA installation.
- Rust (rustup/cargo) - required to build tokenizers/safetensors from source.
- uv (https://github.com/astral-sh/uv) on your PATH.
- Optional but recommended: ccache (the scripts enable/use a CCACHE_DIR).

Substeps

### Install OS packages (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git ccache numactl libnuma-dev g++ curl pkg-config libssl-dev \
    protobuf-compiler ca-certificates
```

### Install CUDA developer files

Install the CUDA toolkit / developer packages appropriate for your GPU and
driver. Follow NVIDIA's installer instructions for your distribution and make
sure CUDA_HOME or CUDA_PATH points to the CUDA install (for example
/usr/local/cuda).

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

### Create a free-threaded uv venv

Create a free-threaded Python 3.14t environment (./.venv).

```bash
uv venv --python=3.14t
```

### Install CUDA PyTorch wheels

Install the CUDA-enabled PyTorch wheels inside the uv venv. Adjust the
--extra-index-url to match your CUDA version (Dockerfile.cuda uses cu128):

```bash
UV_HTTP_TIMEOUT=90 uv pip install \
    "torch>=2.10.0" "torchaudio>=2.10.0" "torchvision>=0.20.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### Install other Python dependencies

This will install all the dependency packages listed in the pyproject.toml
file.

```bash
uv sync
```

### Clone required source repositories and apply patches

The build uses source checkouts for several packages. Clone the required
repos and apply patches using the included helper:

```bash
uv run ./clone-repos.py --repo vllm --repo flash-attention --repo harmony \
    --repo safetensors --repo tokenizers
```

Note: build_uv.py does not automatically run clone-repos.py - run the clone
step before invoking the helper (you can add an automation flag later).

### Build source packages (non-editable)

```bash
uv pip install third_party/safetensors/bindings/python --no-build-isolation --no-deps
uv pip install third_party/tokenizers/bindings/python --no-build-isolation --no-deps
uv pip install third_party/harmony --no-build-isolation --no-deps
```

### Build and install editable vllm (helper)

A helper script build_uv.py mirrors the Dockerfile's editable vllm build and
reuses the same MAX_JOBS / NVCC_THREADS detection logic. It currently targets
CUDA only. Run it from the repository root using uv so the venv is used:

```bash
# run with defaults
uv run ./build_uv.py

# override TORCH arch / parallelism
uv run ./build_uv.py --arch=8.0 --max-jobs=8 --nvcc-threads=2
```

The helper will run the non-editable source builds (safetensors, tokenizers,
harmony) and then perform an editable install of vllm.

## Quick test

Once the build completes, run a quick smoke test without manually activating
the venv:

```bash
uv run python -c "import vllm; print(vllm.__version__)"
```

## Running vLLM

Run vLLM using uv:

```bash
uv run ./run_simple.py
```

For Docker runs, see run_docker.sh for an example invocation.

## Build details

If you are building the CUDA backend, review TORCH_CUDA_ARCH_LIST and the
MAX_JOBS settings. Dockerfile.cuda shows the values used in the containerized
build.

Common TORCH_CUDA_ARCH_LIST values:

- 7.5 - Turing (T4)
- 8.0 - Ampere (A100)
- 8.9 - Ada Lovelace (L4, L40, RTX 4090)
- 9.0a - Hopper (H100, H200)

The repository uses a mixture of pre-built wheels and source builds (via
local git checkouts). See git-repos.txt and patches/ for the repositories and
patches applied during the build.

Compatibility notes

- Some vllm dependencies are not yet compatible with Python 3.14t (free-
  threaded). See the "Build details" section and the repository issues for
  up-to-date notes.

- The uv-based host build assumes CUDA developer files and a working toolchain
  are present on the host. The Dockerfile stages runtime CUDA libraries to
  produce a minimal runtime image; this staging is unnecessary when running
  directly on a machine with a functioning CUDA runtime.

## Contributing / Next steps

build_uv.py currently targets CUDA only. We may extend it to accept --compute
to support ROCm/CPU on the host.

See the scripts in the repository for additional implementation details:
- build_docker.py - build Docker images (accepts --compute)
- build_uv.py - helper to run uv-based editable build (CUDA)
- clone-repos.py - clone git repositories and apply patches

