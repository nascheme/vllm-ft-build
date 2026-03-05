# vllm-ft-build

*Running vLLM on the free-threaded version of Python is an EXPERIMENTAL
configuration and is not officially supported.  You may encounter bugs or
instability.  Also, this build does not support all of the models and options
that the official vLLM package supports.  Use with caution.*

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

#### Install OS packages (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git ccache numactl libnuma-dev g++ curl pkg-config libssl-dev \
    protobuf-compiler ca-certificates
```

#### Install CUDA developer files

Install the CUDA toolkit / developer packages appropriate for your GPU and
driver. Follow NVIDIA's installer instructions for your distribution and make
sure CUDA_HOME or CUDA_PATH points to the CUDA install (for example
/usr/local/cuda).

#### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

#### Create a free-threaded uv venv, install packages

Run uv to create the virtual environment and install packages. Pass "cuda",
"rocm", or "cpu" to select the PyTorch index:

```bash
./setup-venv.sh cuda
```

#### Clone required source repositories and apply patches

The build uses source checkouts for several packages. Clone the required repos
and apply patches using the included helper.  This will clone sources under
the `third_party` folder.  If a folder already exists, it is not cloned and
it is assumed the sources are already ready to build.

```bash
./clone-all.sh
```

#### Build source packages

The following helper will install harmony and then perform an editable install
of vllm.

```bash
./install-all.sh
```

The script `build_uv.py` mirrors the Dockerfile's editable vllm build and
reuses the same MAX_JOBS / NVCC_THREADS detection logic. It currently targets
CUDA only.


#### Quick test

When the build completes, run a quick smoke test:

```bash
uv run python -c "import vllm; print(vllm.__version__)"
```

#### Running vLLM

Run vLLM using uv:

```bash
./run_simple.sh
```

For Docker runs, use the run_docker_*.sh scripts listed above.

## Build details

The repository uses a mixture of pre-built wheels and source builds (via
local git checkouts). See `clone-all.sh` and patches/ for the repositories and
patches applied during the build.


### CUDA build

If you are building the CUDA backend, review TORCH_CUDA_ARCH_LIST and the
MAX_JOBS settings. Dockerfile.cuda shows the values used in the containerized
build.  A summary of nVidia compute hardware follows.

| CUDA Compute Capability | Architecture | Example GPUs | Notes |
|---|---|---|---|
| `7.5` | Turing | T4, RTX 2080 / 2080 Ti, RTX 2070 | Common inference GPU in older cloud deployments |
| `8.0` | Ampere (Datacenter) | A100 | Major AI training GPU; tensor cores with TF32 |
| `8.6` | Ampere (Consumer) | RTX 3090, RTX 3080, RTX 3070 | Widely used for local ML training |
| `8.9` | Ada Lovelace | L4, L40, RTX 4090, RTX 4080 | Popular for modern inference workloads |
| `9.0a` | Hopper | H100, H200 | Latest NVIDIA datacenter architecture with FP8 support |


### ROCm build

For the ROCm build, you will need to set PYTORCH_ROCM_ARCH to the correct
device.  For Dockerfile.rocm, use `--build-arg PYTORCH_ROCM_ARCH=<arch>`.  A
summary of recent hardware follows.

| GPU Generation | GFX Targets | Example Hardware |
|---|---|---|
| RDNA4 | `gfx1200`, `gfx1201` | RX 9060/9070 series |
| RDNA3 | `gfx1100`, `gfx1101`, `gfx1102` | RX 7900 XTX/XT, RX 7800 XT, RX 7700 XT, RX 7600 |
| RDNA3 APU | `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152` | Radeon 780M, 890M (Phoenix / Strix APUs) |
| RDNA2 | `gfx1030–gfx1036` | RX 6900 XT, RX 6800 XT, RX 6700 XT, RX 6600 XT |
| CDNA3 | `gfx942` | Instinct MI300X, MI300A |
| CDNA2 | `gfx90a` | Instinct MI250X, MI250, MI210 |
| CDNA1 | `gfx908` | Instinct MI100 |


### Compatibility notes

- Some vllm dependencies are not yet compatible with Python 3.14t (free-
  threaded).

- The uv-based host build assumes CUDA developer files and a working toolchain
  are present on the host. The Dockerfile stages runtime CUDA libraries to
  produce a minimal runtime image; this staging is unnecessary when running
  directly on a machine with a functioning CUDA runtime.

- build_uv.py currently targets CUDA only.


### Scripts

- build_docker.py - build Docker images (accepts --compute)
- build_uv.py - helper to run uv-based editable build of vLLM (CUDA)
- clone-repos.py - clone git repositories and apply patches
- setup-venv.sh - Create Python venv and install packages
- clone-all.sh - clone all needed source repositories
- install-all.sh - install packages from source repositories
