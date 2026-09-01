# Install guide, host build, CUDA

See README.md for more details.  This is a short summary of how to do a host
build for CUDA.


## Prerequisites

- CUDA toolkit / developer files and a working compiler toolchain.  vLLM
  v0.28.0 wants C++20, so GCC >= 11.3.  This build targets **CUDA 13.1** and
  installs the `cu130` PyTorch wheels (`torch==2.13.0`, `torchvision==0.28.0`,
  `torchaudio==2.11.0`), matching vLLM v0.28.0's own CI lock.
- uv on your PATH.
- A Rust toolchain: `tokenizers` is built from source, and `safetensors`,
  `outlines_core`, and `openai-harmony` are built from their sdists.
- For the optional Triton source build: zlib development files
  (`sudo apt install zlib1g-dev`).  Triton's prebuilt LLVM exports a
  `ZLIB::ZLIB` target, so CMake configuration fails with "the target was not
  found" if only the `libz.so.1` runtime is present.  Triton also downloads its
  own ptxas/nvcc into `$TRITON_HOME/.triton` (default `~/.triton`); if that
  directory contains root-owned leftovers from a Docker run, set `TRITON_HOME`
  to a writable path for the build.
- Optional but recommended: ccache.


## Install commands

```bash
export CUDA_HOME=/usr/local/cuda-13.1.0        # wherever your toolkit lives
export PATH="$CUDA_HOME/bin:$PATH"             # build_uv.py needs nvcc
export TORCH_CUDA_ARCH_LIST=7.5                # your GPU's compute capability

./setup-venv.sh  # create uv venv (python 3.14t) and install deps
./clone-all.sh  # clone source repos
./install-all.sh --arch "$TORCH_CUDA_ARCH_LIST"  # build and install vllm
uv run python -c "import vllm; print(vllm.__version__)"
```

Add `--triton` (or `BUILD_TRITON=1`) to `clone-all.sh` and `install-all.sh` to
build Triton from source with the free-threading patches in `patches/triton/`.
On compute capability 7.5 that is strongly recommended: `vllm-flash-attn` is
sm_80+, so vLLM falls back to the Triton attention backend and the stock Triton
wheel is then exercised on every forward pass.


## Verify

```bash
PYTHON_GIL=0 uv run ./test/verify_build.py     # pins, versions, GIL state, extensions
./run_simple.sh                                # one-model generation
PYTHON_GIL=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    uv run ./test/threaded_stress.py threads   # multi-engine threaded run
```

Note that `PYTHON_GIL=0` is required, not merely advisory: several installed
packages do not declare `Py_MOD_GIL_NOT_USED` and would otherwise re-enable the
GIL on import.
