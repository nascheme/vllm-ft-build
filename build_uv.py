#!/usr/bin/env python3
"""
Helper to build vllm in a uv-created Python virtualenv on the host.

Usage:
  ./build_uv.py [--arch ARCH]

Options:
  --arch ARCH          TORCH_CUDA_ARCH_LIST (default: from env or '7.5')
  --max-jobs N         Override auto-detected MAX_JOBS
  --nvcc-threads N     Override auto-detected NVCC_THREADS

This script expects 'uv' to be on PATH and that you've already created and
activated a uv venv (for example: uv venv /opt/venv --python cpython-3.14t
and exported PATH to include /opt/venv/bin).

It will build safetensors, tokenizers, harmony (non-editable) and then
perform an editable install of vllm (the same command the Dockerfile runs).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path

# vLLM needs ~6-8GB per job to be safe
RAM_PER_JOB = 8


def get_build_args(cpus):
    # Calculate available RAM (in GB)
    with open("/proc/meminfo", "r") as f:
        mem_total_kb = int(
            [line for line in f if "MemTotal" in line][0].split()[1]
        )
    ram_gb = mem_total_kb / 1e6

    # On 64GB, this will result in 8 jobs.
    max_jobs = max(1, int(ram_gb // RAM_PER_JOB))
    max_jobs = min(max_jobs, cpus)

    # Use remaining CPU overhead for NVCC internal threading
    # If we only run 8 jobs on a 28-thread CPU, let each job use 3 threads.
    nvcc_threads = max(1, int(cpus // max_jobs))
    nvcc_threads = min(nvcc_threads, 4)  # NVCC gains diminish after 4

    return max_jobs, nvcc_threads


def run(cmd, env=None, check=True):
    print("+ ", " ".join(cmd))
    subprocess.run(cmd, check=check, env=env)


def main():
    parser = argparse.ArgumentParser(description="Build vllm using uv (CUDA)")
    parser.add_argument(
        "--arch",
        dest="arch",
        default=None,
        help="TORCH_CUDA_ARCH_LIST (default: $TORCH_CUDA_ARCH_LIST or 7.5)",
    )
    parser.add_argument(
        "--max-jobs",
        dest="max_jobs",
        type=int,
        default=None,
        help="Override MAX_JOBS detected from RAM",
    )
    parser.add_argument(
        "--nvcc-threads",
        dest="nvcc_threads",
        type=int,
        default=None,
        help="Override NVCC_THREADS detected from CPUs",
    )

    args = parser.parse_args()

    # Ensure uv is available
    if shutil.which("uv") is None:
        print(
            "Error: 'uv' not found on PATH. Install uv and ensure it's available."
        )
        sys.exit(2)

    # Ensure we're running from repo root (where third_party/ and clone-repos.py exist)
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    cpus = multiprocessing.cpu_count()
    max_jobs, nvcc_threads = get_build_args(cpus)
    if args.max_jobs is not None:
        max_jobs = args.max_jobs
    if args.nvcc_threads is not None:
        nvcc_threads = args.nvcc_threads

    arch_list = args.arch or os.environ.get("TORCH_CUDA_ARCH_LIST") or "7.5"

    print(
        f"Detected: {cpus} CPUs -> MAX_JOBS={max_jobs}, NVCC_THREADS={nvcc_threads}"
    )
    print(f"TORCH_CUDA_ARCH_LIST={arch_list}")

    # Ensure uv venv is active or present
    venv_bin = os.environ.get("VIRTUAL_ENV")
    if not venv_bin:
        print(
            "Warning: VIRTUAL_ENV not set. Make sure you created and activated an uv venv."
        )
    else:
        print(f"Using VIRTUAL_ENV={venv_bin}")

    # Prepare environment for subprocesses
    env = os.environ.copy()
    env["MAX_JOBS"] = str(max_jobs)
    env["NVCC_THREADS"] = str(nvcc_threads)
    env["TORCH_CUDA_ARCH_LIST"] = arch_list

    # Local flash-attention source used by vllm's build
    flash_attn = (repo_root / "third_party" / "flash-attention").resolve()
    env["VLLM_FLASH_ATTN_SRC_DIR"] = str(flash_attn)

    # Compiler wrappers — use ccache only if it's available on PATH.
    if shutil.which("ccache"):
        print("Using ccache for compilation")
        env["CC"] = env.get("CC", "ccache gcc")
        env["CXX"] = env.get("CXX", "ccache g++")
        env["CMAKE_CUDA_COMPILER_LAUNCHER"] = env.get(
            "CMAKE_CUDA_COMPILER_LAUNCHER", "ccache"
        )
    else:
        print("ccache not found on PATH; building without ccache")
        env["CC"] = env.get("CC", "gcc")
        env["CXX"] = env.get("CXX", "g++")

    try:
        # Editable vllm build
        run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                "third_party/vllm",
                "-v",
                "--no-build-isolation",
                "--no-deps",
            ],
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

    print("Build complete. Test with: ./run_simple.sh")


if __name__ == "__main__":
    main()
