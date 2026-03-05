#!/usr/bin/env python3
#
# Build docker image, automatically determining available
# RAM and number of CPUs.

import argparse
import multiprocessing
import os
import subprocess
import sys

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


def main():
    parser = argparse.ArgumentParser(
        description="Build the vllm docker image"
    )
    parser.add_argument(
        "--compute",
        choices=["cuda", "rocm", "cpu"],
        default="cuda",
        help="Which vllm backend to build (default: cuda)",
    )
    args = parser.parse_args()

    compute = args.compute

    cpus = multiprocessing.cpu_count()
    max_jobs, nvcc_threads = get_build_args(cpus)
    print(
        f"Detected: {cpus} CPUs, ~{(max_jobs * RAM_PER_JOB)}GB RAM allocated"
    )
    print(f"Setting MAX_JOBS={max_jobs}, NVCC_THREADS={nvcc_threads}")
    print(f"Building for compute backend: {compute}")

    dockerfile = f"Dockerfile.{compute}"
    if not os.path.exists(dockerfile):
        print(
            f"Error: {dockerfile} does not exist. Please create it or choose a different --compute option."
        )
        sys.exit(2)

    image_tag = f"vllm-freethreaded-{compute}"

    os.environ['BUILDX_EXPERIMENTAL'] = '1'
    cmd = [
        "docker",
        "buildx", "debug", "--invoke", "/bin/bash",
        "build",
        "--progress=plain",
        "--build-arg",
        f"MAX_JOBS={max_jobs}",
    ]

    if compute == "cuda":
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST") or "7.5"
        cmd += [
            "--build-arg",
            f"NVCC_THREADS={nvcc_threads}",
            "--build-arg",
            f"TORCH_CUDA_ARCH_LIST={arch_list}",
        ]
    elif compute == "rocm":
        arch = os.environ.get("TORCH_ROCM_ARCH") or "gfx1030"
        cmd += [
            "--build-arg",
            f"PYTORCH_ROCM_ARCH={arch}",
        ]
    elif compute == "cpu":
        # Pass through any VLLM_CPU_* env vars as build args
        for key, value in os.environ.items():
            if key.startswith("VLLM_CPU_"):
                cmd += ["--build-arg", f"{key}={value}"]
    else:
        raise RuntimeError

    cmd += ["-t", image_tag, "-f", dockerfile, "."]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
