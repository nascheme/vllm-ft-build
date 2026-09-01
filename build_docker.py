#!/usr/bin/env python3
#
# Build docker image, automatically determining available
# RAM and number of CPUs.

import argparse
import glob
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


def detect_rocm_arch():
    """Return the host GPU's gfx target, e.g. "gfx1102", or None.

    The kernel driver publishes gfx_target_version as MMmmss in decimal
    (110002 -> gfx1102, 90402 -> gfx942, 90010 -> gfx90a); the stepping is
    rendered in hex, which is where the trailing letter in gfx90a comes from.
    Node 0 is the CPU node and reports 0, so it is skipped.
    """
    archs = []
    nodes = sorted(
        glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
    )
    for path in nodes:
        try:
            with open(path) as f:
                for line in f:
                    key, _, value = line.partition(" ")
                    if key != "gfx_target_version":
                        continue
                    version = int(value)
                    if version == 0:  # CPU node
                        break
                    major, minor, step = (
                        version // 10000,
                        (version // 100) % 100,
                        version % 100,
                    )
                    arch = f"gfx{major}{minor}{step:x}"
                    if arch not in archs:
                        archs.append(arch)
                    break
        except OSError:
            continue
    return ";".join(archs) if archs else None


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
    parser.add_argument(
        "--triton",
        action="store_true",
        default=os.environ.get("BUILD_TRITON", "0") == "1",
        help="Also build and install Triton from source (with the "
        "free-threading patches). Default: $BUILD_TRITON or off.",
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
        arch = (
            os.environ.get("PYTORCH_ROCM_ARCH")
            or os.environ.get("TORCH_ROCM_ARCH")
            or detect_rocm_arch()
        )
        if not arch:
            print(
                "Error: could not detect the GPU architecture from "
                "/sys/class/kfd. Set PYTORCH_ROCM_ARCH (for example "
                "PYTORCH_ROCM_ARCH=gfx1102) and re-run."
            )
            sys.exit(2)
        print(f"Building HIP kernels for: {arch}")
        cmd += [
            "--build-arg",
            f"PYTORCH_ROCM_ARCH={arch}",
        ]
        # ROCM_VERSION picks the base image tag, TORCH_INDEX_SUFFIX the
        # matching PyTorch wheel index; the two must stay in step.  Both
        # default in the Dockerfile; override together, e.g.
        # ROCM_VERSION=7.1.1 TORCH_INDEX_SUFFIX=rocm7.1 ./build_docker.py --compute=rocm
        for key in ("ROCM_VERSION", "TORCH_INDEX_SUFFIX"):
            value = os.environ.get(key)
            if value:
                cmd += ["--build-arg", f"{key}={value}"]
    elif compute == "cpu":
        # Pass through any VLLM_CPU_* env vars as build args
        for key, value in os.environ.items():
            if key.startswith("VLLM_CPU_"):
                cmd += ["--build-arg", f"{key}={value}"]
    else:
        raise RuntimeError

    cmd += ["--build-arg", f"BUILD_TRITON={1 if args.triton else 0}"]
    if args.triton:
        print("Triton: building from source with free-threading patches")

    cmd += ["-t", image_tag, "-f", dockerfile, "."]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
