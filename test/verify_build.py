#!/usr/bin/env python3
"""Post-build verifier for a free-threaded vllm build (CUDA, ROCm or CPU).

Checks the things that must hold before any model is loaded:

  * source trees sit on the commits git-repos.txt pins;
  * installed versions match requirements/, and `uv pip check` is clean apart
    from documented omissions;
  * every normal-path native module imports, and the expected GIL state
  * native extensions carry cp314t ABI tags, not abi3;
  * omitted optional packages are absent and fail cleanly rather than being
    half-installed.

Run it with PYTHON_GIL=0, the way the Dockerfiles and run_simple.sh do.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def detect_backend() -> str:
    """Return "cuda", "rocm" or "cpu" from the installed torch build."""
    import torch

    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


# Native extensions per backend (v0.28.0).  CUDA gets the stable-libtorch
# modules instead of vllm._C; flash-attention is CUDA-only.
EXTENSIONS = {
    "cuda": (
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm.fs_io_C",
        "vllm.cumem_allocator",
        "vllm.spinloop",
        "vllm.vllm_flash_attn._vllm_fa2_C",
    ),
    "rocm": (
        "vllm._C",
        "vllm._rocm_C",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm.fs_io_C",
        "vllm.cumem_allocator",
        "vllm.spinloop",
    ),
    # vllm._C is omitted: on x86 it is the AMX/AVX512-BF16 build and importing
    # it on a lesser CPU is SIGILL, not ImportError.  check_cpu_x86_kernels()
    # goes through vllm's own dispatch instead.
    "cpu": (
        "vllm.fs_io_C",
        "vllm.spinloop",
    ),
}

# The x86 ISA variants.  All share PyInit__C, so the two non-selected ones
# raise ImportError after registering their ops; vllm/platforms/cpu.py
# swallows that.
CPU_X86_ISA_LIBS = ("_C", "_C_AVX512", "_C_AVX2")

# Modules vllm imports on the normal path.  "declares" is whether the
# module declares Py_MOD_GIL_NOT_USED; the ones marked False keep the GIL
# disabled only because PYTHON_GIL=0 forces it.
NORMAL_PATH = {
    "torch": True,
    "numpy": True,
    "scipy": True,
    "tokenizers": True,
    "safetensors": True,
    "tiktoken": True,
    "llguidance": True,
    "regex": True,
    "msgspec": True,
    "numba": True,
    "sentencepiece": True,
    "zmq": True,
    "blake3": True,
    "pybase64": True,
    "cbor2": True,
    "setproctitle": True,
    "uvloop": True,
    "watchfiles": True,
    "google.protobuf": True,
    "pydantic_core": True,
    "PIL": True,
    "nvtx": True,
    "xgrammar": True,
    "outlines_core": False,
    "openai_harmony": False,
    "triton": False,
}

# Deliberately not installed
OMITTED = [
    "flashinfer",
    "fastsafetensors",
    "cv2",
    "PyNvVideoCodec",
    "tilelang",
    "quack",
    "tokenspeed_mla",
    "humming_kernels",
    "torchcodec",
]

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{name}: {detail}")


def pinned_commits() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "git-repos.txt").read_text().splitlines():
        line = re.sub(r"#.*", "", line).strip()
        if line:
            _org, repo, commit = line.split()
            pins[repo] = commit
    return pins


def check_sources() -> None:
    for repo, commit in pinned_commits().items():
        d = ROOT / "third_party" / repo
        if not d.is_dir():
            notes.append(f"{repo}: not cloned (optional source build?)")
            continue
        # Patched trees sit on descendants of the pin, so check ancestry
        # rather than equality.
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=d, capture_output=True,
        )
        head = subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=d,
            capture_output=True, text=True,
        ).stdout.strip()
        check(f"source {repo} contains pin {commit[:10]}", r.returncode == 0, head)


def check_gil(strict: bool, backend: str) -> None:
    code = "import sys, importlib; importlib.import_module(sys.argv[1]); print(sys._is_gil_enabled())"
    env = dict(os.environ)
    env.pop("PYTHON_GIL", None)  # measure what the module itself declares
    modules = dict(NORMAL_PATH)
    if backend == "cpu":
        # CUDA profiling shim and xgrammar's bare `triton` dependency; neither
        # is on the CPU path.
        for mod in ("nvtx", "triton"):
            if mod in modules:
                notes.append(f"{mod}: installed but unused on the CPU backend")
                del modules[mod]
    for mod, declares in modules.items():
        r = subprocess.run(
            [sys.executable, "-c", code, mod],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            check(f"import {mod}", False, r.stderr.strip().splitlines()[-1][:90])
            continue
        gil_on = r.stdout.strip() == "True"
        if declares:
            check(f"{mod} keeps the GIL disabled", not gil_on)
        elif gil_on:
            notes.append(f"{mod}: re-enables the GIL unless PYTHON_GIL=0 (known)")
            if strict:
                check(f"{mod} keeps the GIL disabled", False, "no FT declaration")
        else:
            notes.append(f"{mod}: now declares FT support")


def check_abi_tags() -> None:
    sp = Path(sys.prefix) / "lib" / f"python3.14t" / "site-packages"
    bad = []
    for mod in ("tokenizers", "safetensors", "tiktoken", "outlines_core"):
        d = sp / mod
        if not d.is_dir():
            continue
        for so in d.glob("*.so"):
            if "abi3" in so.name:
                bad.append(str(so.relative_to(sp)))
    check("no abi3 extensions among the source-built packages", not bad, ", ".join(bad))


def check_vllm_extensions(backend: str) -> None:
    try:
        import vllm  # noqa: F401
    except Exception as e:  # pragma: no cover - reported as a failure
        check("import vllm", False, f"{type(e).__name__}: {e}")
        return
    check("import vllm", True, f"version {vllm.__version__}")
    for ext in EXTENSIONS[backend]:
        try:
            __import__(ext)
            check(f"import {ext}", True)
        except Exception as e:
            check(f"import {ext}", False, f"{type(e).__name__}: {str(e)[:80]}")

    if backend == "cpu" and platform.machine() in ("x86_64", "AMD64"):
        check_cpu_x86_kernels(vllm)


def check_cpu_x86_kernels(vllm) -> None:
    """All three ISA variants present, and the one this host selects works."""
    import torch

    pkg = Path(vllm.__file__).parent
    for lib in CPU_X86_ISA_LIBS:
        found = sorted(pkg.glob(f"{lib}.cpython-*.so"))
        check(f"{lib} extension built", bool(found),
              found[0].name if found else "missing")

    if torch.cpu._is_avx512_supported():
        selected = "_C" if torch.cpu._is_avx512_bf16_supported() else "_C_AVX512"
    else:
        selected = "_C_AVX2"
    notes.append(f"CPU ISA dispatch selects vllm.{selected} on this host")

    from vllm.platforms import current_platform

    current_platform.import_kernels()
    check("vllm CPU kernels registered", hasattr(torch.ops, "_C"),
          f"via {selected}")
    # Prove the selected variant actually executes rather than just loading.
    try:
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, torch.ones(16, dtype=torch.bfloat16), 1e-6)
        ok = bool(torch.isfinite(out).all())
        check("torch.ops._C.rms_norm runs", ok)
    except Exception as e:
        check("torch.ops._C.rms_norm runs", False, f"{type(e).__name__}: {e}")


def check_omitted() -> None:
    present = []
    for mod in OMITTED:
        try:
            __import__(mod)
            present.append(mod)
        except ImportError:
            pass
    check("omitted optional packages are absent", not present, ", ".join(present))


def check_pip() -> None:
    # uv is build-stage only; runtime images lack it.
    if shutil.which("uv") is None:
        notes.append("uv not on PATH (runtime image); skipped `uv pip check`")
        return
    venv = ROOT / ".venv"
    if not venv.is_dir():
        venv = Path(sys.prefix)
    r = subprocess.run(
        ["uv", "pip", "check"], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "VIRTUAL_ENV": str(venv)},
    )
    out = (r.stdout + r.stderr).strip()
    # Expected omissions plus the deliberate tokenizers divergence:
    # transformers caps it at <=0.23.0, we install 0.23.1.
    allowed = re.compile(
        r"tokenizers|opencv-python-headless|opentelemetry-exporter-otlp|"
        r"PyNvVideoCodec|torchcodec|flashinfer|tilelang|apache-tvm-ffi|"
        r"nvidia-cudnn-frontend|nvidia-cutlass-dsl|quack-kernels|"
        r"tokenspeed-mla|humming-kernels|fastsafetensors",
        re.I,
    )
    problems = [
        ln for ln in out.splitlines()
        if ln.strip().startswith("-") or "requires" in ln
    ]
    unexpected = [ln for ln in problems if not allowed.search(ln)]
    check("uv pip check (documented deviations allowed)", not unexpected,
          "; ".join(unexpected)[:160])
    for ln in problems:
        if allowed.search(ln):
            notes.append(f"expected: {ln.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict-gil", action="store_true",
                    help="fail on modules that need PYTHON_GIL=0 rather than noting them")
    args = ap.parse_args()

    backend = detect_backend()
    print(f"python {sys.version.split()[0]} free-threaded={not sys._is_gil_enabled()}")
    print(f"venv   {sys.prefix}")
    print(f"backend {backend}\n")

    check_sources()
    check_gil(args.strict_gil, backend)
    check_abi_tags()
    check_vllm_extensions(backend)
    check_omitted()
    check_pip()

    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")
    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
