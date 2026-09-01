#!/usr/bin/env python3
"""Free-threaded generation stress runs (CUDA, ROCm or CPU build).

Replaces the vllm_ft_bench harness for the non-CUDA backends -- that harness
pins every engine to a cuda:N device.  Modes:

  single      one engine, one pass — the cheapest smoke test
  repeat      one engine, --iters sequential passes (leak / state drift)
  threads     --engines in-process engines generating in threads (one per GPU
              on an accelerator; all sharing the cores on CPU)
  structured  guided-decoding (JSON schema) smoke + threaded stress

Every mode checks the same things: outputs are non-empty, greedy decoding is
reproducible across passes/engines, the GIL stays disabled, and no thread dies.

Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 for the threaded modes (engine cores
must live in this process) and PYTHON_GIL=0 on the free-threaded build.

Defaults: fp16 on the 6 GB Turing cards here (no bf16 below sm_80), bf16 on
CPU, a small model everywhere.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
import traceback

# vllm's CPU attention kernel asserts omp_get_max_threads() is the same when
# metadata is built and when the kernel runs (csrc/cpu/cpu_attn_impl.hpp).
# With OMP_NUM_THREADS unset, threads torch has not initialised default to the
# logical CPU count while torch uses physical cores, so two in-process engines
# fail with "thread_num == thread_num (28 vs. 20)".  Pin it to the physical
# core count.  libgomp reads the variable at load time, so re-exec before
# importing torch.
if (
    "threads" in sys.argv[1:]
    and "OMP_NUM_THREADS" not in os.environ
    and os.environ.get("_VLLM_FT_OMP_REEXEC") != "1"
):
    _cores, _phys = set(), None
    try:
        with open("/proc/cpuinfo") as _f:
            for _line in _f:
                _k, _, _v = _line.partition(":")
                _k, _v = _k.strip(), _v.strip()
                if _k == "physical id":
                    _phys = _v
                elif _k == "core id":
                    _cores.add((_phys, _v))
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(len(_cores) or os.cpu_count() or 1)
    os.environ["_VLLM_FT_OMP_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import torch

IS_ACCEL = torch.cuda.is_available()

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The largest planet in the solar system is",
    "Water freezes at",
]

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}


def greedy_params(max_tokens: int, schema: dict | None = None):
    from vllm import SamplingParams

    kwargs = dict(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=1234)
    if schema is not None:
        from vllm.sampling_params import StructuredOutputsParams

        kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
    return SamplingParams(**kwargs)


def build_engine(device_index: int, args):
    """Construct an engine; on an accelerator it is pinned to cuda:{index}."""
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        # On CPU: fraction of host RAM to reserve.
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if IS_ACCEL:
        kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    elif args.kv_cache_gib:
        # CPU sizes the KV cache from (util * host RAM) - process RSS, so a
        # second in-process engine sees the first one's cache as overhead and
        # gets a negative budget.  An explicit size bypasses that.
        kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gib * 2**30)
    engine_args = EngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    if IS_ACCEL:
        # UniProcExecutor._distributed_args() reads the device index back out
        # of device_config.device, so this is what pins the engine to a GPU.
        vllm_config.device_config.device = torch.device(f"cuda:{device_index}")
    return LLMEngine.from_vllm_config(vllm_config)


def allocated_mib(device_index: int) -> tuple[str, float]:
    """A memory number to watch across iterations, and what it is."""
    if IS_ACCEL:
        return "torch-allocated", torch.cuda.memory_allocated(device_index) / 2**20
    # No per-device allocator to ask on CPU; process RSS is the closest thing.
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return "rss", pages * os.sysconf("SC_PAGE_SIZE") / 2**20


# vllm/forward_context.py keeps _forward_context in a module-global rather than
# a ContextVar, so two in-process engines must not be inside a forward pass at
# the same time.  Pre-existing vLLM free-threading gap, unrelated to what these
# runs are measuring.
FWD_LOCK = threading.Lock()


def run_pass(engine, tag: str, schema: dict | None, args) -> dict[str, str]:
    params = greedy_params(args.max_tokens, schema)
    for i, prompt in enumerate(PROMPTS[: args.num_prompts]):
        engine.add_request(f"{tag}-r{i}", prompt, params)
    texts: dict[str, str] = {}
    while engine.has_unfinished_requests():
        with FWD_LOCK:
            step = engine.step()
        for out in step:
            if out.finished:
                texts[out.request_id] = out.outputs[0].text
    return texts


def check_texts(label: str, texts: dict[str, str], expected_n: int) -> list[str]:
    problems = []
    if len(texts) != expected_n:
        problems.append(f"{label}: got {len(texts)}/{expected_n} outputs")
    for rid, t in sorted(texts.items()):
        if not t.strip():
            problems.append(f"{label}: empty output for {rid}")
    return problems


def mode_single(args) -> list[str]:
    engine = build_engine(args.device, args)
    texts = run_pass(engine, "s", None, args)
    for rid, t in sorted(texts.items()):
        print(f"  {rid}: {t!r}")
    return check_texts("single", texts, args.num_prompts)


def mode_repeat(args) -> list[str]:
    engine = build_engine(args.device, args)
    problems: list[str] = []
    first: dict[str, str] | None = None
    for it in range(args.iters):
        t0 = time.monotonic()
        texts = run_pass(engine, f"i{it}", None, args)
        dt = time.monotonic() - t0
        stripped = {k.split("-", 1)[1]: v for k, v in texts.items()}
        problems += check_texts(f"iter{it}", texts, args.num_prompts)
        if first is None:
            first = stripped
        elif stripped != first:
            problems.append(f"iter{it}: greedy output drifted from iteration 0")
        label, mem = allocated_mib(args.device)
        print(f"  iter {it}: {dt:5.2f}s  {label} {mem:7.1f} MiB")
    return problems


def mode_threads(args) -> list[str]:
    n = args.engines
    if IS_ACCEL:
        available = torch.cuda.device_count()
        if n > available:
            return [f"asked for {n} engines but only {available} GPU(s) present"]

    # Engines are CONSTRUCTED sequentially: set_current_vllm_config() uses a
    # module-global, so concurrent construction races (separate vLLM bug).
    engines = []
    for i in range(n):
        where = f"cuda:{i}" if IS_ACCEL else "cpu"
        print(f"  building engine {i} on {where} ...", flush=True)
        engines.append(build_engine(i, args))

    results: dict[int, dict[str, str]] = {}
    errors: dict[int, str] = {}
    devices: dict[int, int] = {}
    barrier = threading.Barrier(n)

    def worker(idx: int) -> None:
        try:
            if IS_ACCEL:
                # The current device is per-thread; bind this thread to its GPU.
                torch.accelerator.set_device_index(torch.device(f"cuda:{idx}"))
                devices[idx] = torch.accelerator.current_device_index()
            else:
                devices[idx] = idx
            barrier.wait()
            texts: dict[str, str] = {}
            for it in range(args.iters):
                texts = run_pass(engines[idx], f"t{idx}i{it}", None, args)
            results[idx] = {k.split("-", 1)[1]: v for k, v in texts.items()}
        except Exception:
            errors[idx] = traceback.format_exc()

    threads = [threading.Thread(target=worker, args=(i,), name=f"engine-{i}") for i in range(n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.timeout)
    elapsed = time.monotonic() - t0

    problems: list[str] = []
    for t in threads:
        if t.is_alive():
            problems.append(f"{t.name} still running after {args.timeout}s (hang)")
    for idx, tb in sorted(errors.items()):
        print(f"\n!!! engine {idx} raised:\n{tb}")
        problems.append(f"engine {idx} raised an exception")

    print(f"\n  {n} engines x {args.iters} iters in {elapsed:.2f}s")
    if IS_ACCEL and len(set(devices.values())) != n:
        problems.append(f"engines did not land on distinct devices: {devices}")
    for idx, texts in sorted(results.items()):
        problems += check_texts(f"engine{idx}", texts, args.num_prompts)
        for rid, t in sorted(texts.items()):
            print(f"  engine {idx} {rid}: {t!r}")
    # Same model, same greedy params: every engine must agree.
    if len(results) > 1:
        ref_idx = min(results)
        for idx in sorted(results):
            if idx != ref_idx and results[idx] != results[ref_idx]:
                problems.append(f"engine {idx} disagrees with engine {ref_idx}")
    return problems


def mode_structured(args) -> list[str]:
    engine = build_engine(args.device, args)
    params = greedy_params(args.max_tokens, SCHEMA)
    prompt = "Give the city and population of the capital of France as JSON."
    for i in range(args.num_prompts):
        engine.add_request(f"j{i}", prompt, params)
    texts: dict[str, str] = {}
    while engine.has_unfinished_requests():
        with FWD_LOCK:
            for out in engine.step():
                if out.finished:
                    texts[out.request_id] = out.outputs[0].text
    problems = check_texts("structured", texts, args.num_prompts)
    for rid, t in sorted(texts.items()):
        print(f"  {rid}: {t!r}")
        try:
            obj = json.loads(t)
        except json.JSONDecodeError as e:
            problems.append(f"{rid}: not valid JSON ({e})")
            continue
        missing = [k for k in SCHEMA["required"] if k not in obj]
        if missing:
            problems.append(f"{rid}: JSON missing required keys {missing}")
    return problems


MODES = {
    "single": mode_single,
    "repeat": mode_repeat,
    "threads": mode_threads,
    "structured": mode_structured,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--engines", type=int, default=2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--dtype", default="half" if IS_ACCEL else "bfloat16",
                    help="half on sm_75 (no bf16 below sm_80); bfloat16 on CPU")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="default 0.55 on GPU, 0.25 on CPU (fraction of host RAM)")
    ap.add_argument("--kv-cache-gib", type=float, default=None,
                    help="CPU only: explicit KV cache size per engine, in GiB. "
                         "Defaults to 2 in threads mode, unset otherwise.")
    ap.add_argument("--eager", action="store_true", help="disable torch.compile")
    ap.add_argument("--cudagraph-mode", default="NONE",
                    help="NONE keeps a second in-process engine from tripping "
                         "cudaErrorStreamCaptureImplicit against the first")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = 0.55 if IS_ACCEL else 0.25
    if not IS_ACCEL and args.kv_cache_gib is None and args.mode == "threads":
        args.kv_cache_gib = 2.0

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.timeout + 120, exit=True)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    gil = sys._is_gil_enabled()
    where = (f"gpus={torch.cuda.device_count()}" if IS_ACCEL
             else f"device=cpu  omp={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"python {sys.version.split()[0]}  gil_enabled={gil}  "
          f"{where}  dtype={args.dtype}  mem-util={args.gpu_memory_utilization:.3f}  "
          f"mode={args.mode}")
    problems = MODES[args.mode](args)
    if gil:
        problems.append("the GIL was enabled for this run (set PYTHON_GIL=0)")
    if sys._is_gil_enabled() and not gil:
        problems.append("an import during the run re-enabled the GIL")

    print("\nRESULT:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
