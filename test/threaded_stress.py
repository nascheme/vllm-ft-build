#!/usr/bin/env python3
"""Free-threaded generation stress runs for the CUDA build.

Replaces the vllm_ft_bench harness (its tree is not on this machine).  Modes:

  single      one engine, one pass — the cheapest smoke test
  repeat      one engine, --iters sequential passes (leak / state drift)
  threads     --engines in-process engines, one per GPU, generating in threads
  structured  guided-decoding (JSON schema) smoke + threaded stress

Every mode checks the same things: outputs are non-empty, greedy decoding is
reproducible across passes/engines, the GIL stays disabled, and no thread dies.

Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 for the threaded modes (engine cores
must live in this process) and PYTHON_GIL=0 on the free-threaded build.

The GPUs here are 6 GB Turing cards: fp16 only (no bf16 below sm_80) and small
models.
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

import torch

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
    """Construct an engine pinned to cuda:{device_index}."""
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    engine_args = EngineArgs(
        model=args.model,
        dtype="half",  # sm_75 has no bf16
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        compilation_config={"cudagraph_mode": args.cudagraph_mode},
    )
    vllm_config = engine_args.create_engine_config()
    # UniProcExecutor._distributed_args() reads the device index back out of
    # device_config.device, so this is what pins the engine to a GPU.
    vllm_config.device_config.device = torch.device(f"cuda:{device_index}")
    return LLMEngine.from_vllm_config(vllm_config)


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
        mem = torch.cuda.memory_allocated(args.device) / 2**20
        print(f"  iter {it}: {dt:5.2f}s  torch-allocated {mem:7.1f} MiB")
    return problems


def mode_threads(args) -> list[str]:
    n = args.engines
    available = torch.cuda.device_count()
    if n > available:
        return [f"asked for {n} engines but only {available} GPU(s) present"]

    # Engines are CONSTRUCTED sequentially: set_current_vllm_config() uses a
    # module-global, so concurrent construction races (separate vLLM bug).
    engines = []
    for i in range(n):
        print(f"  building engine {i} on cuda:{i} ...", flush=True)
        engines.append(build_engine(i, args))

    results: dict[int, dict[str, str]] = {}
    errors: dict[int, str] = {}
    devices: dict[int, int] = {}
    barrier = threading.Barrier(n)

    def worker(idx: int) -> None:
        try:
            # CUDA current device is per-thread; bind this thread to its GPU.
            torch.accelerator.set_device_index(torch.device(f"cuda:{idx}"))
            devices[idx] = torch.accelerator.current_device_index()
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
    if len(set(devices.values())) != n:
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
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--eager", action="store_true", help="disable torch.compile")
    ap.add_argument("--cudagraph-mode", default="NONE",
                    help="NONE keeps a second in-process engine from tripping "
                         "cudaErrorStreamCaptureImplicit against the first")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.timeout + 120, exit=True)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    gil = sys._is_gil_enabled()
    print(f"python {sys.version.split()[0]}  gil_enabled={gil}  "
          f"gpus={torch.cuda.device_count()}  mode={args.mode}")
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
