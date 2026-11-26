"""Measure standalone Architecture A pure ONNX INT8 runtime (no PyTorch) cold start, RSS, and p50/p95 latency."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from phase2.reasoning_pipeline import EdgeReasoningPipeline
from phase2.schema import load_jsonl


def main() -> None:
    proc = psutil.Process()
    rss_init = proc.memory_info().rss / (1024**2)
    t0 = perf_counter()
    with threadpool_limits(limits=1):
        pipe = EdgeReasoningPipeline(Path("models"))
        cold_sec = round(perf_counter() - t0, 3)
        rss_loaded = proc.memory_info().rss / (1024**2)

        bench = load_jsonl(Path("phase2/data/benchmark_frozen.jsonl"))
        # Warm-up 5 cases, then measure all 370 cases
        for r in bench[:5]:
            pipe.answer_question(r["text"], r["question"], ticker=r.get("issuer_symbol", "OTHER"))
        lats = []
        for r in bench:
            ts = perf_counter_ns()
            pipe.answer_question(r["text"], r["question"], ticker=r.get("issuer_symbol", "OTHER"))
            lats.append((perf_counter_ns() - ts) / 1e6)
    rss_peak = proc.memory_info().rss / (1024**2)

    out = {
        "runtime": ".venv pure ONNX Runtime (no PyTorch/Transformers)",
        "rss_init_mb": round(rss_init, 1),
        "rss_after_load_mb": round(rss_loaded, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "cold_load_sec": cold_sec,
        "cases": len(bench),
        "warm_p50_ms": round(float(np.median(lats)), 2),
        "warm_p95_ms": round(float(np.percentile(lats, 95)), 2),
        "throughput_docs_per_sec": round(1000.0 / float(np.mean(lats)), 2),
    }
    Path("phase2/results/arch_a_standalone_runtime.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
