"""Isolated subprocess CPU runtime benchmark (p50/p95 latency, peak RSS, artifact size, SHA-256)."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from stockbrief.fixtures import Document
from training.adapter import DomainEventAdapter, ONNXEndToEndScorer
from training.schema import load_jsonl

TEST_PATH = Path(__file__).resolve().parent / "data" / "test.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "results" / "runtime_benchmark.json"


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def bench_single(backend: str, model_dir: Path, weights_name: str) -> dict:
    rows = load_jsonl(TEST_PATH)
    docs = [
        Document(r["issuer_symbol"], tuple(r["pages"]), r["doc_id"], r["source"], r["date"], r["url"], "historical")
        for r in rows
    ]

    with threadpool_limits(limits=1):
        if backend == "bge":
            from stockbrief.semantic import load_semantic
            model = load_semantic(model_dir)
            weights_path = model_dir / "onnx" / "model_quantized.onnx"
            name = model.name
        else:
            meta = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
            scorer = ONNXEndToEndScorer(model_dir, weights_name)
            model = DomainEventAdapter(scorer, floor=meta["floor"], margin=meta["margin"])
            weights_path = model_dir / weights_name
            name = f"{meta['name']} ({weights_name})"

        # Warmup on 3 non-empty docs
        for d in docs[:5]:
            try:
                model.analyze(d)
            except ValueError:
                pass

        latencies = []
        peak_rss = psutil.Process().memory_info().rss
        for d in docs:
            t0 = perf_counter_ns()
            try:
                model.analyze(d)
            except ValueError:
                pass
            latencies.append((perf_counter_ns() - t0) / 1e6)
            rss = psutil.Process().memory_info().rss
            if rss > peak_rss:
                peak_rss = rss

    return {
        "name": name,
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_mib": round(weights_path.stat().st_size / (1024**2), 2),
        "weights_sha256": sha256_file(weights_path),
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "mean_ms": round(float(np.mean(latencies)), 3),
        "peak_rss_mib": round(peak_rss / (1024**2), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3, metavar=("BACKEND", "MODEL_DIR", "WEIGHTS"))
    args = parser.parse_args()
    if args.worker:
        res = bench_single(args.worker[0], Path(args.worker[1]), args.worker[2])
        print(json.dumps(res))
        return

    targets = [
        ("bge", "models/semantic", "onnx/model_quantized.onnx"),
        ("onnx", "models/modernbert-frozen-head", "model_quantized.onnx"),
        ("onnx", "models/modernbert-finetuned", "model.onnx"),
        ("onnx", "models/modernbert-finetuned", "model_quantized.onnx"),
    ]
    results = []
    for backend, mdir, wname in targets:
        proc = subprocess.run(
            [sys.executable, "-m", "training.bench_runtime", "--worker", backend, mdir, wname],
            capture_output=True,
            text=True,
            check=True,
        )
        res = json.loads(proc.stdout.strip())
        results.append(res)
        print(
            f"{res['name']}: size={res['weights_mib']} MiB, p50={res['p50_ms']} ms, "
            f"p95={res['p95_ms']} ms, peak_rss={res['peak_rss_mib']} MiB, sha256={res['weights_sha256'][:16]}...",
            flush=True,
        )

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
