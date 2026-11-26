"""Measure standalone Phase 3 MarketIntelligenceEngine pure-ONNX cold load, peak RSS, and latency."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import psutil

from phase3.adapter import MarketIntelligenceEngine


def main() -> None:
    proc = psutil.Process()
    rss_init = proc.memory_info().rss / (1024**2)
    t0 = perf_counter()
    eng = MarketIntelligenceEngine(Path("models"), Path("phase3/results"))
    cold_sec = round(perf_counter() - t0, 3)
    rss_loaded = proc.memory_info().rss / (1024**2)

    samples = [
        ("INFY", "Technology", "operations", "Infosys bagged a USD 450 million multi-year digital contract across European utilities."),
        ("RELIANCE", "Energy", "earnings", "Reliance Industries Q4 FY26 consolidated EBITDA rose 14% YoY to INR 44,200 crore."),
        ("HDFCBANK", "Financial Services", "corporate_action", "HDFC Bank Board recommended a final dividend of Rs. 22 per equity share."),
        ("SUNPHARMA", "Healthcare", "regulatory", "Sun Pharma received USFDA warning letter inspection observations at facility."),
        ("LT", "Industrials", "deal_ma", "Larsen and Toubro completed acquisition of 74% stake in infrastructure JV for INR 1,850 crore."),
    ]
    # Warm-up
    eng.analyze(text=samples[0][3], symbol=samples[0][0], sector=samples[0][1], event_family_hint=samples[0][2], as_of_ts_ist="2026-08-01T10:00:00+05:30")

    lats = []
    for i in range(40):
        sym, sec, fam, txt = samples[i % len(samples)]
        ts = perf_counter_ns()
        eng.analyze(text=txt, symbol=sym, sector=sec, event_family_hint=fam, as_of_ts_ist="2026-08-01T10:00:00+05:30")
        lats.append((perf_counter_ns() - ts) / 1e6)
    rss_peak = proc.memory_info().rss / (1024**2)

    out = {
        "runtime": ".venv pure ONNX Runtime (zero PyTorch/CatBoost/Pandas)",
        "cold_load_sec": cold_sec,
        "rss_init_mb": round(rss_init, 1),
        "rss_after_load_mb": round(rss_loaded, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "warm_p50_ms": round(float(np.median(lats)), 2),
        "warm_p95_ms": round(float(np.percentile(lats, 95)), 2),
        "throughput_docs_per_sec": round(1000.0 / float(np.mean(lats)), 2),
    }
    Path("phase3/results/edge_runtime_benchmark.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
