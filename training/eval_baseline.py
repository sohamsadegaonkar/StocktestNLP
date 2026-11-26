"""Score the pinned BGE-small-en-v1.5 INT8 SemanticEventModel baseline on the frozen domain splits and repo benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from threadpoolctl import threadpool_limits

from stockbrief.semantic import load_semantic, model_directory
from training.evaluator import evaluate_domain_split, evaluate_repo_benchmarks
from training.schema import load_jsonl

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = Path(__file__).resolve().parent / "results" / "bge_baseline.json"


def main() -> None:
    cal_rows = load_jsonl(DATA_DIR / "calibration.jsonl")
    test_rows = load_jsonl(DATA_DIR / "test.jsonl")

    with threadpool_limits(limits=1):
        model = load_semantic(model_directory())
        cal_metrics = evaluate_domain_split(model.name, model.analyze, cal_rows)
        test_metrics = evaluate_domain_split(model.name, model.analyze, test_rows)
        repo_metrics = evaluate_repo_benchmarks(model.analyze)

    payload = {
        "model": model.name,
        "model_version": model.version,
        "floor": model.floor,
        "margin": model.margin,
        "artifact": model.artifact,
        "calibration": cal_metrics,
        "test": test_metrics,
        "repo_benchmarks": repo_metrics,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"BGE baseline test macro_f1={test_metrics['macro_f1']}, review_p={test_metrics['review_precision']}, review_r={test_metrics['review_recall']}, multilabel_micro_f1={test_metrics['multilabel_micro_f1']}, evidence_acc={test_metrics['evidence_accuracy']}, p50={test_metrics['latency_ms']['p50']}ms, p95={test_metrics['latency_ms']['p95']}ms")
    print(f"Negation errors={test_metrics['negation_errors']}, Scheduled errors={test_metrics['scheduled_errors']}, Numeric errors={test_metrics['numeric_errors']}, Failed PDF={test_metrics['failed_pdf_accuracy']}, Mixed P+E={test_metrics['mixed_pricing_earnings_exact']}")
    print("Repo benchmarks:", json.dumps(repo_metrics))


if __name__ == "__main__":
    main()
