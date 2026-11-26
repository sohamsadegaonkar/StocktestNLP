"""Manual-only retraining and gated promotion CLI consuming human-reviewed annotations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from threadpoolctl import threadpool_limits

from training.adapter import DomainEventAdapter, ONNXEndToEndScorer
from training.dataset_utils import build_training_samples
from training.evaluator import evaluate_domain_split, evaluate_repo_benchmarks
from training.schema import AnnotatedDocument, VALID_LABELS, load_jsonl, normalize_text, save_jsonl, text_fingerprint

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "registry.json"


def default_registry(bge_baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "active_version": "stockbrief-bge-events-v1",
        "active_backend": "stockbrief.semantic.SemanticEventModel",
        "active_model_dir": "models/semantic",
        "auto_train_enabled": False,
        "schedule_enabled": False,
        "baseline_metrics": (bge_baseline or {}).get("test", {}),
        "history": [],
    }


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.is_file():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    bge_path = RESULTS_DIR / "bge_baseline.json"
    bge = json.loads(bge_path.read_text(encoding="utf-8")) if bge_path.is_file() else None
    reg = default_registry(bge)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return reg


def validate_reviewed_additions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for idx, raw in enumerate(rows):
        doc_id = raw.get("doc_id", f"row-{idx}")
        if not raw.get("reviewed", False):
            rejected.append(f"{doc_id}: not human-reviewed (reviewed != true)")
            continue
        if raw.get("source") == "user_upload" or raw.get("origin") == "upload":
            rejected.append(f"{doc_id}: user uploads are forbidden in training")
            continue
        try:
            doc = AnnotatedDocument(**{k: v for k, v in raw.items() if k in AnnotatedDocument.__dataclass_fields__})
        except TypeError as exc:
            rejected.append(f"{doc_id}: schema error ({exc})")
            continue
        errs = doc.validate()
        if errs:
            rejected.append("; ".join(errs))
            continue
        accepted.append(doc.to_dict())
    return accepted, rejected


def check_promotion_gates(
    candidate_test: dict[str, Any],
    repo_res: dict[str, Any],
    export_info: dict[str, Any],
    champion_test: dict[str, Any],
    max_p95_ms: float = 150.0,
    max_int8_bytes: int = 80 * 1024 * 1024,
) -> dict[str, Any]:
    gates = {
        "improves_test_macro_f1": candidate_test["macro_f1"] > champion_test.get("macro_f1", 0.0),
        "preserves_review_recall": candidate_test["review_recall"] >= champion_test.get("review_recall", 0.0),
        "improves_or_keeps_multilabel_f1": candidate_test["multilabel_micro_f1"] >= champion_test.get("multilabel_micro_f1", 0.0),
        "repo_tata_passed": bool(repo_res["tata_acceptance_passed"]),
        "repo_test_32_review_recall_ok": repo_res["repo_test_32"]["review_recall"] >= 0.75,
        "cpu_p95_within_budget": (candidate_test["latency_ms"]["p95"] or 9999.0) <= max_p95_ms,
        "int8_size_within_budget": export_info["int8_bytes"] <= max_int8_bytes,
    }
    gates["promote"] = all(gates.values())
    return gates


def run_retrain(
    reviewed_path: Path | None,
    arch: str,
    version_tag: str,
    max_p95_ms: float,
    max_int8_mb: float,
    dry_run: bool,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from training.onnx_export import export_and_quantize
    from training.train_models import BASE_DIR, compare_quantized_vs_unquantized, set_seed, train_finetuned, train_frozen_head

    torch.set_num_threads(4)
    set_seed(19)

    registry = load_registry()
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    cal_rows = load_jsonl(DATA_DIR / "calibration.jsonl")
    test_rows = load_jsonl(DATA_DIR / "test.jsonl")

    # Keep frozen test and calibration strictly isolated from reviewed additions
    heldout_fps = {text_fingerprint(r["text"]) for r in test_rows + cal_rows if r["source_mode"] != "pdf_extraction_failed"}
    heldout_ids = {r["doc_id"] for r in test_rows + cal_rows}
    heldout_issuers = {r["issuer_symbol"] for r in test_rows} - {r["issuer_symbol"] for r in train_rows}

    added_count = 0
    rejected_reasons: list[str] = []
    if reviewed_path is not None and reviewed_path.is_file():
        raw_additions = load_jsonl(reviewed_path)
        valid_additions, rejected_reasons = validate_reviewed_additions(raw_additions)
        existing_ids = {r["doc_id"] for r in train_rows}
        existing_fps = {text_fingerprint(r["text"]) for r in train_rows if r["source_mode"] != "pdf_extraction_failed"}
        for row in valid_additions:
            fp = text_fingerprint(row["text"]) if row["source_mode"] != "pdf_extraction_failed" else row["doc_id"]
            if row["doc_id"] in existing_ids or row["doc_id"] in heldout_ids or fp in heldout_fps or fp in existing_fps:
                continue
            if row["issuer_symbol"] in heldout_issuers:
                rejected_reasons.append(f"{row['doc_id']}: issuer {row['issuer_symbol']} is in frozen held-out test issuers")
                continue
            row["split"] = "train"
            train_rows.append(row)
            existing_ids.add(row["doc_id"])
            existing_fps.add(fp)
            added_count += 1

    train_texts, train_labels, sample_counts = build_training_samples(train_rows)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

    if arch == "modernbert-frozen":
        model, floor, margin, train_meta = train_frozen_head(train_texts, train_labels, cal_rows, tokenizer)
        display_name = "ModernBERT-base Frozen / Supervised Head"
    elif arch == "modernbert-finetuned":
        model, floor, margin, train_meta = train_finetuned(train_texts, train_labels, cal_rows, tokenizer)
        display_name = "ModernBERT-base Fine-Tuned"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    candidate_dir = MODELS_DIR / "candidates" / version_tag
    export_info = export_and_quantize(
        model,
        BASE_DIR,
        candidate_dir,
        {
            "name": display_name,
            "version": version_tag,
            "classes": list(VALID_LABELS),
            "floor": floor,
            "margin": margin,
        },
        train_texts[:6],
    )
    cmp_res = compare_quantized_vs_unquantized(candidate_dir, floor, margin, test_rows)
    with threadpool_limits(limits=1):
        adapter = DomainEventAdapter(ONNXEndToEndScorer(candidate_dir, "model_quantized.onnx"), floor=floor, margin=margin)
        repo_res = evaluate_repo_benchmarks(adapter.analyze)

    champion_test = registry.get("baseline_metrics", {})
    gates = check_promotion_gates(
        cmp_res["int8_test"],
        repo_res,
        export_info,
        champion_test,
        max_p95_ms=max_p95_ms,
        max_int8_bytes=int(max_int8_mb * 1024 * 1024),
    )

    eval_report = {
        "version": version_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arch": arch,
        "reviewed_additions_accepted": added_count,
        "reviewed_additions_rejected": rejected_reasons,
        "training_samples": len(train_texts),
        "training_sample_counts": sample_counts,
        "train_meta": train_meta,
        "export": export_info,
        "quantized_vs_unquantized": cmp_res,
        "repo_benchmarks": repo_res,
        "gates": gates,
    }
    (candidate_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not dry_run:
        registry["history"].append({
            "version": version_tag,
            "timestamp_utc": eval_report["timestamp_utc"],
            "arch": arch,
            "promoted": gates["promote"],
            "test_macro_f1": cmp_res["int8_test"]["macro_f1"],
            "test_review_recall": cmp_res["int8_test"]["review_recall"],
            "cpu_p95_ms": cmp_res["int8_test"]["latency_ms"]["p95"],
            "int8_bytes": export_info["int8_bytes"],
            "gates": gates,
        })
        if gates["promote"]:
            registry["active_version"] = version_tag
            registry["active_backend"] = "training.adapter.DomainEventAdapter"
            registry["active_model_dir"] = str(candidate_dir)
            registry["baseline_metrics"] = cmp_res["int8_test"]
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual gated retraining for StockBrief domain event classifier")
    parser.add_argument("--reviewed-jsonl", type=Path, default=None, help="Optional JSONL of human-reviewed annotations")
    parser.add_argument("--arch", choices=("modernbert-frozen", "modernbert-finetuned"), default="modernbert-frozen")
    parser.add_argument("--version", default=f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--max-p95-ms", type=float, default=150.0)
    parser.add_argument("--max-int8-mb", type=float, default=80.0)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without updating models/registry.json")
    args = parser.parse_args()

    res = run_retrain(
        args.reviewed_jsonl,
        args.arch,
        args.version,
        args.max_p95_ms,
        args.max_int8_mb,
        args.dry_run,
    )
    print(json.dumps({"version": res["version"], "gates": res["gates"], "rejected": res["reviewed_additions_rejected"]}, indent=2))
    if not res["gates"]["promote"]:
        print("Candidate did NOT pass all promotion gates; active default remains unchanged.", file=sys.stderr)


if __name__ == "__main__":
    main()
