"""Unified evaluation harness for BGE baseline, PyTorch encoders, and ONNX INT8 models."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from stockbrief.corpus import TEST as REPO_TEST
from stockbrief.fixtures import CASE_STUDY, Document
from stockbrief.semantic_data import SEMANTIC_ADDITIONAL_TEST, SEMANTIC_LABELS
from training.schema import VALID_LABELS, load_jsonl

CLASSES = list(VALID_LABELS)
TATA_ACCEPTANCE = (
    "Tata Motors Passenger Vehicles (TMPV) has announced a price hike of up to 0.5% across its ICE and EV lineup "
    "starting December 1, 2026, while reporting a Q2 FY27 net loss driven by restructuring costs at Jaguar Land Rover (JLR).\n\n"
    "December 2026 Price Hike & Offers\n\n"
    "• Price revision: TMPV will raise prices by an average of 0.5% from December 1, 2026, citing rising input costs.\n"
    "• Estimated impact: Buyers may see increases ranging from roughly ₹2,384 to ₹14,645 depending on the model and variant.\n"
    "• Pre-hike discounts: Year-end offers and exchange/scrappage benefits in late November reach up to ₹1.25 lakh on "
    "select Curvv variants, ₹1.15 lakh on Sierra diesel, and ₹75,000 on select Nexon petrol manual trimsls."
)


def rss_mib() -> float:
    return round(psutil.Process().memory_info().rss / (1024**2), 1)


def safe_analyze(analyze_fn: Callable[[Document], dict[str, Any]], doc: Document) -> dict[str, Any]:
    """Handle empty/failed-PDF extraction safely without crashing evaluation."""
    try:
        return analyze_fn(doc)
    except ValueError:
        return {
            "category": "review",
            "category_name": SEMANTIC_LABELS["review"],
            "score": 0.0,
            "stage": "Unclear",
            "stage_evidence": None,
            "evidence": [],
            "figures": [],
            "dates": [],
            "events": [],
        }


def span_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def evaluate_domain_split(
    name: str,
    analyze_fn: Callable[[Document], dict[str, Any]],
    rows: list[dict[str, Any]],
    measure_latency: bool = True,
) -> dict[str, Any]:
    expected_primary: list[str] = []
    predicted_primary: list[str] = []
    latencies: list[float] = []

    y_true_multi = np.zeros((len(rows), len(CLASSES)), dtype=int)
    y_pred_multi = np.zeros((len(rows), len(CLASSES)), dtype=int)

    evidence_total = 0
    evidence_hits = 0
    offset_valid_total = 0
    offset_valid_ok = 0

    stage_correct = 0
    numeric_total = 0
    numeric_errors = 0
    negation_total = 0
    negation_errors = 0
    sched_total = 0
    sched_errors = 0
    failed_pdf_total = 0
    failed_pdf_correct = 0
    mixed_pe_total = 0
    mixed_pe_correct = 0
    unseen_total = 0
    unseen_correct = 0

    peak_rss = rss_mib()

    for idx, row in enumerate(rows):
        doc = Document(
            ticker=row["issuer_symbol"],
            pages=tuple(row["pages"]),
            title=row.get("nse_desc") or row["doc_id"],
            source=row["source"],
            date=row["date"],
            url=row["url"],
            origin="historical",
        )
        t0 = perf_counter_ns()
        brief = safe_analyze(analyze_fn, doc)
        dt_ms = (perf_counter_ns() - t0) / 1e6
        if measure_latency:
            latencies.append(dt_ms)
        peak_rss = max(peak_rss, rss_mib())

        exp_lbl = row["label"]
        pred_lbl = brief["category"]
        expected_primary.append(exp_lbl)
        predicted_primary.append(pred_lbl)

        # Multilabel vectors
        for lbl in row["labels"]:
            if lbl in CLASSES:
                y_true_multi[idx, CLASSES.index(lbl)] = 1
        pred_events = [ev["category"] for ev in brief.get("events", []) if ev.get("category") in CLASSES]
        if not pred_events:
            pred_events = [pred_lbl] if pred_lbl in CLASSES else ["review"]
        for lbl in pred_events:
            y_pred_multi[idx, CLASSES.index(lbl)] = 1

        # Stage accuracy
        if brief.get("stage") == row["stage"]:
            stage_correct += 1

        # Evidence offset validity & overlap accuracy
        full_page = row["pages"][0] if row["pages"] else ""
        pred_quotes = brief.get("evidence", [])
        for ev in brief.get("events", []):
            pred_quotes.extend(ev.get("evidence", []))
        for q in pred_quotes:
            if isinstance(q, dict) and "start" in q and "end" in q and "text" in q:
                offset_valid_total += 1
                p_idx = int(q.get("page", 1)) - 1
                page_txt = row["pages"][p_idx] if 0 <= p_idx < len(row["pages"]) else ""
                if page_txt[q["start"]:q["end"]] == q["text"]:
                    offset_valid_ok += 1

        for gt_span in row.get("evidence_offsets", []):
            evidence_total += 1
            hit = False
            for q in pred_quotes:
                if span_overlap(gt_span["start"], gt_span["end"], q.get("start", -1), q.get("end", -1)) >= 0.25:
                    hit = True
                    break
                if gt_span["text"] in q.get("text", "") or q.get("text", "") in gt_span["text"]:
                    if q.get("text"):
                        hit = True
                        break
            if hit:
                evidence_hits += 1

        # Numeric & qualifier preservation check (non-review docs with amounts)
        gt_amounts = row.get("preserves", {}).get("amounts", [])
        if gt_amounts and row["label"] != "review":
            numeric_total += 1
            extracted_fig_text = " ".join(
                f.get("value", "") for f in brief.get("figures", [])
            ) + " " + " ".join(
                f.get("value", "") for ev in brief.get("events", []) for f in ev.get("figures", [])
            )
            evidence_blob = " ".join(q.get("text", "") for q in pred_quotes)
            # Check that at least one ground-truth amount is preserved in figures or cited evidence
            norm_blob = (extracted_fig_text + " " + evidence_blob).lower()
            if not any(amt.lower() in norm_blob for amt in gt_amounts):
                numeric_errors += 1

        # Negation / rumor denial check
        if row["insufficiency_type"] == "rumor_denial" or row["stage"] == "Denied":
            negation_total += 1
            if row["stage"] == "Denied":
                if brief.get("stage") != "Denied" and pred_lbl != "review":
                    negation_errors += 1
            elif row["label"] == "review":
                if pred_lbl != "review":
                    negation_errors += 1

        # Scheduled vs reported check
        if row["stage"] == "Scheduled":
            sched_total += 1
            if pred_lbl != "routine" or brief.get("stage") != "Scheduled":
                sched_errors += 1

        # Failed PDF check
        if row["source_mode"] == "pdf_extraction_failed":
            failed_pdf_total += 1
            if pred_lbl == "review":
                failed_pdf_correct += 1

        # Mixed pricing+earnings check
        if set(row["labels"]) >= {"pricing", "earnings"}:
            mixed_pe_total += 1
            if set(pred_events) >= {"pricing", "earnings"}:
                mixed_pe_correct += 1

    precision, recall, f1, support = precision_recall_fscore_support(
        expected_primary, predicted_primary, labels=CLASSES, zero_division=0,
    )
    covered = [(a, b) for a, b in zip(expected_primary, predicted_primary) if b != "review"]
    review_idx = CLASSES.index("review")

    multilabel_micro_f1 = float(f1_score(y_true_multi, y_pred_multi, average="micro", zero_division=0))
    multilabel_macro_f1 = float(f1_score(y_true_multi, y_pred_multi, average="macro", zero_division=0))

    return {
        "name": name,
        "count": len(rows),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float(np.average(f1, weights=np.maximum(support, 1))), 4),
        "coverage": round(len(covered) / max(1, len(expected_primary)), 4),
        "selective_accuracy": round(sum(a == b for a, b in covered) / len(covered), 4) if covered else None,
        "review_precision": round(float(precision[review_idx]), 4),
        "review_recall": round(float(recall[review_idx]), 4),
        "review_f1": round(float(f1[review_idx]), 4),
        "multilabel_micro_f1": round(multilabel_micro_f1, 4),
        "multilabel_macro_f1": round(multilabel_macro_f1, 4),
        "stage_accuracy": round(stage_correct / max(1, len(rows)), 4),
        "evidence_accuracy": round(evidence_hits / max(1, evidence_total), 4),
        "offset_exact_valid_rate": round(offset_valid_ok / max(1, offset_valid_total), 4),
        "numeric_error_rate": round(numeric_errors / max(1, numeric_total), 4),
        "numeric_errors": f"{numeric_errors}/{numeric_total}",
        "negation_error_rate": round(negation_errors / max(1, negation_total), 4),
        "negation_errors": f"{negation_errors}/{negation_total}",
        "scheduled_error_rate": round(sched_errors / max(1, sched_total), 4),
        "scheduled_errors": f"{sched_errors}/{sched_total}",
        "failed_pdf_accuracy": f"{failed_pdf_correct}/{failed_pdf_total}",
        "mixed_pricing_earnings_exact": f"{mixed_pe_correct}/{mixed_pe_total}",
        "latency_ms": {
            "p50": round(float(np.median(latencies)), 3) if latencies else None,
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
            "mean": round(float(np.mean(latencies)), 3) if latencies else None,
        },
        "peak_rss_mib": peak_rss,
        "classes": [
            {
                "label": lbl,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(fv), 4),
                "support": int(sup),
            }
            for lbl, p, r, fv, sup in zip(CLASSES, precision, recall, f1, support)
        ],
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(expected_primary, predicted_primary, labels=CLASSES).tolist(),
    }


def evaluate_repo_benchmarks(analyze_fn: Callable[[Document], dict[str, Any]]) -> dict[str, Any]:
    """Evaluate untouched repo benchmarks: TEST (32), SEMANTIC_ADDITIONAL_TEST (8), CASE_STUDY (4), TATA."""
    test_pairs = [(lbl, Document("OTHER", (txt,), "Held-out fixture")) for lbl, txts in REPO_TEST.items() for txt in txts]
    exp_t = [lbl for lbl, _ in test_pairs]
    pred_t = [safe_analyze(analyze_fn, doc)["category"] for _, doc in test_pairs]
    t_classes = [lbl for lbl in CLASSES if lbl != "pricing"]
    p, r, f, _ = precision_recall_fscore_support(exp_t, pred_t, labels=t_classes, zero_division=0)
    rev_idx = t_classes.index("review")

    add_pairs = [(lbl, Document("OTHER", (txt,), "Additional fixture")) for lbl, txts in SEMANTIC_ADDITIONAL_TEST.items() for txt in txts]
    exp_a = [lbl for lbl, _ in add_pairs]
    pred_a = [safe_analyze(analyze_fn, doc)["category"] for _, doc in add_pairs]
    pa, ra, fa, _ = precision_recall_fscore_support(exp_a, pred_a, labels=list(SEMANTIC_ADDITIONAL_TEST), zero_division=0)

    cases = []
    for doc, exp_cat, exp_stg in CASE_STUDY:
        b = safe_analyze(analyze_fn, doc)
        cases.append({
            "title": doc.title,
            "expected": [exp_cat, exp_stg],
            "observed": [b["category"], b["stage"]],
            "matched": b["category"] == exp_cat and b["stage"] == exp_stg,
        })

    tata_brief = safe_analyze(analyze_fn, Document("OTHER", (TATA_ACCEPTANCE,), "Tata Motors acceptance"))
    tata_events = {e["category"]: e["stage"] for e in tata_brief.get("events", [])}

    return {
        "repo_test_32": {
            "macro_f1": round(float(f.mean()), 4),
            "review_recall": round(float(r[rev_idx]), 4),
            "review_precision": round(float(p[rev_idx]), 4),
        },
        "repo_additional_8": {
            "macro_f1": round(float(fa.mean()), 4),
            "pricing_f1": round(float(fa[0]), 4),
            "review_f1": round(float(fa[1]), 4),
        },
        "case_study_matched": f"{sum(c['matched'] for c in cases)}/{len(cases)}",
        "tata_acceptance_events": tata_events,
        "tata_acceptance_passed": tata_events == {"pricing": "Announced", "earnings": "Reported"},
    }
