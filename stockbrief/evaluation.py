"""Repeatable fixture evaluation and host measurements."""
from importlib.metadata import version
import os
import platform
from time import perf_counter_ns

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .corpus import CALIBRATION, LABELS, TEST, TRAIN
from .fixtures import CASE_STUDY, Document
from .nlp import MODEL, EventModel, baseline
from .semantic import SemanticEventModel
from .semantic_data import SEMANTIC_ADDITIONAL_TEST, SEMANTIC_CALIBRATION, SEMANTIC_TRAIN

CLASSES = [*LABELS, "review"]


def host_metrics():
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "process_rss_mib": round(psutil.Process().memory_info().rss / (1024**2), 1),
        "available_logical_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "scope": "Host process, including Python and loaded libraries",
        "python": platform.python_version(),
        "scikit_learn": version("scikit-learn"),
    }


def assess(name, predictor, pairs, classes=CLASSES):
    expected = [label for label, _ in pairs]
    predicted, samples = [], []
    for _, document in pairs:
        started = perf_counter_ns()
        predicted.append(predictor(document))
        samples.append((perf_counter_ns() - started) / 1e6)
    matrix_labels = [*classes, *sorted(set(predicted) - set(classes))]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=matrix_labels, zero_division=0,
    )
    covered = [(a, b) for a, b in zip(expected, predicted) if b != "review"]
    review = [(a, b) for a, b in zip(expected, predicted) if a == "review"]
    return {
        "name": name,
        "macro_f1": round(float(f1[:len(classes)].mean()), 3),
        "coverage": round(len(covered) / len(expected), 3),
        "selective_accuracy": round(sum(a == b for a, b in covered) / len(covered), 3) if covered else None,
        "review_recall": round(sum(a == b for a, b in review) / len(review), 3) if review else None,
        "latency_ms": {"p50": round(float(np.median(samples)), 3), "p95": round(float(np.percentile(samples, 95)), 3)},
        "matrix": confusion_matrix(expected, predicted, labels=matrix_labels).tolist(),
        "classes": [
            {"label": label, "precision": round(float(p), 3), "recall": round(float(r), 3),
             "f1": round(float(f), 3), "support": int(n)}
            for label, p, r, f, n in zip(matrix_labels, precision, recall, f1, support)
        ],
    }


def evaluate(model, transformer=None):
    pairs = [
        (label, Document("OTHER", (text,), "Held-out fixture"))
        for label, texts in TEST.items()
        for text in texts
    ]
    semantic = isinstance(model, SemanticEventModel)
    legacy = EventModel() if semantic else model
    reports = [
        assess("Keyword baseline", lambda doc: baseline(doc.pages[0]), pairs),
        assess(MODEL, lambda doc: legacy.analyze(doc)["category"], pairs),
    ]
    if semantic:
        predictor = lambda doc: model.analyze(doc)["category"]
        primary = assess(model.name, predictor, pairs)
        primary["artifact"] = model.artifact
        extra = [(label, Document("OTHER", (phrase,), "Additional authored fixture"))
                 for label, examples in SEMANTIC_ADDITIONAL_TEST.items() for phrase in examples]
        primary["additional_pricing_and_review"] = assess(
            model.name, predictor, extra, list(SEMANTIC_ADDITIONAL_TEST),
        )
        reports.append(primary)
    cases = []
    for document, category, stage in CASE_STUDY:
        brief = model.analyze(document)
        cases.append({
            "title": document.title,
            "source": document.url,
            "expected_category": category,
            "observed_category": brief["category"],
            "expected_stage": stage,
            "observed_stage": brief["stage"],
            "matched": brief["category"] == category and brief["stage"] == stage,
        })
    if transformer is not None:
        report = assess(transformer.name, transformer.classify, pairs)
        report["historical_cases"] = []
        for document, category, _ in CASE_STUDY:
            observed = transformer.classify(document)
            report["historical_cases"].append({
                "title": document.title,
                "source": document.url,
                "expected_category": category,
                "observed_category": observed,
                "category_matched": observed == category,
            })
        artifact = getattr(transformer, "artifact", None)
        if artifact is not None:
            report["artifact"] = artifact
        reports.append(report)
    return {
        "dataset": {
            "train": sum(map(len, (SEMANTIC_TRAIN if semantic else TRAIN).values())),
            "calibration": sum(map(len, (SEMANTIC_CALIBRATION if semantic else CALIBRATION).values())),
            "test": len(pairs), "scope": "Authored English-language fixtures with disjoint splits",
        },
        "thresholds": {"floor": model.floor, "margin": model.margin},
        "labels": {**getattr(model, "labels", LABELS), "review": "Needs review"},
        "reports": reports,
        "historical_cases": cases,
        "host": host_metrics(),
    }
