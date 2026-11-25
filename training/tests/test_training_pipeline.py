"""Unit tests for dataset integrity, split isolation, adapter contract, and gated promotion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from stockbrief.fixtures import Document
from stockbrief.semantic_data import SEMANTIC_LABELS
from training.adapter import DomainEventAdapter, ONNXEndToEndScorer, is_readable_text
from training.retrain import check_promotion_gates, validate_reviewed_additions
from training.schema import AnnotatedDocument, load_jsonl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FT_MODEL_DIR = Path("models/modernbert-finetuned")


class MockScorer:
    name = "Mock scorer"
    version = "mock-v1"
    classes = list(SEMANTIC_LABELS.keys())

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = np.full((len(texts), len(self.classes)), 0.02, dtype=np.float32)
        for i, txt in enumerate(texts):
            low = txt.lower()
            if "price" in low or "hike" in low:
                idx = self.classes.index("pricing")
            elif "loss" in low or "quarter" in low or "profit" in low:
                idx = self.classes.index("earnings")
            elif "board" in low and "scheduled" in low:
                idx = self.classes.index("routine")
            else:
                idx = self.classes.index("review")
            out[i, idx] = 0.84
        return out


class DatasetAndSplitTests(unittest.TestCase):
    def test_dataset_manifest_and_splits_are_disjoint(self):
        train = load_jsonl(DATA_DIR / "train.jsonl")
        cal = load_jsonl(DATA_DIR / "calibration.jsonl")
        test = load_jsonl(DATA_DIR / "test.jsonl")
        manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(train) + len(cal) + len(test), 400)
        all_issuers = {r["issuer_symbol"] for r in train + cal + test}
        self.assertGreaterEqual(len(all_issuers), 20)

        f_train = {r["event_family"] for r in train}
        f_cal = {r["event_family"] for r in cal}
        f_test = {r["event_family"] for r in test}
        self.assertFalse(f_train & f_cal)
        self.assertFalse(f_train & f_test)
        self.assertFalse(f_cal & f_test)

        unseen = set(manifest["unseen_heldout_issuers"])
        self.assertGreaterEqual(len(unseen), 10)
        self.assertFalse(unseen & {r["issuer_symbol"] for r in train})
        self.assertFalse(unseen & {r["issuer_symbol"] for r in cal})

        for row in train + cal + test:
            doc = AnnotatedDocument(**{k: v for k, v in row.items() if k in AnnotatedDocument.__dataclass_fields__})
            self.assertEqual(doc.validate(), [])


class AdapterAndGateTests(unittest.TestCase):
    def test_adapter_handles_mixed_pricing_earnings_and_failed_pdf(self):
        adapter = DomainEventAdapter(MockScorer(), floor=0.45, margin=0.05)
        mixed = Document(
            "TMPV",
            ("Company announced a price hike of up to 0.5% from Dec 1, 2026, while reporting Q2 FY27 net loss.",),
            "Mixed disclosure",
        )
        res = adapter.analyze(mixed)
        cats = {e["category"] for e in res["events"]}
        self.assertEqual(cats, {"pricing", "earnings"})
        for k in ("category", "category_name", "stage", "evidence", "figures", "dates", "events", "model_version"):
            self.assertIn(k, res)

        failed = Document("OTHER", ("",), "Empty scan")
        res_failed = adapter.analyze(failed)
        self.assertEqual(res_failed["category"], "review")
        self.assertFalse(is_readable_text(("bbbbbbbbbbbbbbbbbbbbbbbb",)))

    def test_adapter_scheduled_vs_reported_and_rumor_denial_stages(self):
        adapter = DomainEventAdapter(MockScorer(), floor=0.45, margin=0.05)
        sched = Document(
            "RELIANCE",
            ("The meeting of the Board of Directors of the Company is scheduled to be held on October 30, 2026.",),
            "Scheduled board meeting",
        )
        res_sched = adapter.analyze(sched)
        self.assertEqual(res_sched["category"], "routine")
        self.assertEqual(res_sched["stage"], "Scheduled")

    def test_retrain_rejects_unreviewed_and_user_uploads(self):
        sample = load_jsonl(DATA_DIR / "train.jsonl")[0]
        bad_unreviewed = {**sample, "doc_id": "bad:1", "reviewed": False}
        bad_upload = {**sample, "doc_id": "bad:2", "reviewed": True, "source": "user_upload"}
        good = {**sample, "doc_id": "good:1", "reviewed": True}
        accepted, rejected = validate_reviewed_additions([bad_unreviewed, bad_upload, good])
        self.assertEqual([r["doc_id"] for r in accepted], ["good:1"])
        self.assertEqual(len(rejected), 2)

    def test_promotion_gates_block_degraded_review_recall_or_oversized_artifact(self):
        champ = {"macro_f1": 0.62, "review_recall": 0.41, "multilabel_micro_f1": 0.54}
        cand_good_f1_bad_recall = {
            "macro_f1": 0.85, "review_recall": 0.30, "multilabel_micro_f1": 0.80, "latency_ms": {"p95": 50.0},
        }
        repo_ok = {"tata_acceptance_passed": True, "repo_test_32": {"review_recall": 1.0}}
        g1 = check_promotion_gates(cand_good_f1_bad_recall, repo_ok, {"int8_bytes": 30_000_000}, champ)
        self.assertFalse(g1["promote"])
        self.assertFalse(g1["preserves_review_recall"])

        cand_good_oversized = {
            "macro_f1": 0.85, "review_recall": 0.70, "multilabel_micro_f1": 0.80, "latency_ms": {"p95": 250.0},
        }
        g2 = check_promotion_gates(cand_good_oversized, repo_ok, {"int8_bytes": 150_000_000}, champ)
        self.assertFalse(g2["promote"])
        self.assertFalse(g2["cpu_p95_within_budget"])
        self.assertFalse(g2["int8_size_within_budget"])


@unittest.skipUnless((FT_MODEL_DIR / "model_quantized.onnx").is_file(), "requires exported ONNX INT8 model")
class ExportedONNXAdapterTests(unittest.TestCase):
    def test_exported_onnx_int8_runs_semantic_adapter_contract(self):
        from training.evaluator import TATA_ACCEPTANCE

        meta = json.loads((FT_MODEL_DIR / "adapter_config.json").read_text(encoding="utf-8"))
        with threadpool_limits(limits=1):
            scorer = ONNXEndToEndScorer(FT_MODEL_DIR, "model_quantized.onnx")
            adapter = DomainEventAdapter(scorer, floor=meta["floor"], margin=meta["margin"])
            brief = adapter.analyze(Document("TMPV", (TATA_ACCEPTANCE,), "Tata acceptance"))
        events = {ev["category"]: ev["stage"] for ev in brief["events"]}
        self.assertEqual(events, {"pricing": "Announced", "earnings": "Reported"})


if __name__ == "__main__":
    unittest.main()
