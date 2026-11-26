"""Automated regression & zero-leakage tests for Phase 3 Market Intelligence pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from phase3.adapter import MarketIntelligenceEngine
from phase3.similarity_engine import HistoricalSimilarityEngine


class Phase3AuditAndProvenanceTests(unittest.TestCase):
    def test_leakage_audit_zero_overlap_and_embargo(self) -> None:
        audit_path = Path("phase3/data/leakage_audit.json")
        self.assertTrue(audit_path.is_file())
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["event_id_overlap"], 0)
        self.assertEqual(audit["near_duplicate_text_overlap"], 0)
        self.assertGreater(audit["embargo_sessions_train_to_val"], 20)
        self.assertGreater(audit["embargo_sessions_val_to_test"], 20)
        self.assertTrue(audit["svd_fit_strictly_on_train_only"])
        self.assertGreaterEqual(audit["pit_fundamentals_lag_days"], 60)

    def test_market_provenance_and_event_study_integrity(self) -> None:
        prov = json.loads(Path("phase3/data/market_provenance.json").read_text(encoding="utf-8"))
        self.assertIn("never represented as live", prov["live_feed_warning"].lower())
        es = json.loads(Path("phase3/results/event_study_report.json").read_text(encoding="utf-8"))
        self.assertGreater(es["overall"]["count"], 10000)
        self.assertGreater(es["non_overlapping_10d_overall"]["count"], 5000)
        for h in ("1d", "3d", "5d", "10d", "20d"):
            self.assertIn(h, es["overall"]["horizons"])

    def test_previous_phases_preserved(self) -> None:
        r1 = json.loads(Path("models/registry.json").read_text(encoding="utf-8"))
        self.assertEqual(r1["active_model_dir"], "models/semantic")
        self.assertTrue(Path("models/phase2/registry_phase2.json").is_file())


class SimilarityAndAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MarketIntelligenceEngine(Path("models"), Path("phase3/results"))

    def test_similarity_engine_strict_temporal_cutoff(self) -> None:
        cutoff = "2024-06-15T12:00:00+05:30"
        q = self.engine.similarity.embs[100]
        res = self.engine.similarity.retrieve(q, as_of_ts_ist=cutoff, top_k=8)
        self.assertGreater(len(res["comparables"]), 0)
        for comp in res["comparables"]:
            self.assertLess(comp["published_ts_ist"], cutoff)

        # Unrelated random/high-threshold vector abstains cleanly
        noise = np.ones(384, dtype=np.float32) * 0.01
        noise[0] = -1.0
        bad = self.engine.similarity.retrieve(noise, as_of_ts_ist=cutoff, min_similarity=0.80)
        self.assertEqual(bad["comparables"], [])
        self.assertEqual(bad["warning"], "insufficient_historical_precedents")

    def test_adapter_contract_and_ood_abstention(self) -> None:
        res = self.engine.analyze(
            text="Infosys Limited bagged a USD 450 million multi-year digital transformation order across European markets.",
            symbol="INFY",
            sector="Technology",
            event_family_hint="operations",
            source_url="https://www.nseindia.com/demo",
            as_of_ts_ist="2026-08-01T10:00:00+05:30",
            market_snapshot={"ctx_nifty_ret_60d": 0.06, "ctx_vix_level": 13.5, "tech_ret_5d": 0.02, "tech_vol20": 0.014},
            fundamentals_snapshot={"has_pit_fundamentals": 1.0, "fund_op_margin": 0.21},
        )
        self.assertEqual(res["model_version"], "stockbrief-phase3-market-intel-v1")
        self.assertIn("historical_facts", res)
        self.assertIn("predictions", res)
        self.assertIn("chart_data_contracts", res)
        self.assertTrue(len(res["historical_facts"]["historical_comparable_events"]) > 0)
        for c in res["historical_facts"]["historical_comparable_events"]:
            self.assertLess(c["published_ts_ist"], "2026-08-01T10:00:00+05:30")

        # Non-financial / unfamiliar input triggers abstention or low-evidence warning
        ood = self.engine.analyze(
            text="Bake chocolate muffins at 180C with two cups flour and sugar.",
            symbol="OTHER",
            as_of_ts_ist="2022-01-01T09:00:00+05:30",
        )
        self.assertTrue(ood["predictions"]["abstain"])


if __name__ == "__main__":
    unittest.main()
