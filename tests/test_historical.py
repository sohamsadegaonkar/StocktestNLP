from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stockbrief.historical import DEFAULT_PACKAGE_DIR, HistoricalIndex, IST
from stockbrief.historical_export import export_historical_package
from stockbrief.semantic import CHECKSUMS, WEIGHTS, load_semantic, model_directory


def _synthetic_package(root: Path, events: list[dict], dims: int = 384, vector_seed: int = 7) -> Path:
    rng = np.random.default_rng(vector_seed)
    vecs = rng.normal(size=(len(events), dims)).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    vecs = vecs.astype(np.float16)

    ev_path = root / "events.jsonl.gz"
    emb_path = root / "embeddings_f16.npy"
    with gzip.open(ev_path, "wt", encoding="utf-8") as out:
        for i, item in enumerate(events):
            row = {"emb_row": i, **item}
            out.write(json.dumps(row) + "\n")
    np.save(emb_path, vecs)

    from hashlib import file_digest

    def digest(p: Path) -> str:
        with p.open("rb") as f:
            return file_digest(f, "sha256").hexdigest()

    manifest = {
        "schema_version": "stockbrief-historical-v1",
        "package_version": "test-package-v1",
        "package_mode": "phase3_local",
        "encoder": {"weights_sha256": CHECKSUMS[WEIGHTS], "dim": dims, "dtype": "float16"},
        "record_count": len(events),
        "unique_issuers": len({e["symbol"] for e in events}),
        "files": {
            "events.jsonl.gz": {"sha256": digest(ev_path)},
            "embeddings_f16.npy": {"sha256": digest(emb_path)},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class HistoricalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_semantic(model_directory())

    def test_offline_demo_fallback_when_phase3_package_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "nonexistent_phase3_dir"
            with patch("stockbrief.historical.DEFAULT_PACKAGE_DIR", missing_dir):
                idx = HistoricalIndex(encoder=self.model.encoder, encoder_artifact=self.model.artifact)
        self.assertEqual(idx.mode, "offline_demo")
        result = idx.compare(
            query_text="Infosys reported fiscal first-quarter revenue growth and operating margin guidance.",
            symbol="INFY",
            category="earnings",
            disclosure_date="2026-08-01",
        )
        self.assertEqual(result["state"], "demo")
        self.assertEqual(result["package_mode"], "offline_demo")
        self.assertTrue(result["comparables"])
        self.assertTrue(result["missing_artifact_hint"])
        first = result["comparables"][0]
        self.assertIn("raw_return_pct", first["observed_1d"])
        self.assertIn("excess_vs_nifty50_pct", first["observed_1d"])
        self.assertIn("market_model_car_pct", first["observed_1d"])
        # Analytics-only contract: no predicted directional probabilities or trading signals.
        serialized = json.dumps(result).lower()
        for forbidden in ("buy_signal", "sell_signal", "prob_up", "target_price", "recommendation"):
            self.assertNotIn(forbidden, serialized)

    def test_strict_point_in_time_and_outcome_window_cutoff(self):
        events = [
            {
                "event_id": "nse:past_complete",
                "symbol": "INFY",
                "sector": "Technology",
                "published_ts_ist": "2026-06-01T16:30:00+05:30",
                "published_date": "2026-06-01",
                "t0_date": "2026-06-02",
                "window_1d_end_date": "2026-06-02",
                "window_5d_end_date": "2026-06-08",
                "category": "earnings",
                "category_name": "Earnings & guidance",
                "stage": "Reported",
                "source_url": "https://nsearchives.nseindia.in/past_complete.pdf",
                "excerpt": "Infosys reported strong quarterly revenue and updated full-year margin guidance.",
                "corp_action_verified": True,
                "market_data_provider": "Yahoo Finance (split/dividend adjusted)",
                "has_split_near": False,
                "has_div_near": False,
                "is_overlapping_10d": False,
                "benchmark": "NIFTY50",
                "observed_1d": {"raw_return_pct": 1.2, "excess_vs_nifty50_pct": 0.8, "market_model_car_pct": 0.75},
                "observed_5d": {"raw_return_pct": 2.4, "excess_vs_nifty50_pct": 1.6, "market_model_car_pct": 1.5},
            },
            {
                "event_id": "nse:recent_partial_window",
                "symbol": "INFY",
                "sector": "Technology",
                "published_ts_ist": "2026-07-01T11:00:00+05:30",
                "published_date": "2026-07-01",
                "t0_date": "2026-07-01",
                "window_1d_end_date": "2026-07-01",
                "window_5d_end_date": "2026-07-07",  # After 2026-07-03 cutoff -> MUST be excluded!
                "category": "earnings",
                "category_name": "Earnings & guidance",
                "stage": "Reported",
                "source_url": "https://nsearchives.nseindia.in/recent_partial.pdf",
                "excerpt": "Infosys reported quarterly digital revenue growth and raised margin guidance.",
                "corp_action_verified": False,
                "market_data_provider": "Upstox fallback (daily unadjusted closes)",
                "has_split_near": False,
                "has_div_near": False,
                "is_overlapping_10d": True,
                "benchmark": "NIFTY50",
                "observed_1d": {"raw_return_pct": 0.9, "excess_vs_nifty50_pct": 0.5, "market_model_car_pct": 0.4},
                "observed_5d": {"raw_return_pct": 9.9, "excess_vs_nifty50_pct": 9.5, "market_model_car_pct": 9.1},
            },
            {
                "event_id": "nse:same_day_future",
                "symbol": "INFY",
                "sector": "Technology",
                "published_ts_ist": "2026-07-03T15:00:00+05:30",
                "published_date": "2026-07-03",
                "t0_date": "2026-07-03",
                "window_1d_end_date": "2026-07-03",
                "window_5d_end_date": "2026-07-09",
                "category": "earnings",
                "category_name": "Earnings & guidance",
                "stage": "Reported",
                "source_url": "https://nsearchives.nseindia.in/same_day.pdf",
                "excerpt": "Infosys same-day filing that must be excluded by conservative cutoff.",
                "corp_action_verified": True,
                "market_data_provider": "Yahoo Finance (split/dividend adjusted)",
                "has_split_near": False,
                "has_div_near": False,
                "is_overlapping_10d": False,
                "benchmark": "NIFTY50",
                "observed_1d": {"raw_return_pct": -3.0, "excess_vs_nifty50_pct": -3.0, "market_model_car_pct": -3.0},
                "observed_5d": {"raw_return_pct": -5.0, "excess_vs_nifty50_pct": -5.0, "market_model_car_pct": -5.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _synthetic_package(Path(tmp), events)
            # Test with mismatched encoder -> falls back to category matching deterministically while enforcing all cutoffs!
            idx = HistoricalIndex(
                encoder=self.model.encoder,
                encoder_artifact={"weights_sha256": "deadbeef"},
                package_dir=pkg,
            )
            self.assertFalse(idx.encoder_compatible)
            res = idx.compare(
                query_text="Infosys quarterly revenue growth and margin guidance update.",
                symbol="INFY",
                category="earnings",
                disclosure_date="2026-07-03",
                exclude_url="https://nsearchives.nseindia.in/other.pdf",
            )
            self.assertEqual(res["matching_mode"], "category_fallback")
            self.assertEqual(res["cutoff_ist"], "2026-07-03T00:00:00+05:30")
            ids = [c["event_id"] for c in res["comparables"]]
            self.assertNotIn("nse:same_day_future", ids)
            self.assertIn("nse:past_complete", ids)
            self.assertIn("nse:recent_partial_window", ids)

            partial = next(c for c in res["comparables"] if c["event_id"] == "nse:recent_partial_window")
            self.assertIsNotNone(partial["observed_1d"])
            self.assertIsNone(partial["observed_5d"])
            self.assertEqual(partial["window_5d_status"], "unavailable_before_cutoff")
            self.assertFalse(partial["corp_action_verified"])

            # 5d distribution MUST exclude the 9.9% leakage from recent_partial_window!
            h5 = res["summary_distribution"]["horizon_5d"]
            self.assertEqual(h5["completed_count"], 1)
            self.assertEqual(h5["pending_after_cutoff_count"], 1)
            self.assertAlmostEqual(h5["raw_return_median_pct"], 2.4, places=2)

            # Querying by exact URL of past_complete excludes it cleanly.
            res_excl = idx.compare(
                query_text="Infosys quarterly revenue growth",
                symbol="INFY",
                category="earnings",
                disclosure_date="2026-07-03",
                exclude_url="https://nsearchives.nseindia.in/past_complete.pdf",
            )
            self.assertNotIn("nse:past_complete", [c["event_id"] for c in res_excl["comparables"]])

    def test_incompatible_corrupt_and_disabled_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _synthetic_package(Path(tmp), [{
                "event_id": "nse:1", "symbol": "TCS", "sector": "Technology",
                "published_ts_ist": "2025-01-10T10:00:00+05:30", "published_date": "2025-01-10",
                "t0_date": "2025-01-10", "window_1d_end_date": "2025-01-10", "window_5d_end_date": "2025-01-16",
                "category": "earnings", "category_name": "Earnings & guidance", "stage": "Reported",
                "source_url": "", "excerpt": "TCS announced quarterly earnings.",
                "corp_action_verified": False, "market_data_provider": "Upstox fallback (daily unadjusted closes)",
                "has_split_near": False, "has_div_near": False, "is_overlapping_10d": False,
                "benchmark": "NIFTY50",
                "observed_1d": {"raw_return_pct": 1.0, "excess_vs_nifty50_pct": 0.5, "market_model_car_pct": 0.4},
                "observed_5d": {"raw_return_pct": 2.0, "excess_vs_nifty50_pct": 1.2, "market_model_car_pct": 1.0},
            }])
            # Corrupt events checksum
            (pkg / "events.jsonl.gz").write_bytes(b"corrupt")
            broken = HistoricalIndex(encoder=self.model.encoder, package_dir=pkg)
            self.assertEqual(broken.mode, "unavailable")
            self.assertEqual(broken.compare(query_text="Quarterly financial results reported.")["state"], "unavailable")

        disabled = HistoricalIndex(encoder=self.model.encoder, disabled=True)
        self.assertEqual(disabled.compare(query_text="Quarterly financial results reported.")["state"], "disabled")

    def test_unrelated_review_query_returns_insufficient_evidence(self):
        idx = HistoricalIndex(encoder=self.model.encoder, encoder_artifact=self.model.artifact)
        res = idx.compare(
            query_text="Sunny weather in Bengaluru today and light traffic.",
            symbol="OTHER",
            category="review",
            disclosure_date="2026-08-01",
        )
        self.assertEqual(res["state"], "insufficient_evidence")
        self.assertEqual(res["comparables"], [])

    def test_full_phase3_local_package_if_present_matches_checksums_and_cutoff(self):
        if not (DEFAULT_PACKAGE_DIR / "manifest.json").is_file():
            self.skipTest("Optional local Phase 3 historical package not exported.")
        idx = HistoricalIndex(encoder=self.model.encoder, encoder_artifact=self.model.artifact)
        self.assertEqual(idx.mode, "phase3_local")
        self.assertTrue(idx.encoder_compatible)
        self.assertGreaterEqual(len(idx.records), 10_000)
        res = idx.compare(
            query_text="Tata Motors Passenger Vehicles announced an average price increase of 0.5% effective December 2026.",
            symbol="TMPV",
            category="pricing",
            disclosure_date="2026-11-24",
        )
        self.assertEqual(res["state"], "available")
        self.assertTrue(res["comparables"])
        for comp in res["comparables"]:
            self.assertLess(comp["published_ts_ist"], res["cutoff_ist"])
            if comp["observed_5d"] is not None:
                self.assertLess(comp["window_5d_end_date"], res["cutoff_date"])

    def test_exporter_reports_exact_missing_artifacts(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            export_historical_package(
                features_path=Path("nonexistent_features.parquet"),
                embs_path=Path("nonexistent_embs.npy"),
                indices_path=Path("nonexistent_indices.pkl"),
            )
        self.assertIn("nonexistent_features.parquet", str(ctx.exception))
