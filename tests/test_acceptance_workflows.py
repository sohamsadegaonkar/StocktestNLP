import asyncio
from datetime import datetime
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter
from threadpoolctl import threadpool_limits

from stockbrief.app import create_app
from stockbrief.fixtures import ACQUISITION, Document
from stockbrief.historical import HistoricalIndex
from stockbrief.intelligence import build_intelligence
from stockbrief.market_context import disclosure_context
from stockbrief.nlp import EventModel
from stockbrief.semantic import load_semantic, model_directory
from stockbrief.store import Inbox
from tests.test_live_feed import feed


def _scanned_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


class AcceptanceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.semantic = load_semantic(model_directory())

    async def test_legacy_sqlite_database_upgrades_and_preserves_entries(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db_path = Path(tmp) / "legacy.sqlite"
            # Populate using legacy TF-IDF EventModel first, then strip newer tables to mimic an earlier schema.
            old_inbox = Inbox(db_path, EventModel())
            old_entry, _ = old_inbox.record(
                Document("INFY", ("Infosys completed an acquisition of an engineering firm.",), "Legacy entry", origin="text")
            )
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("DROP TABLE IF EXISTS live_receipts")
                conn.execute("DROP TABLE IF EXISTS edge_state")
                conn.execute("DROP TABLE IF EXISTS edge_jobs")
                conn.commit()
            finally:
                conn.close()

            app = await asyncio.to_thread(create_app, db_path, model=self.semantic)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                inbox = (await client.get("/api/inbox")).json()
                ids = {item["id"] for item in inbox["disclosures"]}
                self.assertIn(old_entry["id"], ids)
                comp = await client.get(f"/api/disclosures/{old_entry['id']}/comparables")
                self.assertEqual(comp.status_code, 200)
                analyzed = await client.post("/api/analyze", data={
                    "ticker": "INFY",
                    "disclosure_date": "2026-07-17",
                    "text": "Infosys reported quarterly revenue growth of 4.1% and revised full-year guidance.",
                })
                self.assertEqual(analyzed.status_code, 200)
                new_id = analyzed.json()["entry"]["id"]

            reopened = Inbox(db_path, self.semantic)
            reopened_ids = {item["id"] for item in reopened.state()["disclosures"]}
            self.assertIn(old_entry["id"], reopened_ids)
            self.assertIn(new_id, reopened_ids)

    async def test_tata_mixed_rumor_denial_scheduled_vs_reported_and_unrelated_inputs(self):
        tata_mixed = (
            "Tata Motors Passenger Vehicles Limited announced an average price increase of 0.5% across its passenger vehicle portfolio effective December 1, 2026. "
            "In November, the company reported consolidated Q2 quarterly financial results with JLR net profit impacted by restructuring costs."
        )
        rumor = (
            "Clarification on media report: The company denies the market speculation regarding any buyout transaction "
            "and confirms no agreement has been signed."
        )
        scheduled = (
            "Notice is hereby given that a meeting of the Board of Directors is scheduled to be held on January 16, 2027 "
            "to consider and approve the unaudited financial results for the quarter ended December 31, 2026."
        )
        unrelated = "Sunny weather across Bengaluru today with light evening traffic near Whitefield."

        with tempfile.TemporaryDirectory() as tmp:
            app = await asyncio.to_thread(create_app, Path(tmp) / "app.sqlite", model=self.semantic)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                r_mixed = await client.post("/api/analyze", data={
                    "ticker": "TMPV", "disclosure_date": "2026-11-24", "text": tata_mixed,
                })
                self.assertEqual(r_mixed.status_code, 200)
                m_entry = r_mixed.json()["entry"]
                m_events = {e["category"]: e for e in m_entry["brief"]["events"]}
                self.assertEqual(set(m_events), {"pricing", "earnings"})
                self.assertEqual(m_events["pricing"]["stage"], "Announced")
                self.assertEqual(m_events["earnings"]["stage"], "Reported")
                self.assertIn("0.5%", [f["value"] for f in m_events["pricing"]["figures"]])
                self.assertIn("December 1, 2026", [d["value"] for d in m_events["pricing"]["dates"]])

                r_rumor = await client.post("/api/analyze", data={"ticker": "OTHER", "text": rumor})
                self.assertEqual(r_rumor.status_code, 200)
                rumor_brief = r_rumor.json()["entry"]["brief"]
                self.assertIn(rumor_brief["stage"], ("Denied", "Unclear"))
                rumor_intel = (await client.get(f"/api/disclosures/{r_rumor.json()['entry']['id']}/intelligence")).json()
                self.assertTrue(
                    rumor_intel["state"] == "needs_review"
                    or "do not treat" in (rumor_intel["events"][0]["interpretation"] or "")
                )

                r_sched = await client.post("/api/analyze", data={"ticker": "INFY", "text": scheduled})
                self.assertEqual(r_sched.status_code, 200)
                self.assertNotEqual(
                    (r_sched.json()["entry"]["brief"]["category"], r_sched.json()["entry"]["brief"]["stage"]),
                    ("earnings", "Reported"),
                )

                r_unrel = await client.post("/api/analyze", data={"ticker": "OTHER", "text": unrelated})
                self.assertEqual(r_unrel.status_code, 200)
                unrel_id = r_unrel.json()["entry"]["id"]
                self.assertEqual(r_unrel.json()["entry"]["brief"]["category"], "review")
                self.assertIn("unrelated", r_unrel.json()["entry"]["brief"]["review_message"].lower())
                unrel_intel = (await client.get(f"/api/disclosures/{unrel_id}/intelligence")).json()
                self.assertEqual(unrel_intel["state"], "needs_review")
                self.assertTrue(unrel_intel["events"][0].get("missing_guidance"))
                unrel_comp = (await client.get(f"/api/disclosures/{unrel_id}/comparables")).json()
                self.assertEqual(unrel_comp["state"], "insufficient_evidence")

                # Scanned / blank PDF should give clear actionable guidance
                r_scanned = await client.post(
                    "/api/analyze",
                    data={"ticker": "INFY"},
                    files={"file": ("scan.pdf", _scanned_blank_pdf(), "application/pdf")},
                )
                self.assertEqual(r_scanned.status_code, 422)
                self.assertIn("scanned", r_scanned.json()["detail"].lower())

    async def test_live_feed_exposes_distinct_check_success_and_publication_timestamps(self):
        rss_bytes = feed(
            description="Infosys announced an agreement to acquire an engineering services firm.",
            published="Tue, 24 Nov 2026 19:15:00 +0530",
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = await asyncio.to_thread(
                create_app, Path(tmp) / "live.sqlite", model=self.semantic, live_fetch=lambda u, l: rss_bytes
            )
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                r = (await client.post("/api/live/refresh")).json()
                self.assertEqual(r["state"], "available")
                self.assertTrue(r["checked_at"])
                self.assertTrue(r["last_success"])
                self.assertEqual(r["latest_supported_at"], "2026-11-24T19:15:00+05:30")

    async def test_offline_mode_with_no_network_and_absent_phase3_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"STOCKBRIEF_DISABLE_MARKET": "1"}):
                with patch("stockbrief.historical.DEFAULT_PACKAGE_DIR", Path(tmp) / "missing_phase3"):
                    hist = HistoricalIndex(encoder=self.semantic.encoder, encoder_artifact=self.semantic.artifact)
                    app = await asyncio.to_thread(
                        create_app, Path(tmp) / "offline.sqlite", model=self.semantic, historical=hist
                    )
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                inbox = (await client.get("/api/inbox")).json()
                self.assertTrue(inbox["historical_available"])
                self.assertEqual(inbox["historical_mode"], "offline_demo")
                self.assertEqual(inbox["market_tickers"], [])
                first_id = inbox["disclosures"][0]["id"]
                comp = (await client.get(f"/api/disclosures/{first_id}/comparables")).json()
                self.assertEqual(comp["state"], "demo")
                self.assertTrue(comp["comparables"])
                example = (await client.get("/api/market-example")).json()
                self.assertEqual(example["state"], "recorded")
                sim = (await client.post("/api/edge/simulate", json={"count": 12, "scenario": "source_outage"})).json()
                self.assertEqual(sim["runs"]["cooperative"]["completed"], 12)
                bench = (await client.post("/api/edge/benchmark")).json()
                self.assertTrue(bench["same_categories"])
