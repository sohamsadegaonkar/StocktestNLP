import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from httpx import ASGITransport, AsyncClient
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from threadpoolctl import threadpool_limits

from stockbrief.app import MAX_REQUEST, create_app
from stockbrief.corpus import CALIBRATION, TEST, TRAIN
from stockbrief.evaluation import evaluate
from stockbrief.fixtures import ACQUISITION, CASE_STUDY, Document, SEED
from stockbrief.inputs import read_upload
from stockbrief.nlp import EventModel, chunks_for, stage_for
from stockbrief.store import Inbox


def pdf_bytes(lines):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for line in lines:
        page = writer.add_blank_page(width=400, height=200)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 8 120 Td ({line}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class NLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.model = EventModel()

    def test_each_source_backed_case_extracts_expected_category_and_stage(self):
        for document, label, stage in CASE_STUDY:
            with self.subTest(title=document.title):
                result = self.model.analyze(document)
                self.assertEqual((result["category"], result["stage"]), (label, stage))
                for quote in result["evidence"]:
                    self.assertEqual(document.pages[quote["page"] - 1][quote["start"]:quote["end"]], quote["text"])
                for figure in result["figures"]:
                    value = figure["value"]
                    page = document.pages[figure["page"] - 1]
                    self.assertEqual(page[figure["start"]:figure["start"] + len(value)], value)

    def test_abbreviations_and_decimal_amounts_keep_offsets(self):
        pages = ("  Example Ltd. declared a dividend of INR 8.50 per share. It fixed a record date. ",)
        chunks = chunks_for(pages)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Example Ltd. declared a dividend of INR 8.50 per share.")
        for chunk in chunks:
            self.assertEqual(pages[0][chunk.start:chunk.end], chunk.text)

    def test_deal_status_does_not_treat_future_or_denied_completion_as_closed(self):
        for text, expected in (
            ("The company did not complete the acquisition this quarter.", "Denied"),
            ("The company expects approval upon completion of the acquisition.", "Unclear"),
            ("The parties completed due diligence for the acquisition.", "Unclear"),
            ("The parties completed the acquisition after receiving clearance.", "Completed"),
        ):
            self.assertEqual(stage_for("deal", chunks_for((text,)))[0], expected)

    def test_authored_splits_are_disjoint_and_benchmark_reports_review(self):
        def samples(split):
            return {" ".join(s.lower().split()) for group in split.values() for s in group}

        self.assertFalse(samples(TRAIN) & samples(CALIBRATION))
        self.assertFalse(samples(TRAIN) & samples(TEST))
        self.assertFalse(samples(CALIBRATION) & samples(TEST))
        report = evaluate(self.model)
        baseline, learned = report["reports"]
        self.assertEqual(report["dataset"]["test"], 32)
        self.assertGreater(learned["macro_f1"], baseline["macro_f1"])
        self.assertEqual(sum(map(sum, learned["matrix"])), 32)
        self.assertIn("review_recall", learned)

    def test_ambiguous_announcement_can_abstain(self):
        document = Document("OTHER", ("The board took note of the matter; a further update will follow.",), "Update")
        self.assertEqual(self.model.analyze(document)["category"], "review")

    def test_pdf_page_numbers_survive_extraction(self):
        payload = pdf_bytes(("Page one outlines the meeting schedule.", "The company imposed an operating shutdown."))
        pages, origin = read_upload("sample.pdf", payload)
        self.assertEqual(origin, "pdf")
        self.assertEqual(len(pages), 2)
        self.assertIn("shutdown", pages[1])


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.model = EventModel()

    def test_offline_queue_survives_restart_and_drains_exactly_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inbox.sqlite"
            item = Inbox(path, self.model)
            item.link(False)
            item.step()
            item.step()
            local = Document("OTHER", ("The directors declared a dividend per share.",), "Local document", origin="text")
            _, was_duplicate = item.record(local)
            self.assertFalse(was_duplicate)
            recovered = Inbox(path, self.model)
            self.assertEqual(recovered.state()["pending"], 2)
            self.assertEqual(len(recovered.state()["disclosures"]), len(SEED) + 1)
            self.assertEqual(recovered.link(True)["duplicates"], 1)
            state = recovered.state()
            self.assertEqual(state["pending"], 0)
            self.assertEqual(len(state["disclosures"]), len(SEED) + 2)
            self.assertEqual(recovered.link(True)["duplicates"], 1)
            recovered.reset()
            self.assertIn("Local document", {row["title"] for row in recovered.state()["disclosures"]})

    def test_concurrent_exact_retries_share_one_id(self):
        with tempfile.TemporaryDirectory() as folder:
            item = Inbox(Path(folder) / "inbox.sqlite", self.model)
            text = "The management reported higher revenue and operating margins for the quarter."
            record = Document("OTHER", (text,), "Same local result")
            with ThreadPoolExecutor(max_workers=4) as workers:
                results = list(workers.map(item.record, [record] * 4))
            self.assertEqual(len({row["id"] for row, _ in results}), 1)
            self.assertEqual(sum(duplicate for _, duplicate in results), 3)

    def test_duplicates_are_keyed_per_disclosure_date_when_source_is_known(self):
        with tempfile.TemporaryDirectory() as folder:
            item = Inbox(Path(folder) / "inbox.sqlite", self.model)
            repeated = Document(ACQUISITION.ticker, ACQUISITION.pages, ACQUISITION.title, ACQUISITION.source, "2024-08-18")
            self.assertFalse(item.record(repeated)[1])


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "inbox.sqlite"
        self.time_limits = threadpool_limits(limits=1)
        self.addCleanup(self.time_limits.restore_original_limits)
        app = await asyncio.to_thread(create_app, self.path)
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
        self.addAsyncCleanup(self.client.aclose)

    async def test_home_static_health_inbox_and_display(self):
        for url in ("/", "/static/workspace.js", "/api/health", "/api/inbox", "/api/evaluation"):
            with self.subTest(url=url):
                response = await self.client.get(url)
                self.assertEqual(response.status_code, 200)
        response = await self.client.get("/api/display.png")
        self.assertEqual(response.status_code, 200)
        with Image.open(BytesIO(response.content)) as png:
            self.assertEqual((png.mode, png.size, len(png.tobytes())), ("1", (128, 64), 1024))

    async def test_text_and_pdf_import_and_replay(self):
        disconnected = await self.client.post("/api/replay/link", json={"connected": False})
        self.assertFalse(disconnected.json()["connected"])
        for i in (1, 2):
            tick = await self.client.post("/api/replay/step")
            self.assertEqual(tick.json()["pending"], i)
        uploaded = await self.client.post(
            "/api/analyze", data={"ticker": "OTHER"},
            files={"file": ("sample.pdf", pdf_bytes(("The board appointed a new director.",)), "application/pdf")},
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        self.assertEqual(uploaded.json()["entry"]["brief"]["category"], "governance")
        local = await self.client.post("/api/analyze", data={
            "ticker": "OTHER", "title": "Local text",
            "text": "The company declared a special dividend of INR 4 per share to eligible investors.",
        })
        self.assertEqual(local.status_code, 200, local.text)
        self.assertEqual(local.json()["entry"]["brief"]["category"], "corporate_action")
        rebuilt = await self.client.post("/api/replay/link", json={"connected": True})
        self.assertEqual(rebuilt.json()["duplicates"], 1)
        self.assertEqual(rebuilt.json()["pending"], 0)
        record = await self.client.get("/api/disclosures/" + str(uploaded.json()["entry"]["id"]))
        self.assertRegex(record.json()["pages"][0], "appointed a new director")

    async def test_intelligence_uses_saved_source_and_handles_missing_entries(self):
        uploaded = await self.client.post("/api/analyze", data={
            "ticker": "OTHER", "text": "The company declared a special dividend of INR 4 per share.",
        })
        self.assertEqual(uploaded.status_code, 200)
        entry_id = uploaded.json()["entry"]["id"]
        result = await self.client.get(f"/api/disclosures/{entry_id}/intelligence")
        self.assertEqual(result.status_code, 200)
        event = result.json()["events"][0]
        self.assertEqual(event["category"], "corporate_action")
        raw = (await self.client.get(f"/api/disclosures/{entry_id}")).json()["pages"]
        for quote in event["evidence"]:
            self.assertEqual(raw[quote["page"] - 1][quote["start"]:quote["end"]], quote["text"])
        self.assertEqual((await self.client.get("/api/disclosures/999/intelligence")).status_code, 404)

    async def test_live_refresh_routes_to_local_feed(self):
        from unittest.mock import patch
        with patch("stockbrief.live_feed.LiveFeed.refresh", return_value={"state": "available", "imported": 0}):
            result = await self.client.post("/api/live/refresh")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["state"], "available")
        denied = await self.client.post("/api/live/refresh", headers={"Origin": "https://example.org"})
        self.assertEqual(denied.status_code, 403)

    async def test_feed_to_bge_intelligence_and_matching_market_context(self):
        from datetime import datetime
        from unittest.mock import patch
        from tests.test_live_feed import feed

        raw = feed(description="Infosys reported fiscal first-quarter results. It reported revenue of USD 4,714 million and 3.6% sequential growth in constant currency.",
                   published="Mon, 15 Jul 2024 18:05:00 +0530")
        def provider(url, limit):
            from stockbrief.live_feed import RSS_URL
            self.assertEqual(url, RSS_URL)
            return raw
        with patch.dict("os.environ", {"STOCKBRIEF_NSE_ATTACHMENTS": "0", "STOCKBRIEF_DISABLE_MARKET": "0"}):
            app = await asyncio.to_thread(create_app, self.path, live_fetch=provider)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            result = await client.post("/api/live/refresh")
            self.assertEqual(result.json()["imported"], 1, result.text)
            inbox = (await client.get("/api/inbox")).json()
            imported = next(x for x in inbox["disclosures"] if x["origin"] == "nse-rss")
            brief = (await client.get(f"/api/disclosures/{imported['id']}/intelligence")).json()
            self.assertEqual(brief["source_scope"], "rss_excerpt")
            self.assertEqual(brief["events"][0]["category"], "earnings")
            self.assertTrue(brief["events"][0]["evidence"])
            self.assertTrue(any("RSS excerpt" in note for note in brief["limitations"]))
            self.assertEqual(imported["date"], "2024-07-15")
            before = datetime.fromisoformat("2024-07-12T09:15:00+05:30")
            after = datetime.fromisoformat("2024-07-16T09:15:00+05:30")
            def candles(path):
                self.assertTrue(path.endswith("/days/1/2024-07-22/2024-07-08"))
                return [(before, 100), (after, 102 if path.startswith("NSE_INDEX") else 105)]
            with patch("stockbrief.market.MarketData._get", side_effect=candles):
                context = (await client.get("/api/market-context/INFY?filing_date=" + imported["date"])).json()
            self.assertEqual(context["relative_return_pp"], 3)

    async def test_offline_market_panel_has_a_recorded_chart(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"STOCKBRIEF_DISABLE_MARKET": "1"}):
            app = await asyncio.to_thread(create_app, self.path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            disabled = (await client.get("/api/market/NIFTY50")).json()
            self.assertEqual((disabled["state"], disabled["candles"]), ("disabled", []))
            example = (await client.get("/api/market-example")).json()
            self.assertEqual((example["state"], example["symbol"], example["filing_date"]), ("recorded", "INFY", "2024-07-18"))
            self.assertEqual(len(example["candles"]), 6)
            self.assertAlmostEqual(example["comparisons"][0]["difference_pp"], 4.21, places=2)
            self.assertAlmostEqual(example["comparisons"][4]["difference_pp"], 6.56, places=2)
            self.assertEqual(len(example["source_links"]), 3)

    async def test_cooperative_simulation_is_bounded_and_does_not_modify_inbox(self):
        before = (await self.client.get("/api/inbox")).json()
        response = await self.client.post("/api/edge/simulate", json={"count":24,"scenario":"worker_outage"})
        self.assertEqual(response.status_code, 200)
        run = response.json()["runs"]["cooperative"]
        self.assertEqual(run["completed"], 24)
        self.assertGreater(run["lease_retries"], 0)
        self.assertEqual((await self.client.get("/api/inbox")).json(), before)
        self.assertEqual((await self.client.post("/api/edge/simulate", json={"count":61})).status_code, 422)

    async def test_edge_lab_actual_model_and_local_config(self):
        config = await self.client.post("/api/edge/config", json={"watchlist":["INFY"],"policy":"adaptive"})
        self.assertEqual(config.status_code, 200)
        self.assertEqual(config.json()["watchlist"], ["INFY"])
        self.assertEqual((await self.client.post("/api/edge/config", json={"watchlist":["UNKNOWN"],"policy":"fifo"})).status_code, 422)
        self.assertEqual((await self.client.post("/api/edge/drain")).json()["pending"], 0)
        report = (await self.client.post("/api/edge/benchmark")).json()
        self.assertTrue(report["same_categories"])
        for policy in ("fifo", "adaptive"):
            self.assertEqual(len(report["runs"][policy]), 2)
            self.assertEqual(len(report["runs"][policy][0]["jobs"]), 6)
        self.assertEqual((await self.client.post("/api/edge/config", headers={"Origin":"https://example.org"},
                         json={"watchlist":[],"policy":"fifo"})).status_code, 403)

    async def test_bad_and_cross_origin_requests_are_rejected(self):
        responses = [
            (await self.client.post("/api/analyze", data={"ticker": "NONE", "text": "Something happened at a board meeting."}), 422),
            (await self.client.post("/api/analyze", data={"ticker": "INFY"}, files={"file": ("x.pdf", b"bad", "application/pdf")}), 422),
            (await self.client.post("/api/analyze", data={"ticker": "INFY"}, files={"file": ("x.pdf", pdf_bytes(("",)), "application/pdf")}), 422),
            (await self.client.post("/api/replay/link", json={"connected": "false"}), 422),
            (await self.client.post("/api/replay/reset", headers={"Origin": "https://example.org"}), 403),
            (await self.client.post("/api/replay/reset", content=b"x" * (MAX_REQUEST + 1)), 413),
            (await self.client.get("/api/disclosures/999/second-opinion"), 404),
        ]
        for response, expected in responses:
            self.assertEqual(response.status_code, expected, response.text)

    async def test_optional_comparator_runs_on_the_saved_source(self):
        class Comparator:
            name = "Fixture transformer"

            def classify(self, document):
                return "governance" if document.pages else "review"

        app = await asyncio.to_thread(create_app, self.path, Comparator())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            inbox = (await client.get("/api/inbox")).json()
            self.assertEqual(inbox["transformer"], "Fixture transformer")
            entry = inbox["disclosures"][0]["id"]
            result = (await client.get(f"/api/disclosures/{entry}/second-opinion")).json()
            self.assertEqual(result["category"], "governance")
