from pathlib import Path
import tempfile
import unittest

from threadpoolctl import threadpool_limits

from stockbrief.evaluation import evaluate
from stockbrief.fixtures import Document
from stockbrief.nlp import EventModel
from stockbrief.semantic import load_semantic, model_directory
from stockbrief.store import Inbox

TATA = """Tata Motors Passenger Vehicles (TMPV) has announced a price hike of up to 0.5% across its ICE and EV lineup starting December 1, 2026, while reporting a Q2 FY27 net loss driven by restructuring costs at Jaguar Land Rover (JLR).

December 2026 Price Hike & Offers

• Price revision: TMPV will raise prices by an average of 0.5% from December 1, 2026, citing rising input costs.
• Estimated impact: Buyers may see increases ranging from roughly ₹2,384 to ₹14,645 depending on the model and variant.
• Pre-hike discounts: Year-end offers and exchange/scrappage benefits in late November reach up to ₹1.25 lakh on select Curvv variants, ₹1.15 lakh on Sierra diesel, and ₹75,000 on select Nexon petrol manual trimsls."""


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.model = load_semantic(model_directory())

    def test_mixed_disclosure_has_two_source_backed_categories(self):
        brief = self.model.analyze(Document("OTHER", (TATA,), "Tata Motors update"))
        events = {event["category"]: event for event in brief["events"]}
        self.assertEqual(set(events), {"pricing", "earnings"})
        self.assertEqual(events["pricing"]["stage"], "Announced")
        self.assertEqual(events["earnings"]["stage"], "Reported")
        self.assertEqual(events["earnings"]["figures"], [])
        self.assertEqual([item["value"] for item in events["earnings"]["dates"]], ["Q2 FY27"])
        values = {item["value"] for item in events["pricing"]["figures"]}
        self.assertTrue({"up to 0.5%", "average of 0.5%", "up to ₹1.25 lakh"}.issubset(values))
        for event in events.values():
            for quote in event["evidence"]:
                self.assertEqual(TATA[quote["start"]:quote["end"]], quote["text"])
            for fact in event["dates"] + event["figures"]:
                self.assertTrue(TATA[fact["start"]:].startswith(fact["value"]))

    def test_semantic_beats_tfidf_on_the_original_holdout(self):
        with threadpool_limits(limits=1):
            result = evaluate(self.model)
        _, tfidf, semantic = result["reports"]
        self.assertGreater(semantic["macro_f1"], tfidf["macro_f1"])
        self.assertEqual(semantic["artifact"]["execution_providers"], ["CPUExecutionProvider"])
        self.assertEqual(sum(map(sum, semantic["matrix"])), result["dataset"]["test"])
        self.assertTrue(all(item["matched"] for item in result["historical_cases"]))

    def test_repeat_import_updates_a_legacy_brief_with_same_id(self):
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            path = Path(folder) / "inbox.sqlite"
            document = Document("OTHER", (TATA,), "Saved Tata disclosure")
            old, duplicate = Inbox(path, EventModel()).record(document)
            self.assertFalse(duplicate)
            upgraded = Inbox(path, self.model)
            refreshed, duplicate = upgraded.record(document)
            self.assertTrue(duplicate)
            self.assertEqual(refreshed["id"], old["id"])
            self.assertEqual(refreshed["title"], old["title"])
            self.assertEqual(len(refreshed["brief"]["events"]), 2)
            self.assertEqual(upgraded.document(old["id"])["brief"]["model_version"], self.model.version)
