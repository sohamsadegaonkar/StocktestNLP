from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from stockbrief.edge import EdgeQueue, ordered, revision_delta, benchmark
from stockbrief.fixtures import Document
from stockbrief.live_feed import Announcement, LiveFeed
from stockbrief.store import Inbox

BUDGET = {"max_jobs": 1, "cpu_percent": 90, "available_memory_mib": 1024}
STAMP = datetime(2026, 11, 25, tzinfo=timezone.utc)
LINK = "https://nsearchives.nseindia.com/corporate/example.pdf"


class Model:
    name = "fixture"
    def analyze(self, document):
        return {"category": "earnings", "evidence": [], "model": self.name}


def announcement(key, ticker="INFY", text="Reported revenue of INR 20 crore.", link=LINK):
    return Announcement(key, ticker, ticker, text, STAMP, link)


class EdgeTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "db.sqlite"
        self.inbox = Inbox(self.path, Model())
        self.live = LiveFeed(self.inbox, fetch=lambda *args: self.fail("Offline processing used network"))
        self.queue = self.live.edge

    def test_restart_offline_priority_and_atomic_receipt(self):
        self.queue.configure(["INFY"], "adaptive")
        self.queue.enqueue([announcement("important"), announcement("routine", "TCS", "Annual general meeting notice.")])
        self.assertEqual(self.queue.status()["pending"], 2)
        restarted = LiveFeed(Inbox(self.path, Model()), fetch=lambda *args: self.fail("Network used"))
        with patch("stockbrief.edge.resource_budget", return_value=BUDGET):
            result = restarted.drain_offline()
        self.assertEqual((result["imported"], result["pending"]), (1, 1))
        self.assertEqual(restarted.edge.status()["recent"][0]["ticker"], "INFY")
        self.assertEqual(restarted.edge.enqueue([announcement("important")])["duplicates"], 1)
        self.assertTrue(restarted.inbox.live_seen("important"))
        self.assertEqual(restarted.edge.status()["watchlist"], ["INFY"])

    def test_fifo_and_aging(self):
        rows = [{"feed_key":"a","ticker":"TCS","description":"Routine notice","received":0},
                {"feed_key":"b","ticker":"INFY","description":"Earnings correction","received":1}]
        self.assertEqual(ordered(rows, ["INFY"], "fifo", 1)[0]["feed_key"], "a")
        self.assertEqual(ordered(rows, ["INFY"], "adaptive", 1)[0]["feed_key"], "b")
        rows[1]["received"] = 1000
        self.assertEqual(ordered(rows, ["INFY"], "adaptive", 1000)[0]["feed_key"], "a")

    def test_same_link_revision_compares_numbers_and_different_links_do_not(self):
        self.queue.enqueue([announcement("one")])
        self.queue.drain(self.live.prepare, BUDGET)
        self.queue.enqueue([announcement("two", text="Corrected revenue of INR 25 crore.")])
        self.queue.drain(self.live.prepare, BUDGET)
        delta = self.queue.status()["recent"][0]["delta"]
        self.assertEqual(delta["state"], "linked_excerpt_changed")
        self.assertEqual(delta["removed_figures"], ["INR 20 crore"])
        self.assertEqual(delta["added_figures"], ["INR 25 crore"])
        self.queue.enqueue([announcement("three", text="Reported revenue of INR 30 crore.", link=LINK.replace("example", "other"))])
        self.queue.drain(self.live.prepare, BUDGET)
        self.assertEqual(self.queue.status()["recent"][0]["delta"]["state"], "no_prior_linked_excerpt")

    def test_capacity_and_failed_jobs_retained(self):
        with patch("stockbrief.edge.CAPACITY", 1):
            result = self.queue.enqueue([announcement("one"), announcement("two")])
        self.assertEqual((result["queued"],result["capacity_deferred"]), (1,1))
        def bad(_):
            raise ValueError("invalid")
        for _ in range(3):
            self.queue.drain(bad, BUDGET)
        self.assertEqual((self.queue.status()["pending"],self.queue.status()["failed"]), (0,1))

    def test_benchmark_measures_identical_workload(self):
        report = benchmark(Model())
        self.assertTrue(report["same_categories"])
        for policy in ("fifo","adaptive"):
            self.assertEqual(len(report["runs"][policy]), 2)
            for run in report["runs"][policy]:
                self.assertEqual(len(run["jobs"]), 6)
                self.assertGreaterEqual(run["total_ms"], run["first_relevant_ms"])
        self.assertTrue(report["runs"]["adaptive"][0]["jobs"][0]["relevant"])
        self.assertFalse(report["runs"]["fifo"][0]["jobs"][0]["relevant"])
