from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from stockbrief.live_feed import LiveFeed, RSS_URL, attachment_url, parse_rss
from stockbrief.store import Inbox

PDF = "https://nsearchives.nseindia.com/corporate/INFY_test.pdf"


def feed(title="Infosys Limited", description="The company reported quarterly earnings.", link=PDF,
         published="Wed, 25 Nov 2026 18:05:00 +0530"):
    return (f'<rss version="2.0"><channel><item><title>{title}</title>'
            f'<description>{description}</description><link>{link}</link>'
            f'<pubDate>{published}</pubDate></item></channel></rss>').encode()


class RSSParserTests(unittest.TestCase):
    def test_real_shape_and_ist_date(self):
        rows, unmatched, rejected = parse_rss(feed())
        self.assertEqual((unmatched, rejected), (0, 0))
        self.assertEqual(rows[0].ticker, "INFY")
        self.assertEqual(rows[0].published.isoformat(), "2026-11-25T18:05:00+05:30")
        self.assertEqual(rows[0].url, PDF)

    def test_unknown_issuer_and_bad_timestamp_are_not_guessed(self):
        self.assertEqual(parse_rss(feed(title="Tata Motors Limited"))[1], 1)
        self.assertEqual(parse_rss(feed(published="unknown"))[2], 1)

    def test_xml_entities_and_arbitrary_urls_rejected(self):
        with self.assertRaises(ValueError):
            parse_rss(b'<!DOCTYPE rss [<!ENTITY x "x">]><rss/>')
        for url in ("http://nsearchives.nseindia.com/a.pdf", "https://nsearchives.nseindia.com.evil/a.pdf",
                    "https://127.0.0.1/a.pdf", "https://user@nsearchives.nseindia.com/a.pdf"):
            self.assertEqual(attachment_url(url), "")
        self.assertEqual(parse_rss(feed(link="https://example.org/a.pdf"))[0][0].url, "")

    def test_correction_has_distinct_identity(self):
        one = parse_rss(feed())[0][0]
        two = parse_rss(feed(description="The company corrected its quarterly earnings."))[0][0]
        self.assertNotEqual(one.key, two.key)


class FakeModel:
    def analyze(self, document):
        return {"category": "earnings", "evidence": [], "model": "fixture"}


class LiveFeedTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "inbox.sqlite"
        self.inbox = Inbox(self.path, FakeModel())

    def test_rss_import_deduplication_after_restart_and_reset(self):
        provider = Mock(return_value=feed())
        result = LiveFeed(self.inbox, fetch=provider).refresh()
        self.assertEqual((result["imported"], result["rss_only"]), (1, 1))
        entry = self.inbox.state()["disclosures"][0]
        self.assertEqual(entry["origin"], "nse-rss")
        self.assertEqual(entry["source_scope"], "rss_excerpt")
        self.assertEqual(entry["published_at"], "2026-11-25T18:05:00+05:30")
        self.inbox.reset()
        reopened = Inbox(self.path, FakeModel())
        result = LiveFeed(reopened, fetch=provider).refresh()
        self.assertEqual((result["imported"], result["duplicates"]), (0, 1))
        self.assertEqual(sum(x["origin"] == "nse-rss" for x in reopened.state()["disclosures"]), 1)
        self.assertTrue(all(call.args[0] == RSS_URL for call in provider.call_args_list))

    def test_attachment_is_opt_in_and_failure_falls_back(self):
        provider = Mock(side_effect=[feed(), httpx.ConnectError("offline")])
        result = LiveFeed(self.inbox, attachments=True, fetch=provider).refresh()
        self.assertEqual((result["imported"], result["rss_only"], result["errors"]), (1, 1, 1))
        self.assertEqual(provider.call_args_list[1].args[0], PDF)

    def test_successful_attachment_keeps_pages_and_provenance(self):
        provider = Mock(side_effect=[feed(), b"%PDF-fixture"])
        with patch("stockbrief.live_feed.read_upload", return_value=(("Quarterly results were reported.",), "pdf")):
            result = LiveFeed(self.inbox, attachments=True, fetch=provider).refresh()
        self.assertEqual(result["pdf"], 1)
        entry = self.inbox.state()["disclosures"][0]
        self.assertEqual(entry["source_scope"], "pdf_text")
        self.assertEqual(self.inbox.document(entry["id"])["pages"], ["Quarterly results were reported."])

    def test_network_error_and_cooldown_do_not_create_fake_news(self):
        provider = Mock(side_effect=httpx.ConnectError("offline"))
        live = LiveFeed(self.inbox, fetch=provider)
        result = live.refresh()
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["imported"], 0)
        self.assertTrue(live.refresh()["throttled"])
        provider.assert_called_once()
        self.assertFalse(any(x["origin"].startswith("nse-") for x in self.inbox.state()["disclosures"]))
