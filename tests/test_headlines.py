from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import patch

import httpx

from stockbrief.headlines import Headlines, parse_headlines

NOW = datetime(2026, 11, 26, tzinfo=timezone.utc)


def row(title, publisher="Example News", link="https://news.google.com/rss/articles/x", date="Wed, 25 Nov 2026 09:00:00 GMT"):
    return f"<item><title>{title}</title><source>{publisher}</source><link>{link}</link><pubDate>{date}</pubDate></item>"


def payload(*rows):
    return ("<rss><channel>" + "".join(rows) + "</channel></rss>").encode()


class HeadlinesTests(TestCase):
    def test_only_recent_watchlist_publisher_heads_are_returned(self):
        title = "Tata Motors PV announces price hike"
        data = payload(
            row(title + " - Example News"), row(title), row("Infosys reports earnings"),
            row("Tata Motors commercial vehicles earnings"), row("Tata Motors PV picnic"),
            row("TMPV first quarter earnings", date="Sun, 01 Nov 2026 09:00:00 GMT"),
            row("TMPV stock", link="https://unknown.test/rss/articles/x"),
        )
        result = parse_headlines(data, "TMPV", NOW)
        self.assertEqual([item["title"] for item in result], [title])
        self.assertEqual(result[0]["publisher"], "Example News")
        self.assertEqual(result[0]["published_at"], "2026-11-25T09:00:00+00:00")

    def test_outage_labels_recent_cached_headlines(self):
        data = payload(row("Tata Motors PV announces price hike"))
        with patch("stockbrief.headlines.datetime") as clock, patch("stockbrief.headlines._fetch") as fetch, patch(
            "stockbrief.headlines.monotonic", side_effect=[0, 121, 122],
        ):
            clock.now.return_value = NOW
            clock.fromisoformat.side_effect = datetime.fromisoformat
            fetch.side_effect = [data, httpx.ConnectError("offline")]
            feed = Headlines()
            fresh = feed.latest("TMPV")
            cached = feed.latest("TMPV")
        self.assertEqual(fresh["state"], "available")
        self.assertEqual(cached, {**fresh, "state": "cached"})
