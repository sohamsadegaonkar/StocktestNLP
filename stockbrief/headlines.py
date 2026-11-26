"""Optional publisher headlines when the official exchange feed is quiet."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

from .corpus import TICKERS

GOOGLE_NEWS = "https://news.google.com/rss/search"
MAX_BYTES = 600_000
REFRESH_SECONDS = 120
HEADLINE_LIMIT = 6
ISSUER_PATTERNS = {
    "INFY": r"infosys|infy",
    "TMPV": r"tata motors (?:passenger vehicles|pv)|tmpv",
    "RELIANCE": r"reliance industries|reliance (?:shares|stock)|ril",
    "TCS": r"tata consultancy services|tcs",
    "HDFCBANK": r"hdfc\s*bank",
}
ISSUERS = {symbol: re.compile(rf"\b(?:{pattern})\b", re.I) for symbol, pattern in ISSUER_PATTERNS.items()}
MARKET_TOPIC = re.compile(
    r"\b(?:shares?|stocks?|earnings|results?|dividends?|guidance|contracts?|acqui\w+|merger|demerger|"
    r"deals?|appoint\w+|buybacks?|sales|deliveries|price (?:hikes?|increase|cuts?)|"
    r"partnerships?|launch\w+|orders?|margins?|penalt\w+|profit|revenue)\b", re.I,
)


def parse_headlines(feed: bytes, symbol: str, now: datetime) -> list[dict]:
    if len(feed) > MAX_BYTES:
        raise ValueError("Headline feed is too large.")
    text = feed.decode("utf-8-sig")
    if "\x00" in text or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.I):
        raise ValueError("Unsupported headline feed.")
    root = ElementTree.fromstring(text)
    channel = root.find("channel")
    if root.tag != "rss" or channel is None:
        raise ValueError("Expected an RSS channel.")
    unique = {}
    for item in channel.findall("item"):
        publisher = " ".join((item.findtext("source") or "").split())[:100]
        title = " ".join((item.findtext("title") or "").split())
        if publisher and title.endswith(" - " + publisher):
            title = title[:-(3 + len(publisher))]
        url = (item.findtext("link") or "").strip()
        try:
            stamp = parsedate_to_datetime(item.findtext("pubDate") or "")
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.netloc != "news.google.com" or
                    not parsed.path.startswith("/rss/articles/") or not 0 < len(url) <= 2048 or
                    any(char.isspace() for char in url) or stamp.tzinfo is None):
                continue
        except (ValueError, TypeError, OverflowError):
            continue
        if not (publisher and now - timedelta(days=7) <= stamp <= now + timedelta(minutes=10)
                and ISSUERS[symbol].search(title) and MARKET_TOPIC.search(title)):
            continue
        record = {"title": title[:360], "publisher": publisher, "url": url,
                  "published_at": stamp.astimezone(timezone.utc).isoformat()}
        key = title.casefold()
        if key not in unique or record["published_at"] > unique[key]["published_at"]:
            unique[key] = record
    return sorted(unique.values(), key=lambda record: record["published_at"], reverse=True)[:HEADLINE_LIMIT]


def _fetch(symbol: str) -> bytes:
    aliases = '"Tata Motors PV" OR "Tata Motors Passenger Vehicles"' if symbol == "TMPV" else f'"{TICKERS[symbol]}"'
    params = {"q": f"({aliases} OR {symbol}) when:7d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    with httpx.stream("GET", GOOGLE_NEWS, params=params, timeout=6, follow_redirects=False) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes(16384):
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("Headline feed is too large.")
        return bytes(data)


class Headlines:
    def __init__(self):
        self._cache = {}
        self._locks = {symbol: Lock() for symbol in ISSUERS}

    def latest(self, symbol: str) -> dict:
        with self._locks[symbol]:
            previous = self._cache.get(symbol)
            if previous and monotonic() < previous[0]:
                return previous[1]
            now = datetime.now(timezone.utc)
            result = {"state": "unavailable", "source": "Google News", "checked_at": None, "items": []}
            try:
                items = parse_headlines(_fetch(symbol), symbol, now)
                result.update(state="available" if items else "empty", checked_at=now.isoformat(), items=items)
            except (httpx.HTTPError, ValueError, ElementTree.ParseError, ImportError):
                if previous:
                    cached = previous[1]
                    recent = [item for item in cached["items"]
                              if datetime.fromisoformat(item["published_at"]) >= now - timedelta(days=7)]
                    if recent:
                        result = {**cached, "state": "cached", "items": recent}
            self._cache[symbol] = (monotonic() + REFRESH_SECONDS, result)
            return result
