"""Bounded local reader for NSE's published announcements RSS subscription."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html.parser import HTMLParser
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

import httpx

from .inputs import MAX_TEXT, MAX_UPLOAD, make_document, read_upload

RSS_URL = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"
IST = timezone(timedelta(hours=5, minutes=30))
ARCHIVE_HOSTS = {"archives.nseindia.com", "nsearchives.nseindia.com", "nsearchives.nseindia.in"}
POLL_SECONDS = 120
MAX_FEED = 5 * 1024 * 1024
BATCH = 5
WINDOW = 50
ISSUERS = {
    "infosys limited": "INFY", "hdfc bank limited": "HDFCBANK",
    "tata motors passenger vehicles limited": "TMPV",
    "reliance industries limited": "RELIANCE",
    "tata consultancy services limited": "TCS",
}


class Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    parser = Text()
    parser.feed(value or "")
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def attachment_url(value):
    try:
        url = urlsplit(value)
        if (url.scheme == "https" and url.hostname in ARCHIVE_HOSTS
                and url.netloc == url.hostname and not url.query and not url.fragment
                and url.path.lower().endswith(".pdf") and not any(c.isspace() for c in value)):
            return value
    except ValueError:
        pass
    return ""


def timestamp(value):
    try:
        result = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        result = None
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                result = datetime.strptime(value, fmt)
                break
            except ValueError:
                pass
    if result is None:
        raise ValueError("Invalid announcement time.")
    return (result if result.tzinfo else result.replace(tzinfo=IST)).astimezone(IST)


@dataclass(frozen=True)
class Announcement:
    key: str
    ticker: str
    issuer: str
    description: str
    published: datetime
    url: str


def parse_rss(payload):
    if len(payload) > MAX_FEED or b"<!doctype" in payload.lower() or b"<!entity" in payload.lower():
        raise ValueError("Unsupported RSS payload.")
    # Decode before parsing to reject alternate XML encodings with hidden declarations.
    raw = payload.decode("utf-8-sig")
    if "<!doctype" in raw.lower() or "<!entity" in raw.lower() or "\x00" in raw:
        raise ValueError("Unsupported RSS encoding.")
    root = ET.fromstring(raw)
    channel = root.find("channel")
    if root.tag != "rss" or channel is None:
        raise ValueError("Expected an RSS channel.")
    found = {}
    unmatched = rejected = 0
    for row in channel.findall("item"):
        issuer = plain(row.findtext("title"))
        ticker = ISSUERS.get(issuer.casefold())
        if not ticker:
            unmatched += 1
            continue
        description = plain(row.findtext("description"))
        url = attachment_url((row.findtext("link") or "").strip())
        try:
            published = timestamp((row.findtext("pubDate") or "").strip())
            if not 12 <= len(description) <= 90_000:
                raise ValueError("Missing usable RSS text.")
        except ValueError:
            rejected += 1
            continue
        # Content-sensitive identity preserves corrections instead of dropping them.
        key = sha256("\n".join((ticker, published.isoformat(), url, description)).encode()).hexdigest()
        found[key] = Announcement(key, ticker, issuer, description, published, url)
    return sorted(found.values(), key=lambda row: row.published, reverse=True), unmatched, rejected


def fetch_bytes(url, limit):
    # Only configured RSS and exact NSE archive PDF hosts. No redirect following,
    # cookies, proxy environment, credentials, or user-controlled arbitrary URLs.
    if url != RSS_URL and not attachment_url(url):
        raise ValueError("Unsupported source URL.")
    with httpx.Client(timeout=8, follow_redirects=False, trust_env=False) as client:
        with client.stream("GET", url, headers={"User-Agent": "StockBriefEdge/1.0 (RSS reader)"}) as response:
            response.raise_for_status()
            content = bytearray()
            for block in response.iter_bytes(65536):
                if len(content) + len(block) > limit:
                    raise ValueError("Source exceeds size limit.")
                content.extend(block)
            return bytes(content)


class LiveFeed:
    def __init__(self, inbox, *, attachments=False, fetch=fetch_bytes):
        self.inbox = inbox
        self.attachments = attachments
        self.fetch = fetch
        self.lock = Lock()
        self.next_check = 0
        self.last = None
        self.last_success = None

    def refresh(self):
        if not self.lock.acquire(blocking=False):
            return {"state": "busy", "retry_seconds": POLL_SECONDS}
        try:
            if self.last is not None and monotonic() < self.next_check:
                return {**self.last, "throttled": True, "imported": 0}
            result = {"state": "available", "checked_at": datetime.now(timezone.utc).isoformat(),
                      "last_success": self.last_success, "imported": 0, "duplicates": 0,
                      "pending": 0, "unmatched": 0, "rejected": 0, "rss_only": 0, "pdf": 0,
                      "errors": 0, "retry_seconds": POLL_SECONDS, "attachment_access": self.attachments}
            try:
                rows, result["unmatched"], result["rejected"] = parse_rss(self.fetch(RSS_URL, MAX_FEED))
            except (httpx.HTTPError, ValueError, ET.ParseError):
                result["state"] = "unavailable"
            else:
                self.last_success = result["checked_at"]
                result["last_success"] = self.last_success
                result["latest_supported_at"] = rows[0].published.isoformat() if rows else None
                result["outside_window"] = max(0, len(rows) - WINDOW)
                unseen = []
                for row in rows[:WINDOW]:
                    if self.inbox.live_seen(row.key):
                        result["duplicates"] += 1
                    else:
                        unseen.append(row)
                result["pending"] = max(0, len(unseen) - BATCH)
                for row in reversed(unseen[:BATCH]):
                    pages, scope = (row.description,), "rss_excerpt"
                    if self.attachments and row.url:
                        try:
                            extracted, _ = read_upload("filing.pdf", self.fetch(row.url, MAX_UPLOAD))
                            if not 12 <= sum(len(page.strip()) for page in extracted) <= MAX_TEXT:
                                raise ValueError("Extracted PDF text exceeds analysis bounds.")
                            pages, scope = extracted, "pdf_text"
                        except (httpx.HTTPError, ValueError):
                            result["errors"] += 1
                    try:
                        document = make_document(
                            row.ticker, row.issuer + " — NSE announcement", pages,
                            "NSE corporate announcements", row.published.date().isoformat(),
                            row.url or RSS_URL, "nse-pdf" if scope == "pdf_text" else "nse-rss",
                        )
                        document = replace(document, published_at=row.published.isoformat(),
                                           retrieved_at=datetime.now(timezone.utc).isoformat(), source_scope=scope)
                        _, duplicate = self.inbox.record_live(row.key, document)
                    except ValueError:
                        result["errors"] += 1
                        continue
                    result["duplicates"] += int(duplicate)
                    result["imported"] += int(not duplicate)
                    if not duplicate:
                        result["pdf" if scope == "pdf_text" else "rss_only"] += 1
            self.last = result
            self.next_check = monotonic() + POLL_SECONDS
            return result
        finally:
            self.lock.release()
