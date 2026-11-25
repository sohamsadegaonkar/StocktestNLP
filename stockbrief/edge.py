"""Local durable financial event scheduling and bounded revision comparison."""
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import re
from time import perf_counter, time

import psutil

from .fixtures import Document
from .nlp import FIGURES

SYMBOLS = ("INFY", "HDFCBANK", "TMPV", "RELIANCE", "TCS")
CAPACITY = 200
HISTORY = 200
CORRECTION = re.compile(r"\b(?:corrigendum|rectification|correction|corrected|revised|withdrawn|clarification)\b", re.I)
URGENT = re.compile(r"\b(?:acquisition|demerger|penalty|default|resignation|shutdown|earnings|financial results|price hike)\b", re.I)
AGE_SECONDS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_queue (
    feed_key TEXT PRIMARY KEY, ticker TEXT NOT NULL, issuer TEXT NOT NULL,
    description TEXT NOT NULL, published_at TEXT NOT NULL, link TEXT NOT NULL,
    received REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edge_history (
    id INTEGER PRIMARY KEY, feed_key TEXT NOT NULL UNIQUE, ticker TEXT NOT NULL,
    link TEXT NOT NULL, description TEXT NOT NULL, published_at TEXT NOT NULL,
    disclosure_id INTEGER NOT NULL, wait_ms REAL NOT NULL, processing_ms REAL NOT NULL,
    delta TEXT NOT NULL, relevant INTEGER NOT NULL, completed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS edge_config (
    id INTEGER PRIMARY KEY CHECK(id=1), watchlist TEXT NOT NULL, policy TEXT NOT NULL
);
INSERT OR IGNORE INTO edge_config VALUES (1, '[]', 'adaptive');
"""


def priority(row, watchlist, now):
    # Aging is unbounded, unlike the bounded relevance boost.
    age = max(0, now - row["received"]) / AGE_SECONDS
    return age + 4 * (row["ticker"] in watchlist) + 3 * bool(CORRECTION.search(row["description"])) + bool(URGENT.search(row["description"]))


def ordered(rows, watchlist, policy, now):
    return sorted(rows, key=lambda r: (-priority(r, watchlist, now) if policy == "adaptive" else 0,
                                      r["received"], r["feed_key"]))


def revision_delta(previous, current):
    """Exact-link candidate only; report textual changes, not economic direction."""
    if previous is None:
        return {"state": "no_prior_linked_excerpt"}
    old, new = previous["description"], current["description"]
    if old == new:
        return {"state": "unchanged_excerpt", "previous_entry_id": previous["disclosure_id"]}
    old_figures = Counter(FIGURES.findall(old))
    new_figures = Counter(FIGURES.findall(new))
    # Bounded word diff; numbers from the entire RSS excerpt are compared.
    a, b = old[:4000].split(), new[:4000].split()
    changes = []
    for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=True).get_opcodes():
        if op != "equal":
            changes.append({"before": " ".join(a[i:j])[:500], "after": " ".join(b[k:l])[:500]})
        if len(changes) >= 8:
            break
    return {"state": "linked_excerpt_changed", "previous_entry_id": previous["disclosure_id"],
            "previous_published_at": previous["published_at"],
            "removed_figures": list((old_figures - new_figures).elements())[:20],
            "added_figures": list((new_figures - old_figures).elements())[:20],
            "changes": changes, "truncated": len(old) > 4000 or len(new) > 4000 or len(changes) >= 8,
            "note": "Same issuer and attachment link; compared with last processed excerpt, not necessarily chronological revision. Verify the original filing."}


def resource_budget():
    cpu = psutil.cpu_percent(interval=0.05)
    free_mib = psutil.virtual_memory().available / (1024 * 1024)
    return {"cpu_percent": cpu, "available_memory_mib": round(free_mib),
            "max_jobs": 1 if cpu >= 80 or free_mib < 512 else 5}


class EdgeQueue:
    def __init__(self, inbox):
        self.inbox = inbox

    def _config(self, db):
        row = db.execute("SELECT watchlist, policy FROM edge_config WHERE id=1").fetchone()
        return json.loads(row["watchlist"]), row["policy"]

    def configure(self, watchlist, policy):
        if policy not in {"fifo", "adaptive"} or len(watchlist) > len(SYMBOLS) or any(s not in SYMBOLS for s in watchlist):
            raise ValueError("Select supported watchlist symbols and a scheduling policy.")
        watchlist = sorted(set(watchlist))
        with self.inbox.transaction(write=True) as db:
            db.execute("UPDATE edge_config SET watchlist=?, policy=? WHERE id=1", (json.dumps(watchlist), policy))
        return self.status()

    def enqueue(self, rows):
        counts = {"queued": 0, "duplicates": 0, "capacity_deferred": 0}
        with self.inbox.transaction(write=True) as db:
            size = db.execute("SELECT COUNT(*) FROM edge_queue").fetchone()[0]
            for row in reversed(rows):
                if db.execute("SELECT 1 FROM live_receipts WHERE feed_key=? UNION SELECT 1 FROM edge_queue WHERE feed_key=?",
                              (row.key, row.key)).fetchone():
                    counts["duplicates"] += 1
                    continue
                if size >= CAPACITY:
                    counts["capacity_deferred"] += 1
                    continue
                db.execute("INSERT INTO edge_queue VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                           (row.key, row.ticker, row.issuer, row.description, row.published.isoformat(), row.url, time()))
                size += 1
                counts["queued"] += 1
        return counts

    def status(self):
        with self.inbox.transaction() as db:
            watchlist, policy = self._config(db)
            rows = [dict(r) for r in db.execute("SELECT * FROM edge_queue")]
            history = [dict(r) for r in db.execute("SELECT id, ticker, disclosure_id, wait_ms, processing_ms, delta, relevant, completed_at FROM edge_history ORDER BY id DESC LIMIT 20")]
        now = time()
        pending = ordered([r for r in rows if r["attempts"] < 3], watchlist, policy, now)
        for r in history:
            r["delta"] = json.loads(r["delta"])
        return {"watchlist": watchlist, "policy": policy, "supported": list(SYMBOLS),
                "pending": len(pending), "failed": len(rows)-len(pending), "capacity": CAPACITY,
                "queue": [{"ticker": r["ticker"], "published_at": r["published_at"],
                           "priority": round(priority(r, watchlist, now), 2), "age_seconds": round(max(0, now-r["received"]), 1),
                           "attempts": r["attempts"]} for r in pending[:20]], "recent": history}

    def drain(self, prepare, budget=None):
        budget = budget or resource_budget()
        with self.inbox.transaction() as db:
            watchlist, policy = self._config(db)
            rows = [dict(r) for r in db.execute("SELECT * FROM edge_queue WHERE attempts < 3")]
        chosen = ordered(rows, watchlist, policy, time())[:max(1, min(5, budget["max_jobs"]))]
        result = {"imported": 0, "duplicates": 0, "errors": 0, "rss_only": 0, "pdf": 0, "budget": budget}
        for row in chosen:
            started = perf_counter()
            started_at = time()
            try:
                document, warnings = prepare(row)
                result["errors"] += warnings
                with self.inbox.transaction(write=True) as db:
                    if not db.execute("SELECT 1 FROM edge_queue WHERE feed_key=?", (row["feed_key"],)).fetchone():
                        continue
                    if db.execute("SELECT 1 FROM live_receipts WHERE feed_key=?", (row["feed_key"],)).fetchone():
                        db.execute("DELETE FROM edge_queue WHERE feed_key=?", (row["feed_key"],))
                        result["duplicates"] += 1
                        continue
                    previous = db.execute(
                        "SELECT description, disclosure_id, published_at FROM edge_history WHERE ticker=? AND link=? AND link<>'' ORDER BY id DESC LIMIT 1",
                        (row["ticker"], row["link"])).fetchone()
                    delta = revision_delta(previous, row)
                    entry, duplicate = self.inbox._record(db, document)
                    db.execute("INSERT INTO live_receipts VALUES (?, ?)", (row["feed_key"], entry["id"]))
                    db.execute("""INSERT INTO edge_history (feed_key,ticker,link,description,published_at,disclosure_id,wait_ms,processing_ms,delta,relevant,completed_at)
                                  VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                               (row["feed_key"], row["ticker"], row["link"], row["description"], row["published_at"],
                                entry["id"], round(max(0, started_at-row["received"])*1000, 2),
                                round((perf_counter()-started)*1000, 2), json.dumps(delta),
                                int(row["ticker"] in watchlist), time()))
                    db.execute("DELETE FROM edge_queue WHERE feed_key=?", (row["feed_key"],))
                    db.execute("DELETE FROM edge_history WHERE id NOT IN (SELECT id FROM edge_history ORDER BY id DESC LIMIT ?)", (HISTORY,))
                result["duplicates"] += int(duplicate)
                result["imported"] += int(not duplicate)
                if not duplicate:
                    result["pdf" if document.source_scope == "pdf_text" else "rss_only"] += 1
            except ValueError:
                # Bounded failure retries; a failed job is visible and retained.
                with self.inbox.transaction(write=True) as db:
                    db.execute("UPDATE edge_queue SET attempts=attempts+1 WHERE feed_key=?", (row["feed_key"],))
                result["errors"] += 1
        result["pending"] = self.status()["pending"]
        return result


def benchmark(model):
    """Same six synthetic documents, two actual inference passes per policy."""
    workload = [
        ("TCS", "The company sent instructions for remote voting before its annual general meeting."),
        ("RELIANCE", "The board approved the minutes of the previous meeting."),
        ("HDFCBANK", "The company published its audited annual financial statements."),
        ("INFY", "The company reported quarterly earnings and revenue of INR 400 crore."),
        ("TMPV", "The company announced a price increase of up to 0.5% across its vehicle lineup."),
        ("INFY", "The company issued a correction to its reported revenue of INR 420 crore."),
    ]
    watchlist = ["INFY", "TMPV"]
    rows = [{"feed_key": str(i), "received": 0, "ticker": ticker, "description": text}
            for i, (ticker, text) in enumerate(workload)]
    reports = {"fifo": [], "adaptive": []}
    # Warm-up excluded; alternating ordering reduces warm-cache/order effects.
    model.analyze(Document("INFY", (workload[3][1],), "Warmup", origin="synthetic"))
    for policy in ("fifo", "adaptive", "adaptive", "fifo"):
        start = perf_counter()
        times = []
        output = []
        for row in ordered(rows, watchlist, policy, 0):
            brief = model.analyze(Document(row["ticker"], (row["description"],), "Benchmark", origin="synthetic"))
            elapsed = round((perf_counter()-start)*1000, 2)
            times.append(elapsed)
            output.append({"job": row["feed_key"], "ticker": row["ticker"], "category": brief["category"],
                           "relevant": row["ticker"] in watchlist, "completion_ms": elapsed})
        relevant = [r["completion_ms"] for r in output if r["relevant"]]
        reports[policy].append({"first_relevant_ms": min(relevant), "all_relevant_ms": max(relevant),
                                "total_ms": times[-1], "jobs": output})
    signatures = [{r["job"]: r["category"] for r in run["jobs"]} for runs in reports.values() for run in runs]
    return {"scope": "Six authored synthetic texts, actual local model inference, two runs per policy; not live-feed latency.",
            "model": model.name, "same_categories": all(s == signatures[0] for s in signatures),
            "watchlist": watchlist, "runs": reports}
