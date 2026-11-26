"""Transactional SQLite inbox and source-link simulation."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from unicodedata import normalize

from .fixtures import Document, REPLAY, SEED
from .edge import SCHEMA as EDGE_SCHEMA

SCHEMA = """
CREATE TABLE IF NOT EXISTS disclosures (
    id INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    origin TEXT NOT NULL,
    payload TEXT NOT NULL,
    brief TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_receipts (
    feed_key TEXT PRIMARY KEY,
    disclosure_id INTEGER NOT NULL REFERENCES disclosures(id)
);
CREATE TABLE IF NOT EXISTS replay (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    connected INTEGER NOT NULL,
    cursor INTEGER NOT NULL,
    duplicates INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    packet INTEGER PRIMARY KEY
);
"""


def fingerprint(document: Document) -> str:
    text = "\n".join(document.pages)
    text = re.sub(r"\s+", " ", normalize("NFKC", text)).strip()
    has_source = document.source != "Local import"
    fields = [document.ticker, document.source.casefold() if has_source else "", document.date if has_source else "", text]
    if document.origin in {"nse-rss", "nse-pdf"}:
        fields.extend((document.origin, document.url, document.published_at))
    return sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


class Inbox:
    def __init__(self, path: Path, model):
        self.path = path
        self.model = model
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction(write=True, initialize=True) as db:
            if db.execute("INSERT OR IGNORE INTO replay VALUES (1, 1, 0, 0)").rowcount:
                for document in SEED:
                    self._record(db, document)

    @contextmanager
    def transaction(self, write=False, initialize=False):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if initialize:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(SCHEMA + EDGE_SCHEMA)
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _entry(row, include_pages=False):
        payload = json.loads(row["payload"])
        if not include_pages:
            payload.pop("pages")
        return {"id": row["id"], **payload, "brief": json.loads(row["brief"])}

    def _record(self, db, document):
        key = fingerprint(document)
        row = db.execute("SELECT * FROM disclosures WHERE fingerprint = ?", (key,)).fetchone()
        if row:
            entry = self._entry(row)
            version = getattr(self.model, "version", None)
            if version and entry["brief"].get("model_version") != version:
                payload = json.loads(row["payload"])
                payload["pages"] = tuple(payload["pages"])
                entry["brief"] = self.model.analyze(Document(**payload))
                db.execute("UPDATE disclosures SET brief = ? WHERE id = ?", (json.dumps(entry["brief"]), row["id"]))
            return entry, True
        brief = self.model.analyze(document)
        result = db.execute(
            "INSERT INTO disclosures(fingerprint, origin, payload, brief) VALUES (?, ?, ?, ?)",
            (key, document.origin, json.dumps(document.record()), json.dumps(brief)),
        )
        entry = document.record()
        entry.pop("pages")
        return {"id": result.lastrowid, **entry, "brief": brief}, False

    def record(self, document):
        with self.transaction(write=True) as db:
            return self._record(db, document)

    def live_seen(self, feed_key):
        with self.transaction() as db:
            return db.execute("SELECT 1 FROM live_receipts WHERE feed_key = ?", (feed_key,)).fetchone() is not None

    def record_live(self, feed_key, document):
        with self.transaction(write=True) as db:
            row = db.execute(
                "SELECT d.* FROM disclosures d JOIN live_receipts r ON d.id = r.disclosure_id WHERE r.feed_key = ?",
                (feed_key,),
            ).fetchone()
            if row:
                return self._entry(row), True
            entry, duplicate = self._record(db, document)
            db.execute("INSERT INTO live_receipts VALUES (?, ?)", (feed_key, entry["id"]))
            return entry, duplicate

    def document(self, entry_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM disclosures WHERE id = ?", (entry_id,)).fetchone()
            return self._entry(row, include_pages=True) if row else None

    def _state(self, db):
        replay = db.execute("SELECT * FROM replay WHERE id = 1").fetchone()
        pending = db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        return {
            "connected": bool(replay["connected"]),
            "cursor": replay["cursor"],
            "total_packets": len(REPLAY),
            "duplicates": replay["duplicates"],
            "pending": pending,
        }

    def state(self):
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM disclosures ORDER BY id DESC").fetchall()
            return {**self._state(db), "disclosures": [self._entry(row) for row in rows]}

    def _drain(self, db):
        packets = db.execute("SELECT packet FROM pending ORDER BY packet").fetchall()
        for (packet,) in packets:
            _, duplicate = self._record(db, REPLAY[packet])
            if duplicate:
                db.execute("UPDATE replay SET duplicates = duplicates + 1 WHERE id = 1")
            db.execute("DELETE FROM pending WHERE packet = ?", (packet,))

    def step(self):
        with self.transaction(write=True) as db:
            state = self._state(db)
            cursor = state["cursor"]
            if cursor < len(REPLAY):
                db.execute("INSERT INTO pending VALUES (?)", (cursor,))
                db.execute("UPDATE replay SET cursor = cursor + 1 WHERE id = 1")
                if state["connected"]:
                    self._drain(db)
            return self._state(db)

    def link(self, connected):
        with self.transaction(write=True) as db:
            db.execute("UPDATE replay SET connected = ? WHERE id = 1", (connected,))
            if connected:
                self._drain(db)
            return self._state(db)

    def reset(self):
        with self.transaction(write=True) as db:
            db.execute("DELETE FROM pending")
            db.execute("DELETE FROM disclosures WHERE origin IN ('historical', 'synthetic')")
            db.execute("UPDATE replay SET connected = 1, cursor = 0, duplicates = 0 WHERE id = 1")
            for document in SEED:
                self._record(db, document)
            return self._state(db)
