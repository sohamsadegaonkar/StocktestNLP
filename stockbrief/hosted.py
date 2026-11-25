"""Browser workspaces for HTTPS deployments with SQLite or PostgreSQL storage."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import secrets
import sqlite3
from time import time
from urllib.parse import urlsplit

from .fixtures import REPLAY, SEED
from .store import fingerprint

COOKIE_NAME = "__Host-stockbrief"
COOKIE_AGE = 7 * 24 * 60 * 60
COOKIE_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
MAX_IMPORTS = 12
MAX_WORKSPACES = 200
MAX_WORKSPACE_BYTES = 512 * 1024
SCHEMA = """
CREATE TABLE IF NOT EXISTS stockbrief_workspaces (
    workspace_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at BIGINT NOT NULL
)
"""


class WorkspaceLimit(Exception):
    """A workspace or the shared demo instance has reached its storage limit."""


def public_https_origin(origin):
    try:
        origin = origin.rstrip("/")
        parts = urlsplit(origin)
        hostname = parts.hostname
        port = parts.port
        expected_host = hostname if port in (None, 443) else f"{hostname}:{port}"
        if (
            not hostname or not re.fullmatch(r"[a-z0-9.-]+", hostname)
            or origin != f"https://{expected_host}" or parts.netloc != expected_host
        ):
            raise ValueError
        return origin, hostname
    except ValueError as error:
        raise ValueError("Hosted mode needs a canonical HTTPS origin.") from error


def workspace_session(cookie):
    new = not cookie or not COOKIE_PATTERN.fullmatch(cookie)
    token = secrets.token_urlsafe(32) if new else cookie
    return sha256(token.encode("ascii")).hexdigest(), token if new else None


class HostedWorkspaces:
    def __init__(self, path: Path, model, database_url=None):
        self.model = model
        self.path = path
        self.database_url = database_url or None
        self.storage = "postgresql" if self.database_url else "sqlite"
        if self.database_url:
            if not self.database_url.startswith(("postgres://", "postgresql://")):
                raise ValueError("Hosted database URL must use PostgreSQL.")
            import psycopg

            self.pg = psycopg
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as db:
                db.execute("PRAGMA journal_mode=WAL")
        with self.connection() as db:
            db.execute(SCHEMA)
        self.fixtures = {
            document: self._prepare(document) for document in dict.fromkeys((*SEED, *REPLAY))
        }

    @contextmanager
    def connection(self, *, write=False):
        if self.database_url:
            with self.pg.connect(self.database_url, connect_timeout=5) as db:
                if write:
                    db.execute("LOCK TABLE stockbrief_workspaces IN SHARE ROW EXCLUSIVE MODE")
                yield db
            return
        db = sqlite3.connect(self.path, timeout=5)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        finally:
            db.close()

    def execute(self, db, sql, params=()):
        return db.execute(sql.replace("?", "%s") if self.database_url else sql, params)

    def healthy(self):
        try:
            with self.connection() as db:
                db.execute("SELECT 1 FROM stockbrief_workspaces LIMIT 1")
            return True
        except (sqlite3.Error, self.pg.Error if self.database_url else OSError):
            return False

    def workspace(self, key):
        return BrowserInbox(self, key)

    def _prepare(self, document):
        return {
            "fingerprint": fingerprint(document),
            **document.record(),
            "brief": self.model.analyze(document),
        }

    def _fresh(self, next_id=1):
        state = {
            "connected": True, "cursor": 0, "duplicates": 0,
            "pending_packets": [], "disclosures": [], "next_id": next_id,
        }
        for document in SEED:
            self._add(state, self.fixtures[document])
        return state

    @staticmethod
    def _add(state, prepared):
        for entry in state["disclosures"]:
            if entry["fingerprint"] == prepared["fingerprint"]:
                return entry, True
        entry = {"id": state["next_id"], **prepared}
        state["next_id"] += 1
        state["disclosures"].append(entry)
        return entry, False

    def _load(self, db, key, now):
        row = self.execute(
            db, "SELECT payload FROM stockbrief_workspaces WHERE workspace_key = ? AND updated_at >= ?",
            (key, now - COOKIE_AGE),
        ).fetchone()
        return json.loads(row[0]) if row else None

    @contextmanager
    def snapshot(self, key, write=False):
        now = int(time())
        with self.connection(write=write) as db:
            saved = self._load(db, key, now)
            state = saved if saved is not None else self._fresh()
            yield state
            if not write:
                return
            encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_WORKSPACE_BYTES:
                raise WorkspaceLimit("This workspace is full. Clear it or use a shorter document.")
            if saved is None:
                self.execute(db, "DELETE FROM stockbrief_workspaces WHERE updated_at < ?", (now - COOKIE_AGE,))
                size = db.execute("SELECT COUNT(*) FROM stockbrief_workspaces").fetchone()[0]
                if size >= MAX_WORKSPACES:
                    raise WorkspaceLimit("The demo is at workspace capacity. Try again later.")
            self.execute(
                db,
                """INSERT INTO stockbrief_workspaces (workspace_key, payload, updated_at) VALUES (?, ?, ?)
                ON CONFLICT (workspace_key) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at""",
                (key, encoded, now),
            )


class BrowserInbox:
    def __init__(self, workspaces, key):
        self.workspaces = workspaces
        self.key = key

    @staticmethod
    def _entry(entry, include_pages=False):
        output = {key: value for key, value in entry.items() if key != "fingerprint"}
        if not include_pages:
            output.pop("pages")
        return output

    @staticmethod
    def _state(state):
        return {
            "connected": state["connected"], "cursor": state["cursor"],
            "total_packets": len(REPLAY), "duplicates": state["duplicates"],
            "pending": len(state["pending_packets"]),
        }

    def _drain(self, state):
        for packet in state["pending_packets"]:
            _, duplicate = self.workspaces._add(state, self.workspaces.fixtures[REPLAY[packet]])
            state["duplicates"] += int(duplicate)
        state["pending_packets"].clear()

    def state(self):
        with self.workspaces.snapshot(self.key) as state:
            return {
                **self._state(state), "disclosures": [
                    self._entry(row) for row in sorted(state["disclosures"], key=lambda row: -row["id"])
                ],
            }

    def record(self, document):
        prepared = self.workspaces._prepare(document)
        with self.workspaces.snapshot(self.key, write=True) as state:
            entry, duplicate = self.workspaces._add(state, prepared)
            if not duplicate:
                uploads = sum(row["origin"] not in {"historical", "synthetic"} for row in state["disclosures"])
                if uploads > MAX_IMPORTS:
                    raise WorkspaceLimit(f"This workspace can save up to {MAX_IMPORTS} imported disclosures.")
            return self._entry(entry), duplicate

    def document(self, entry_id):
        with self.workspaces.snapshot(self.key) as state:
            for entry in state["disclosures"]:
                if entry["id"] == entry_id:
                    return self._entry(entry, include_pages=True)
            return None

    def step(self):
        with self.workspaces.snapshot(self.key, write=True) as state:
            cursor = state["cursor"]
            if cursor < len(REPLAY):
                state["pending_packets"].append(cursor)
                state["cursor"] += 1
                if state["connected"]:
                    self._drain(state)
            return self._state(state)

    def link(self, connected):
        with self.workspaces.snapshot(self.key, write=True) as state:
            state["connected"] = connected
            if connected:
                self._drain(state)
            return self._state(state)

    def reset(self):
        with self.workspaces.snapshot(self.key, write=True) as state:
            imports = [row for row in state["disclosures"] if row["origin"] not in {"historical", "synthetic"}]
            state.update(self.workspaces._fresh(state["next_id"]))
            state["disclosures"].extend(imports)
            return self._state(state)

    def clear(self):
        with self.workspaces.connection(write=True) as db:
            self.workspaces.execute(db, "DELETE FROM stockbrief_workspaces WHERE workspace_key = ?", (self.key,))
        return self._state(self.workspaces._fresh())
