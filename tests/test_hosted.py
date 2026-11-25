import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from threadpoolctl import threadpool_limits

from stockbrief.app import create_app
from stockbrief.hosted import COOKIE_NAME, public_https_origin

ORIGIN = "https://workspace.example"
SAME_ORIGIN = {"Origin": ORIGIN}
TEXT = "The board appointed an independent director for a five-year period."


class HostedAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "hosted.sqlite"
        self.time_limits = threadpool_limits(limits=1)
        self.addCleanup(self.time_limits.restore_original_limits)
        self.app = await asyncio.to_thread(create_app, self.path, public_origin=ORIGIN)

    def client(self, app=None, *, base_url=ORIGIN, cookies=None):
        client = AsyncClient(transport=ASGITransport(app=app or self.app), base_url=base_url, cookies=cookies)
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_dashboard_serves_from_hosted_origin_with_secure_session(self):
        client = self.client()
        homepage = await client.get("/")
        self.assertEqual(homepage.status_code, 200)
        self.assertIn("StockBrief", homepage.text)
        self.assertEqual(homepage.headers["Cache-Control"], "no-store")
        cookie = homepage.headers["Set-Cookie"]
        for attribute in (f"{COOKIE_NAME}=", "HttpOnly", "Secure", "Path=/", "SameSite=lax", "Max-Age=604800"):
            self.assertIn(attribute, cookie)
        inbox = await client.get("/api/inbox")
        self.assertEqual(inbox.json()["mode"], "hosted")
        self.assertEqual(inbox.json()["storage"], "sqlite")
        self.assertNotIn("Set-Cookie", inbox.headers)
        health = await client.get("/api/health")
        self.assertEqual(health.json()["inference"], "server")
        self.assertNotIn("process_rss_mb", health.json())

    async def test_isolated_upload_replay_and_resume(self):
        first, second = self.client(), self.client()
        for client in (first, second):
            self.assertEqual((await client.get("/api/inbox")).status_code, 200)
        self.assertNotEqual(first.cookies.get(COOKIE_NAME), second.cookies.get(COOKIE_NAME))
        upload = await first.post(
            "/api/analyze", headers=SAME_ORIGIN, data={"ticker": "OTHER", "title": "Browser one upload"},
            files={"file": ("disclosure.txt", TEXT.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        record = upload.json()["entry"]
        foreign = await second.get(f"/api/disclosures/{record['id']}")
        self.assertEqual(foreign.status_code, 404)
        own = await first.get(f"/api/disclosures/{record['id']}/intelligence")
        self.assertEqual(own.status_code, 200)
        self.assertEqual(own.json()["entry_id"], record["id"])
        self.assertEqual((await second.get(f"/api/disclosures/{record['id']}/intelligence")).status_code, 404)
        await first.post("/api/replay/link", headers=SAME_ORIGIN, json={"connected": False})
        for _ in range(2):
            step = await first.post("/api/replay/step", headers=SAME_ORIGIN)
            self.assertEqual(step.status_code, 200)
        self.assertTrue((await second.get("/api/inbox")).json()["connected"])
        reopened = await asyncio.to_thread(create_app, self.path, public_origin=ORIGIN)
        returning = self.client(reopened, cookies=first.cookies)
        state = (await returning.get("/api/inbox")).json()
        self.assertEqual(state["pending"], 2)
        self.assertEqual(len(state["disclosures"]), 4)
        saved = await returning.get(f"/api/disclosures/{record['id']}")
        self.assertIn(TEXT, saved.json()["pages"])
        self.assertNotIn("fingerprint", saved.json())
        resumed = await returning.post("/api/replay/link", headers=SAME_ORIGIN, json={"connected": True})
        self.assertEqual(resumed.json()["duplicates"], 1)
        await returning.post("/api/replay/reset", headers=SAME_ORIGIN)
        self.assertEqual((await returning.get(f"/api/disclosures/{record['id']}")).status_code, 200)
        erased = await returning.post("/api/workspace/clear", headers=SAME_ORIGIN)
        self.assertEqual(erased.status_code, 200)
        self.assertEqual(len((await returning.get("/api/inbox")).json()["disclosures"]), 3)

    async def test_exact_public_origin_works_behind_proxy_and_blocks_foreign_hosts(self):
        internal = self.client(base_url="http://workspace.example")
        response = await internal.post("/api/replay/step", headers=SAME_ORIGIN)
        self.assertEqual(response.status_code, 200)
        for headers in (
            {}, {"Origin": "http://workspace.example"}, {"Origin": "https://workspace.example.bad.invalid"},
            {**SAME_ORIGIN, "sec-fetch-site": "cross-site"},
        ):
            self.assertEqual((await internal.post("/api/replay/step", headers=headers)).status_code, 403)
        wrong_host = self.client(base_url="https://wrong.example")
        self.assertEqual((await wrong_host.get("/api/health")).status_code, 400)

    async def test_workspace_cap_keeps_the_original_and_allows_exact_retries(self):
        client = self.client()
        await client.get("/")
        payload = {"ticker": "OTHER", "title": "First", "text": TEXT}
        with patch("stockbrief.hosted.MAX_IMPORTS", 1):
            saved = await client.post("/api/analyze", headers=SAME_ORIGIN, data=payload)
            self.assertEqual(saved.status_code, 200)
            duplicate = await client.post("/api/analyze", headers=SAME_ORIGIN, data=payload)
            self.assertTrue(duplicate.json()["duplicate"])
            blocked = await client.post(
                "/api/analyze", headers=SAME_ORIGIN, data={**payload, "text": TEXT + " Management reported the decision."},
            )
            self.assertEqual(blocked.status_code, 429)
        self.assertEqual(len((await client.get("/api/inbox")).json()["disclosures"]), 4)


class HostedConfigurationTests(unittest.TestCase):
    def test_official_origin_requires_https_host_only(self):
        self.assertEqual(public_https_origin(ORIGIN + "/"), (ORIGIN, "workspace.example"))
        for origin in ("https://workspace.example/path", "http://workspace.example", "https://u@workspace.example"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                public_https_origin(origin)


@unittest.skipUnless(os.environ.get("STOCKBRIEF_TEST_DATABASE_URL"), "requires a disposable PostgreSQL database")
class HostedPostgresTests(unittest.IsolatedAsyncioTestCase):
    async def test_disclosures_and_replay_survive_two_instances(self):
        database_url = os.environ["STOCKBRIEF_TEST_DATABASE_URL"]
        with threadpool_limits(limits=1):
            first = await asyncio.to_thread(create_app, public_origin=ORIGIN, database_url=database_url)
            second = await asyncio.to_thread(create_app, public_origin=ORIGIN, database_url=database_url)
            async with (
                AsyncClient(transport=ASGITransport(app=first), base_url=ORIGIN) as client,
                AsyncClient(transport=ASGITransport(app=second), base_url=ORIGIN) as reopened,
            ):
                initial = await client.get("/api/inbox")
                self.assertEqual(initial.json()["storage"], "postgresql")
                try:
                    record = await client.post(
                        "/api/analyze", headers=SAME_ORIGIN,
                        data={"ticker": "OTHER", "title": "PostgreSQL import", "text": TEXT},
                    )
                    self.assertEqual(record.status_code, 200, record.text)
                    await client.post("/api/replay/link", headers=SAME_ORIGIN, json={"connected": False})
                    for _ in range(2):
                        await client.post("/api/replay/step", headers=SAME_ORIGIN)
                    unopened = await reopened.get("/api/inbox")
                    self.assertEqual(len(unopened.json()["disclosures"]), 3)
                    reopened.cookies.update(client.cookies)
                    state = (await reopened.get("/api/inbox")).json()
                    self.assertEqual((len(state["disclosures"]), state["pending"]), (4, 2))
                    resumed = await reopened.post("/api/replay/link", headers=SAME_ORIGIN, json={"connected": True})
                    self.assertEqual(resumed.json()["duplicates"], 1)
                finally:
                    await client.post("/api/workspace/clear", headers=SAME_ORIGIN)
