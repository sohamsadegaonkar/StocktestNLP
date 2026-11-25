import asyncio
from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from httpx import ASGITransport, AsyncClient
from threadpoolctl import threadpool_limits

from stockbrief.app import create_app
from stockbrief.corpus import LABELS
from stockbrief.evaluation import evaluate
from stockbrief.fixtures import ACQUISITION, CASE_STUDY
from stockbrief import transformer


class ModelArtifactTests(unittest.TestCase):
    def test_fetch_uses_pinned_revision_and_marks_only_verified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in transformer.FILES:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test artifact")

            download = Mock()
            hub = SimpleNamespace(snapshot_download=download)
            marker = folder / "stockbrief-revision.txt"
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                with self.assertRaisesRegex(ValueError, "SHA-256 verification"):
                    transformer.fetch_model(folder)
                self.assertFalse(marker.exists())
                download.assert_called_once_with(
                    repo_id=transformer.REPOSITORY,
                    revision=transformer.REVISION,
                    local_dir=folder,
                    allow_patterns=list(transformer.FILES),
                )

                digest = sha256(b"test artifact").hexdigest()
                with patch.object(transformer, "WEIGHTS_SHA256", digest):
                    self.assertEqual(transformer.fetch_model(folder), transformer.REVISION)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), transformer.REVISION)

    def test_load_checks_artifact_before_importing_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Model files are missing"):
                transformer.Transformer(Path(directory))

    def test_evaluation_adds_provenance_and_historical_category_comparison(self):
        learned = SimpleNamespace(
            floor=0.5, margin=0.1,
            analyze=lambda document: {"category": "review", "stage": "Needs review"},
        )
        comparator = SimpleNamespace(
            name="Fixture comparator",
            artifact={"revision": transformer.REVISION, "weights_sha256": transformer.WEIGHTS_SHA256},
            classify=lambda document: "deal" if document == ACQUISITION else "review",
        )
        self.assertEqual(len(evaluate(learned)["reports"]), 2)
        report = evaluate(learned, comparator)["reports"][-1]
        self.assertEqual(report["artifact"], comparator.artifact)
        self.assertEqual(len(report["historical_cases"]), len(CASE_STUDY))
        self.assertEqual(report["historical_cases"][0], {
            "title": ACQUISITION.title,
            "source": ACQUISITION.url,
            "expected_category": "deal",
            "observed_category": "deal",
            "category_matched": True,
        })
        self.assertFalse(report["historical_cases"][1]["category_matched"])


@unittest.skipUnless(os.environ.get("STOCKBRIEF_MODEL_DIR"), "requires the downloaded MobileBERT model")
class RealModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_seeded_disclosure_second_opinion_uses_verified_cpu_model(self):
        model_dir = Path(os.environ["STOCKBRIEF_MODEL_DIR"])
        comparator = await asyncio.to_thread(transformer.Transformer, model_dir)
        self.assertEqual(comparator.artifact["repository"], transformer.REPOSITORY)
        self.assertEqual(comparator.artifact["revision"], transformer.REVISION)
        self.assertEqual(comparator.artifact["weights_sha256"], transformer.WEIGHTS_SHA256)
        self.assertEqual(comparator.artifact["execution_providers"], ["CPUExecutionProvider"])
        self.assertTrue(comparator.artifact["onnxruntime_version"])

        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            app = await asyncio.to_thread(create_app, Path(directory) / "inbox.sqlite", comparator)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                inbox = await client.get("/api/inbox")
                self.assertEqual(inbox.status_code, 200)
                entries = inbox.json()["disclosures"]
                entry = next(record for record in entries if record["title"] == ACQUISITION.title)
                response = await client.get(f"/api/disclosures/{entry['id']}/second-opinion")
                self.assertEqual(response.status_code, 200)
                result = response.json()
                category_names = {**LABELS, "review": "Needs review"}
                self.assertIn(result["category"], category_names)
                self.assertEqual(result, {
                    "category": result["category"],
                    "name": category_names[result["category"]],
                    "model": comparator.name,
                })
