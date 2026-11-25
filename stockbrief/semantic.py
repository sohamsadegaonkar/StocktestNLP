"""Verified CPU sentence encoder and supervised disclosure classifier."""
from dataclasses import asdict
from functools import lru_cache
from hashlib import file_digest
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression

from .fixtures import Document
from .nlp import FIGURES, MAX_CHUNKS, PATTERN, Chunk, chunks_for, stage_for
from .semantic_data import SEMANTIC_LABELS, SEMANTIC_TRAIN

MODEL_NAME = "BGE-small-en-v1.5 INT8 / supervised classifier"
MODEL_VERSION = "stockbrief-bge-events-v1"
REPOSITORY = "Xenova/bge-small-en-v1.5"
REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"
WEIGHTS = "onnx/model_quantized.onnx"
CHECKSUMS = {
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    WEIGHTS: "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
}
BATCH_SIZE = 12

EVENT_BOUNDARY = re.compile(
    PATTERN.pattern
    + r"|(?:\r?\n)[ \t]*(?=[•*\-][ \t]+)"
    + r"|(?<=,)\s+(?=(?:while|and)\s+(?:report\w*|announc\w*|declar\w*|approv\w*)\b)"
)
DEAL_CONDITION = re.compile(
    r"^\W*(?:the\s+)?(?:transaction|deal|scheme|agreement|merger|demerger)\s+"
    r"(?:is|was|remains?|remained|will remain)\s+(?:subject to|conditional on|pending)\b", re.I,
)
FIGURE_WITH_QUALIFIER = re.compile(
    r"(?:(?:\bup to|\baverage of|\broughly|\bapproximately)\s+)?(?:" + FIGURES.pattern + ")", re.I,
)
DATES = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,\s*\d{4})?"
    r"|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?"
    r"|Q[1-4]\s+FY\s*\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.I,
)


def model_directory(path: Path | None = None) -> Path:
    return (path or Path(os.environ.get("STOCKBRIEF_SEMANTIC_MODEL_DIR", "models/semantic"))).resolve()


def verified(path: Path, checksum: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest() == checksum


def fetch_semantic(folder: Path) -> None:
    """Write each checked model artifact atomically."""
    for name, checksum in CHECKSUMS.items():
        target = folder / name
        if verified(target, checksum):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with httpx.stream(
                    "GET", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                    timeout=45, follow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    for block in response.iter_bytes(65536):
                        output.write(block)
            if not verified(temporary, checksum):
                raise ValueError(f"Downloaded model file failed SHA-256 verification: {name}")
            temporary.replace(target)
        except httpx.HTTPError:
            raise RuntimeError(f"Unable to download the pinned semantic model file {name}.") from None
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


class SentenceEncoder:
    def __init__(self, folder: Path):
        for name, checksum in CHECKSUMS.items():
            if not verified(folder / name, checksum):
                raise ValueError(
                    f"Semantic model file {name} is missing or invalid. Run python -m stockbrief fetch-semantic."
                )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"))
        self.tokenizer.enable_truncation(max_length=160)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / WEIGHTS), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.inputs = [item.name for item in self.session.get_inputs()]
        self.artifact = {
            "repository": REPOSITORY, "revision": REVISION, "weights_sha256": CHECKSUMS[WEIGHTS],
            "weights_bytes": (folder / WEIGHTS).stat().st_size, "onnxruntime_version": ort.__version__,
            "execution_providers": self.session.get_providers(),
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        batches = []
        for offset in range(0, len(texts), BATCH_SIZE):
            tokens = self.tokenizer.encode_batch(texts[offset:offset + BATCH_SIZE])
            fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
            feed = {key: np.asarray([getattr(token, fields[key]) for token in tokens], dtype=np.int64)
                    for key in self.inputs}
            pooled = self.session.run(None, feed)[0][:, 0, :]
            norm = np.maximum(1e-9, np.linalg.norm(pooled, axis=1, keepdims=True))
            batches.append(pooled / norm)
        return np.concatenate(batches)


def extract_values(pattern, chunks: list[Chunk]) -> list[dict]:
    values = {}
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            item = {"value": match.group(), "page": chunk.page, "start": chunk.start + match.start()}
            values[(item["page"], item["start"])] = item
    return list(values.values())[:12]


class SemanticEventModel:
    name = MODEL_NAME
    version = MODEL_VERSION
    labels = SEMANTIC_LABELS
    # Selected using only the independent calibration fixtures.
    floor = 0.35
    margin = 0.0

    def __init__(self, encoder: SentenceEncoder):
        self.encoder = encoder
        self.artifact = encoder.artifact
        samples = [(text, label) for label, texts in SEMANTIC_TRAIN.items() for text in texts]
        texts, labels = zip(*samples)
        embeddings = encoder.encode(list(texts))
        self.head = LogisticRegression(C=40, max_iter=600, random_state=19).fit(embeddings, labels)
        self.classes = self.head.classes_.tolist()

    def analyze(self, document: Document) -> dict:
        started = perf_counter_ns()
        chunks = chunks_for(document.pages, EVENT_BOUNDARY)
        if not chunks:
            raise ValueError("Provide at least 12 characters of readable disclosure text.")
        texts = [chunk.text for chunk in chunks]
        sentence_scores = self.head.predict_proba(self.encoder.encode(texts))
        events = {}
        guesses = []
        for index, row in enumerate(sentence_scores):
            winner = int(np.argmax(row))
            label = self.classes[winner]
            # An approval condition describes the preceding deal's stage.
            if index and guesses[-1] == "deal" and DEAL_CONDITION.search(texts[index]):
                guesses.append("review")
                continue
            guesses.append(label if row[winner] >= self.floor else "review")
            if guesses[-1] != "review":
                events.setdefault(label, []).append(index)

        results = []
        for label, positions in events.items():
            class_index = self.classes.index(label)
            ranked = sorted(positions, key=lambda index: -sentence_scores[index, class_index])
            nearby = {
                adjacent for position in ranked[:2] for adjacent in (position - 1, position + 1)
                if 0 <= adjacent < len(chunks)
                and chunks[adjacent].page == chunks[position].page
                and guesses[adjacent] == "review"
            }
            evidence = [chunks[i] for i in sorted(ranked[:4])]
            stage_context = [chunks[i] for i in sorted(set(ranked[:5]) | nearby)]
            # A continuation can supply status, while figures remain tied to event-bearing text.
            stage, stage_evidence = stage_for(label, stage_context)
            exact_quotes = [asdict(chunk) for chunk in evidence]
            if stage_evidence and stage_evidence not in exact_quotes:
                exact_quotes.append(stage_evidence)
            results.append({
                "category": label, "category_name": self.labels[label],
                "score": round(float(sentence_scores[ranked[0], class_index]), 4),
                "stage": stage, "stage_evidence": stage_evidence, "evidence": exact_quotes,
                "figures": extract_values(FIGURE_WITH_QUALIFIER, evidence),
                "dates": extract_values(DATES, evidence),
            })

        if results:
            primary = max(results, key=lambda item: item["score"])
            candidate = primary["category"]
        else:
            strength = sentence_scores.max(axis=0)
            ranking = np.argsort(-strength, kind="stable")
            candidate = next(self.classes[index] for index in ranking if self.classes[index] != "review")
            primary = {
                "category": "review", "category_name": self.labels["review"],
                "review_message": "Review the excerpts for a company event with enough supporting detail.",
                "score": round(float(strength[self.classes.index(candidate)]), 4),
                "stage": "Unclear", "stage_evidence": None,
                "evidence": [asdict(chunk) for chunk in chunks[:2]],
                "figures": [], "dates": [],
            }

        return {
            **primary, "candidate": candidate, "candidate_name": self.labels[candidate],
            "scores": {key: round(float(value), 4) for key, value in zip(self.classes, sentence_scores.max(axis=0))},
            "events": results, "chunk_count": len(chunks), "chunk_limit_reached": len(chunks) == MAX_CHUNKS,
            "model": self.name, "model_version": self.version,
            "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
        }


@lru_cache(maxsize=2)
def load_semantic(folder: Path) -> SemanticEventModel:
    return SemanticEventModel(SentenceEncoder(folder))
