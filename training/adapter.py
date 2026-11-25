"""Adapter contract matching SemanticEventModel.analyze(document) for trained PyTorch/ONNX models."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np

from stockbrief.fixtures import Document
from stockbrief.nlp import MAX_CHUNKS, Chunk, chunks_for, stage_for
from stockbrief.semantic import DEAL_CONDITION, EVENT_BOUNDARY, extract_values
from stockbrief.semantic_data import SEMANTIC_LABELS

try:
    from stockbrief.semantic import DATES, FIGURE_WITH_QUALIFIER
except ImportError:  # pragma: no cover
    from stockbrief.nlp import FIGURES
    FIGURE_WITH_QUALIFIER = FIGURES
    DATES = re.compile(r"\b(?:Q[1-4]\s+FY\s*\d{2,4}|\d{4}-\d{2}-\d{2})\b", re.I)

SCHEDULED_RE = re.compile(
    r"\b(?:is\s+scheduled|scheduled\s+(?:on|to\s+be\s+held)|will\s+be\s+held|shall\s+be\s+held|"
    r"prior\s+(?:notice|intimation)\s+of\s+board\s+meeting|trading\s+window\s+(?:remains?|will\s+remain)\s+closed)\b",
    re.I,
)
DENIED_RE = re.compile(
    r"\b(?:denied|denies|categorically\s+den(?:y|ies|ied)|refutes|rejects|speculative\s+in\s+nature|"
    r"baseless|factually\s+incorrect|completely\s+false|absolutely\s+no\s+truth|"
    r"no\s+such\s+(?:proposal|talks|deal|acquisition|agreement)|has\s+not\s+received)\b",
    re.I,
)
GARBLED_TEXT_RE = re.compile(r"([a-zA-Z])\1{12,}")


class ChunkScorer(Protocol):
    name: str
    version: str
    classes: list[str]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        ...


def domain_stage_for(label: str, candidates: list[Chunk]) -> tuple[str, dict | None]:
    """Extend repo stage_for with Scheduled and broader rumor-denial coverage without modifying stockbrief/nlp.py."""
    if label == "routine":
        for chunk in candidates:
            norm = " ".join(chunk.text.split())
            if SCHEDULED_RE.search(norm):
                return "Scheduled", asdict(chunk)
        return "Announced", asdict(candidates[0]) if candidates else None
    if label in ("deal", "pricing", "regulatory", "operations"):
        for chunk in candidates:
            norm = " ".join(chunk.text.split())
            if DENIED_RE.search(norm):
                return "Denied", asdict(chunk)
    stage, ev = stage_for(label, candidates)
    if stage != "Unclear":
        return stage, ev
    if label == "deal" and candidates:
        return "Proposed", asdict(candidates[0])
    if label == "regulatory" and candidates:
        return "Notice received", asdict(candidates[0])
    if label == "corporate_action" and candidates:
        return "Announced", asdict(candidates[0])
    if label == "governance" and candidates:
        return "Announced", asdict(candidates[0])
    if label == "operations" and candidates:
        return "Announced", asdict(candidates[0])
    return "Unclear", None


def is_readable_text(pages: tuple[str, ...]) -> bool:
    joined = "\n".join(pages).strip()
    if sum(c.isalnum() for c in joined) < 12:
        return False
    ascii_letters = sum("a" <= c.lower() <= "z" for c in joined)
    if (ascii_letters / max(1, len(joined))) < 0.30:
        return False
    if GARBLED_TEXT_RE.search(joined):
        return False
    return True


class DomainEventAdapter:
    """Drop-in SemanticEventModel.analyze(document) compatible adapter."""

    labels = SEMANTIC_LABELS

    def __init__(
        self,
        scorer: ChunkScorer,
        floor: float = 0.45,
        margin: float = 0.0,
        artifact: dict[str, Any] | None = None,
    ):
        self.scorer = scorer
        self.name = scorer.name
        self.version = scorer.version
        self.classes = scorer.classes
        self.floor = float(floor)
        self.margin = float(margin)
        self.artifact = artifact or {}

    def analyze(self, document: Document) -> dict[str, Any]:
        started = perf_counter_ns()
        if not is_readable_text(document.pages):
            return {
                "category": "review",
                "category_name": self.labels["review"],
                "review_message": "Document text could not be reliably extracted; route for manual review.",
                "score": 0.0,
                "stage": "Unclear",
                "stage_evidence": None,
                "evidence": [],
                "figures": [],
                "dates": [],
                "candidate": "routine",
                "candidate_name": self.labels["routine"],
                "scores": {k: 0.0 for k in self.classes},
                "events": [],
                "chunk_count": 0,
                "chunk_limit_reached": False,
                "model": self.name,
                "model_version": self.version,
                "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
            }

        chunks = chunks_for(document.pages, EVENT_BOUNDARY)
        if not chunks:
            raise ValueError("Provide at least 12 characters of readable disclosure text.")
        texts = [c.text for c in chunks]
        sentence_scores = self.scorer.predict_proba(texts)

        events: dict[str, list[int]] = {}
        guesses: list[str] = []
        for idx, row in enumerate(sentence_scores):
            order = np.argsort(row)[::-1]
            top_i, second_i = int(order[0]), int(order[1])
            lbl = self.classes[top_i]
            top_p = float(row[top_i])
            sec_p = float(row[second_i])
            if idx and guesses[-1] == "deal" and DEAL_CONDITION.search(texts[idx]):
                guesses.append("review")
                continue
            if lbl == "review" or top_p < self.floor or (top_p - sec_p) < self.margin:
                guesses.append("review")
            else:
                guesses.append(lbl)
                events.setdefault(lbl, []).append(idx)

        results: list[dict[str, Any]] = []
        for lbl, positions in events.items():
            class_idx = self.classes.index(lbl)
            ranked = sorted(positions, key=lambda i: -sentence_scores[i, class_idx])
            nearby = {
                adj
                for pos in ranked[:2]
                for adj in (pos - 1, pos + 1)
                if 0 <= adj < len(chunks) and chunks[adj].page == chunks[pos].page and guesses[adj] == "review"
            }
            evidence = [chunks[i] for i in sorted(ranked[:4])]
            stage_ctx = [chunks[i] for i in sorted(set(ranked[:5]) | nearby)]
            stage, stage_ev = domain_stage_for(lbl, stage_ctx)
            exact_quotes = [asdict(chunk) for chunk in evidence]
            if stage_ev and stage_ev not in exact_quotes:
                exact_quotes.append(stage_ev)
            results.append({
                "category": lbl,
                "category_name": self.labels[lbl],
                "score": round(float(sentence_scores[ranked[0], class_idx]), 4),
                "stage": stage,
                "stage_evidence": stage_ev,
                "evidence": exact_quotes,
                "figures": extract_values(FIGURE_WITH_QUALIFIER, evidence),
                "dates": extract_values(DATES, evidence),
            })

        if results:
            primary = max(results, key=lambda item: item["score"])
            candidate = primary["category"]
        else:
            strength = sentence_scores.max(axis=0)
            ranking = np.argsort(-strength, kind="stable")
            candidate = next(self.classes[i] for i in ranking if self.classes[i] != "review")
            primary = {
                "category": "review",
                "category_name": self.labels["review"],
                "review_message": "Review the excerpts for a company event with enough supporting detail.",
                "score": round(float(strength[self.classes.index(candidate)]), 4),
                "stage": "Unclear",
                "stage_evidence": None,
                "evidence": [asdict(chunk) for chunk in chunks[:2]],
                "figures": [],
                "dates": [],
            }

        return {
            **primary,
            "candidate": candidate,
            "candidate_name": self.labels[candidate],
            "scores": {key: round(float(val), 4) for key, val in zip(self.classes, sentence_scores.max(axis=0))},
            "events": results,
            "chunk_count": len(chunks),
            "chunk_limit_reached": len(chunks) == MAX_CHUNKS,
            "model": self.name,
            "model_version": self.version,
            "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
        }


class ONNXEndToEndScorer:
    """ONNX Runtime CPU scorer for an exported encoder + linear classification head."""

    def __init__(self, model_dir: Path, weights_name: str = "model_quantized.onnx"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        self.name = meta["name"]
        self.version = meta["version"]
        self.classes = list(meta["classes"])
        self.max_length = int(meta.get("max_length", 192))
        self.batch_size = int(meta.get("batch_size", 12))

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        pad_token = meta.get("pad_token", "[PAD]")
        pad_id = self.tokenizer.token_to_id(pad_token)
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        self.tokenizer.enable_truncation(max_length=self.max_length)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / weights_name), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self.inputs = [i.name for i in self.session.get_inputs()]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probs_all = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            enc = self.tokenizer.encode_batch(batch)
            fields = {
                "input_ids": np.asarray([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in enc], dtype=np.int64),
            }
            logits = self.session.run(None, {k: fields[k] for k in self.inputs})[0]
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs_all.append(exp / np.sum(exp, axis=1, keepdims=True))
        return np.concatenate(probs_all, axis=0)
