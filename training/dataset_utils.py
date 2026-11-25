"""Prepare chunk- and document-level training examples from curated JSONL splits."""
from __future__ import annotations

from collections import Counter
from typing import Any

from stockbrief.nlp import chunks_for
from stockbrief.semantic import EVENT_BOUNDARY
from training.schema import VALID_LABELS, load_jsonl, normalize_text


def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def build_training_samples(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, int]]:
    """Build deduplicated chunk + single-event document training pairs without mixing multi-event docs."""
    texts: list[str] = []
    labels: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add(txt: str, lbl: str) -> None:
        clean = txt.strip()
        if len(clean) < 12 or lbl not in VALID_LABELS:
            return
        key = (normalize_text(clean), lbl)
        if key not in seen:
            seen.add(key)
            texts.append(clean)
            labels.append(lbl)

    for row in rows:
        if row["source_mode"] == "pdf_extraction_failed":
            continue
        full_text = row["text"]
        doc_labels = row["labels"]

        if len(doc_labels) == 1:
            primary = row["label"]
            add(full_text, primary)
            # Also add sentence/event chunks so chunk-level inference matches document-level behavior
            chunks = chunks_for((full_text,), EVENT_BOUNDARY)
            if len(chunks) > 1:
                ev_spans = row.get("evidence_offsets", [])
                for ch in chunks:
                    if ev_spans:
                        best_ov = max(overlap_len(ch.start, ch.end, sp["start"], sp["end"]) for sp in ev_spans)
                        if best_ov >= 15:
                            add(ch.text, primary)
                    elif primary == "review":
                        add(ch.text, "review")
        else:
            # Multi-label document: use only the labeled evidence spans and matching chunks per label
            ev_spans = row.get("evidence_offsets", [])
            for sp in ev_spans:
                add(sp["text"], sp["label"])
            chunks = chunks_for((full_text,), EVENT_BOUNDARY)
            for ch in chunks:
                matched_lbls = {
                    sp["label"]
                    for sp in ev_spans
                    if overlap_len(ch.start, ch.end, sp["start"], sp["end"]) >= 20
                }
                if len(matched_lbls) == 1:
                    add(ch.text, next(iter(matched_lbls)))

    return texts, labels, dict(Counter(labels))
