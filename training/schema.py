"""Dataset schema, label contract, and integrity checks for domain event training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from stockbrief.semantic_data import SEMANTIC_LABELS

VALID_LABELS = tuple(SEMANTIC_LABELS.keys())
SUBSTANTIVE_LABELS = tuple(k for k in VALID_LABELS if k != "review")

VALID_SOURCE_MODES = (
    "excerpt",
    "paraphrase",
    "headline_only",
    "pdf_extraction_failed",
)

VALID_INSUFFICIENCY_TYPES = (
    "none",
    "headline_only",
    "insufficient_detail",
    "pdf_extraction_failed",
    "rumor_denial",
    "non_financial",
)

VALID_STAGES = (
    "Reported",
    "Announced",
    "Declared",
    "Recommended",
    "Approved",
    "Proposed",
    "Board approved",
    "Agreement signed",
    "Subject to approvals",
    "Completed",
    "Terminated",
    "Denied",
    "Appointment",
    "Resignation",
    "Notice received",
    "Penalty imposed",
    "Under appeal",
    "Contract awarded",
    "Disruption reported",
    "Operations resumed",
    "Scheduled",
    "Unclear",
)

SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


@dataclass
class EvidenceSpan:
    start: int
    end: int
    text: str
    label: str
    page: int = 1


@dataclass
class PreservedFacts:
    amounts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)


@dataclass
class AnnotatedDocument:
    doc_id: str
    source: str
    url: str
    issuer_symbol: str
    issuer_name: str
    date: str
    raw_date: str
    nse_desc: str
    event_family: str
    source_mode: str
    text: str
    pages: list[str]
    label: str
    labels: list[str]
    stage: str
    evidence_offsets: list[dict[str, Any]]
    insufficiency_type: str
    preserves: dict[str, list[str]]
    license_note: str
    annotator: str = "curated-v1"
    reviewed: bool = True
    split: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.doc_id:
            errors.append("missing doc_id")
        if not self.url or not self.url.startswith("http"):
            errors.append(f"{self.doc_id}: invalid url {self.url!r}")
        if not self.issuer_symbol:
            errors.append(f"{self.doc_id}: missing issuer_symbol")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.date):
            errors.append(f"{self.doc_id}: invalid ISO date {self.date!r}")
        if not self.event_family:
            errors.append(f"{self.doc_id}: missing event_family")
        if self.source_mode not in VALID_SOURCE_MODES:
            errors.append(f"{self.doc_id}: invalid source_mode {self.source_mode!r}")
        if self.label not in VALID_LABELS:
            errors.append(f"{self.doc_id}: invalid label {self.label!r}")
        if not self.labels or any(lbl not in VALID_LABELS for lbl in self.labels):
            errors.append(f"{self.doc_id}: invalid labels {self.labels!r}")
        if self.label not in self.labels:
            errors.append(f"{self.doc_id}: primary label {self.label!r} not in labels {self.labels!r}")
        if self.stage not in VALID_STAGES:
            errors.append(f"{self.doc_id}: invalid stage {self.stage!r}")
        if self.insufficiency_type not in VALID_INSUFFICIENCY_TYPES:
            errors.append(f"{self.doc_id}: invalid insufficiency_type {self.insufficiency_type!r}")
        if not self.pages:
            errors.append(f"{self.doc_id}: missing pages")
        # Verify exact character offsets against page text.
        for span in self.evidence_offsets:
            page_idx = int(span.get("page", 1)) - 1
            if not (0 <= page_idx < len(self.pages)):
                errors.append(f"{self.doc_id}: invalid evidence page {page_idx + 1}")
                continue
            page_text = self.pages[page_idx]
            start, end = int(span["start"]), int(span["end"])
            if not (0 <= start <= end <= len(page_text)):
                errors.append(f"{self.doc_id}: out-of-bounds offset [{start}:{end}]")
            elif page_text[start:end] != span["text"]:
                errors.append(f"{self.doc_id}: offset text mismatch at [{start}:{end}]")
            if span.get("label") not in VALID_LABELS:
                errors.append(f"{self.doc_id}: invalid span label {span.get('label')!r}")
        # Verify preserved amounts/dates/qualifiers/negations appear in source text.
        full_text = "\n".join(self.pages)
        lower_text = full_text.lower()
        for key in ("amounts", "dates", "qualifiers", "negations"):
            for item in self.preserves.get(key, []):
                if item.lower() not in lower_text:
                    errors.append(f"{self.doc_id}: preserved {key} item {item!r} not found in text")
        if self.source_mode == "pdf_extraction_failed" and self.label != "review":
            errors.append(f"{self.doc_id}: pdf_extraction_failed must map to 'review'")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
