"""Phase 2 unified task schemas, tagsets, provenance tiers, and license tracking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

PROVENANCE_TIERS = (
    "human_reviewed",
    "deterministic",
    "weak_supervision",
    "synthetic_paraphrase",
)

LICENSE_TIERS = (
    "permissive_mit_or_public",
    "academic_finred",
    "nc_sa_research_only",
)

NER_ENTITY_TYPES = (
    "ORG",
    "TICKER",
    "FIN_INST",
    "MONEY",
    "PERCENT",
    "DATE_PERIOD",
    "REGULATOR",
    "PRODUCT_SECTOR",
    "GEO_MARKET",
    "STAGE_COND",
)

NER_BIO_LABELS = ["O"] + [
    f"{prefix}-{etype}" for etype in NER_ENTITY_TYPES for prefix in ("B", "I")
]

REL_LABELS = (
    "no_relation",
    "acquires",
    "acquisition_approved_for",
    "acquisition_denied_or_cancelled",
    "subsidiary_or_owned_by",
    "leadership_role",
    "penalized_or_ordered_by",
    "contract_awarded_to",
    "price_or_product_for",
    "reported_metric_for",
    "located_or_listed_at",
)

IMPACT_IMPLICATIONS = (
    "positive_implication",
    "negative_implication",
    "mixed_implication",
    "neutral_implication",
    "insufficient_evidence",
)

MATERIALITY_LEVELS = ("high", "medium", "low", "unclear")

EVENT_MATURITY_LEVELS = (
    "rumor_speculation",
    "proposed",
    "pending_approval",
    "confirmed_active",
    "completed",
    "denied_cancelled",
    "unclear",
)

UNCERTAINTY_LEVELS = ("low", "moderate", "high")

STAKEHOLDER_GROUPS = (
    "shareholders",
    "customers",
    "employees",
    "creditors_lenders",
    "regulators",
    "counterparties_suppliers",
)

BENCHMARK_SLICES = (
    "unseen_issuer",
    "multi_event",
    "contradictory_or_rumor",
    "numerical_reasoning",
    "misleading_headline",
    "incomplete_or_failed_pdf",
    "long_report",
    "unanswerable",
    "non_financial",
    "prompt_injection",
)

SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
