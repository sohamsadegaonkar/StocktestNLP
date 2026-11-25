"""Versioned, extractive news brief derived from a saved disclosure.

No network or new model inference. Evidence spans are checked against stored
pages; category-specific research questions are hypotheses, not source claims.
"""
from math import isfinite

VERSION = "stockbrief-intelligence-v1"

# Conditional research questions, not claims about any specific issuer.
PLAYBOOK = {
    "earnings": (
        "Shareholders and analysts",
        "Reported results or guidance may change expectations for profitability and cash flow.",
        ("Compare results with prior periods and published guidance.",
         "Check one-off items, operating margins, cash flow and management outlook."),
    ),
    "pricing": (
        "Customers, shareholders and distributors",
        "If implemented, price changes may affect affordability, demand, revenue per unit and margins.",
        ("Confirm effective dates, affected products, geography and exact price terms.",
         "Watch sales volumes, competitive responses and subsequent margin disclosures."),
    ),
    "deal": (
        "Shareholders, employees and counterparties",
        "If the transaction proceeds, consideration, financing and integration may affect future cash flows and risk.",
        ("Check consideration, funding, approvals and closing conditions.",
         "Look for the next official filing confirming progress, completion or termination."),
    ),
    "corporate_action": (
        "Existing shareholders and prospective investors",
        "If implemented, the action may affect shareholder entitlements, share count or capital allocation.",
        ("Verify approval, record date, ex-date and eligibility from the exchange.",
         "Check the final terms and whether the action has taken effect."),
    ),
    "governance": (
        "Shareholders and management",
        "A leadership or governance change may affect oversight or business continuity; direction is not established.",
        ("Check appointment or departure effective dates and stated reasons.",
         "Watch official succession, board composition and governance disclosures."),
    ),
    "regulatory": (
        "Shareholders, customers and compliance teams",
        "A regulatory development may create compliance costs, restrictions or legal uncertainty.",
        ("Verify the authority, order or notice and whether it is final.",
         "Check appeals, remediation, quantified exposure and follow-up disclosures."),
    ),
    "operations": (
        "Customers, employees, suppliers and shareholders",
        "An operating change may affect delivery, capacity, costs or revenue, depending on scale and duration.",
        ("Verify affected facilities, contracts, volumes and expected duration.",
         "Watch execution milestones, resumption updates and financial guidance."),
    ),
    "routine": (
        "Shareholders and disclosure readers",
        "A routine notice may require an administrative action; no financial effect is established.",
        ("Check meeting, voting or filing deadlines if relevant.",),
    ),
}
NON_ACTIVE = {"Denied", "Terminated", "Unclear"}


def _span(pages, item):
    if not isinstance(item, dict):
        return None
    page, start, end, text = (item.get(key) for key in ("page", "start", "end", "text"))
    if not all(type(value) is int for value in (page, start, end)):
        return None
    if not 1 <= page <= len(pages) or not isinstance(pages[page - 1], str):
        return None
    raw = pages[page - 1]
    if not isinstance(text, str) or not text.strip() or not 0 <= start < end <= len(raw):
        return None
    if raw[start:end] != text:
        return None
    return {"page": page, "start": start, "end": end, "text": text}


def _values(pages, values, evidence):
    valid = []
    for item in values or []:
        if not isinstance(item, dict) or not isinstance(item.get("value"), str) or not item["value"]:
            continue
        start = item.get("start")
        if type(start) is not int:
            continue
        span = _span(pages, {"page": item.get("page"), "start": start,
                            "end": start + len(item["value"]), "text": item["value"]})
        if span and any(q["page"] == span["page"] and q["start"] <= span["start"]
                        and q["end"] >= span["end"] for q in evidence):
            clean = {"value": span["text"], "page": span["page"], "start": span["start"]}
            if clean not in valid:
                valid.append(clean)
    return valid[:12]


def _event(pages, event):
    evidence = []
    for item in event.get("evidence") or []:
        quote = _span(pages, item)
        if quote and quote not in evidence:
            evidence.append(quote)
    stage_quote = _span(pages, event.get("stage_evidence"))
    if stage_quote and stage_quote not in evidence:
        evidence.append(stage_quote)
    category = event.get("category")
    category_quote_count = sum(_span(pages, item) is not None for item in event.get("evidence") or [])
    supported = category in PLAYBOOK and category_quote_count > 0
    stage = event.get("stage", "Unclear") if stage_quote and supported else "Unclear"
    if not isinstance(stage, str):
        stage = "Unclear"
    score = event.get("score")
    score = score if type(score) in (float, int) and isfinite(score) and 0 <= score <= 1 else None
    result = {
        "category": category if supported else "review",
        "category_name": event.get("category_name", category) if supported else "Needs review",
        "stage": stage, "stage_evidence": stage_quote if supported else None,
        "model_score": score,
        "evidence": evidence[:6],
        "figures": _values(pages, event.get("figures"), evidence[:6]) if supported else [],
        "dates": _values(pages, event.get("dates"), evidence[:6]) if supported else [],
        "interpretation": None,
        "potentially_affected": None,
        "monitor": [],
    }
    if not supported:
        result["monitor"] = ["Read the original source and verify the issuer and event before drawing conclusions."]
    elif stage in NON_ACTIVE:
        result["monitor"] = ["Verify the stated status or clarification in the original source and subsequent filings."]
        if stage in {"Denied", "Terminated"}:
            result["interpretation"] = "The model identified a denial or termination; do not treat the underlying event as confirmed."
        else:
            result["interpretation"] = "Event category identified, but its stage is not established by the model."
    else:
        stakeholders, implication, monitor = PLAYBOOK[category]
        result.update(potentially_affected=stakeholders, interpretation=implication, monitor=list(monitor))
    return result


def build_intelligence(entry: dict) -> dict:
    pages = entry["pages"]
    brief = entry["brief"]
    raw_events = brief.get("events") or [brief]
    events = [_event(pages, event) for event in raw_events]
    supported = [event for event in events if event["category"] != "review"]
    return {
        "version": VERSION, "entry_id": entry["id"], "ticker": entry["ticker"],
        "title": entry["title"], "source": entry["source"], "date": entry["date"],
        "url": entry["url"], "origin": entry["origin"],
        "published_at": entry.get("published_at", ""), "retrieved_at": entry.get("retrieved_at", ""),
        "source_scope": entry.get("source_scope", ""),
        "model": brief.get("model"), "model_version": brief.get("model_version"),
        "state": "identified" if supported else "needs_review",
        "events": events,
        "limitations": [
            "Source excerpts match the stored text, not an independently authenticated publisher document.",
            "Event labels and stages are model interpretations; model scores are not calibrated probabilities.",
            "Potential implications and monitoring questions are general category-based hypotheses, not established consequences.",
            "No buy/sell recommendation, sentiment forecast or predicted price is provided.",
            *(
                ["Historical examples contain authored paraphrases, not verbatim publisher quotations."]
                if entry["origin"] == "historical" else
                ["Synthetic replay data is not real news."] if entry["origin"] == "synthetic" else []
            ),
            *(["Only the published RSS excerpt was analyzed; the complete filing may contain additional information."]
              if entry.get("source_scope") == "rss_excerpt" else []),
            *(["The model reached its chunk limit; later text may not have been analyzed."]
              if brief.get("chunk_limit_reached") else []),
        ],
    }
