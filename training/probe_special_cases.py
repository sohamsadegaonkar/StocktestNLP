"""Probe multi-event, rumor-denial, scheduled-vs-reported, and pricing filings in nse_candidates.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
import re

from training.inspect_cleaned import clean_excerpt, is_garbled

CANDIDATES = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"


def full_clean(pages: list[str]) -> str:
    p1 = clean_excerpt(pages[:1], max_chars=900)
    if re.search(r"enclose\s+herewith|enclosed\s+herewith|attached\s+herewith", p1, re.I) and len(pages) > 1 and len(p1) < 380:
        p2 = clean_excerpt(pages[1:3], max_chars=900)
        if len(p2) > len(p1):
            return p2
    return clean_excerpt(pages[:3], max_chars=900)


def main() -> None:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("=== RUMOR DENIALS / VERIFICATIONS (inspecting denials vs vague) ===")
    for r in rows:
        if r["bucket"] in ("rumor_reg30", "rumor_or_clarification"):
            body = full_clean(r["pages"])
            if re.search(r"\b(den(?:y|ied|ies)|baseless|incorrect|factually\s+incorrect|no\s+such|not\s+true|unaware|not\s+aware|no\s+undisclosed|unable\s+to\s+comment|speculative)\b", body, re.I):
                print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {r['nse_desc']}")
                print(f"  {body[:300]}")

    print("\n=== SCHEDULED BOARD MEETINGS (board_intimation_sched) ===")
    for r in rows:
        if r["bucket"] == "board_intimation_sched":
            body = full_clean(r["pages"])
            print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {r['headline'][:90]}")
            print(f"  {body[:250]}")

    print("\n=== MIXED OUTCOME OF BOARD MEETING (results + dividend?) ===")
    cnt = 0
    for r in rows:
        if r["bucket"] == "earnings":
            body = full_clean(r["pages"])
            if re.search(r"\bdividend\b", body, re.I) and re.search(r"\b(financial\s+results|net\s+profit|revenue)\b", body, re.I):
                print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {body[:260]}")
                cnt += 1
                if cnt >= 8:
                    break


if __name__ == "__main__":
    main()
