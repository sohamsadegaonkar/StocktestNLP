"""Inspect full body text (pages 1-3) for pricing, product_launch, rumour_reg30, and mixed filings."""
from __future__ import annotations

import json
from pathlib import Path
import re

from training.inspect_cleaned import clean_excerpt, is_garbled

CANDIDATES = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    for bucket in ("pricing", "product_launch", "rumour_reg30"):
        print(f"\n================ {bucket.upper()} ================")
        for r in [x for x in rows if x["bucket"] == bucket][:35]:
            # Check page 2 if page 1 is short covering letter
            p1 = clean_excerpt(r["pages"][:1], max_chars=700)
            if ("enclose" in p1.lower() or "attached" in p1.lower()) and len(r["pages"]) > 1 and len(p1) < 350:
                body = clean_excerpt(r["pages"][1:3], max_chars=700) or p1
            else:
                body = clean_excerpt(r["pages"][:3], max_chars=700)
            print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {r['nse_desc']}")
            print(f"  HEAD: {r['headline'][:120]}")
            print(f"  BODY: {body[:350]}")


if __name__ == "__main__":
    main()
