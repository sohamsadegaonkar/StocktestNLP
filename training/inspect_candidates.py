"""Inspect extracted NSE candidate filings by bucket and issuer."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

CANDIDATES = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Total candidates: {len(rows)}, unique issuers: {len({r['issuer_symbol'] for r in rows})}")
    print("\n=== PRICING CANDIDATES ===")
    for r in [x for x in rows if x["bucket"] == "pricing"]:
        txt = " ".join(" ".join(r["pages"]).split())[:220]
        print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {r['nse_desc']} | {r['headline'][:70]} :: {txt}")

    print("\n=== FAILED / SCANNED PDFS ===")
    for r in [x for x in rows if x["alnum_chars"] < 30 or r["extract_error"]]:
        print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | alnum={r['alnum_chars']} | err={r['extract_error']} | {r['headline'][:90]}")

    for b in ["rumor_or_clarification", "earnings", "corporate_action", "deal", "governance", "regulatory", "operations", "routine"]:
        print(f"\n=== {b.upper()} (first 5) ===")
        for r in [x for x in rows if x["bucket"] == b][:5]:
            txt = " ".join(" ".join(r["pages"]).split())[:220]
            print(f"{r['doc_id']} | {r['issuer_symbol']} | {r['date']} | {r['nse_desc']} :: {txt}")

    print("\nTop 25 issuers:", Counter(r["issuer_symbol"] for r in rows).most_common(25))


if __name__ == "__main__":
    main()
