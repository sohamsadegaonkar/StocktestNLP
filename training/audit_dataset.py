"""Audit curated dataset records across all labels, stages, and edge cases."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

CURATED_PATH = Path(__file__).resolve().parent / "data" / "nse_announcements.jsonl"
CANDIDATES_PATH = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"


def main() -> None:
    rows = [json.loads(ln) for ln in CURATED_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_lbl = defaultdict(list)
    for r in rows:
        by_lbl[r["label"]].append(r)

    for lbl, items in sorted(by_lbl.items()):
        print(f"\n=== {lbl} ({len(items)}) ===")
        for x in items[:3] + items[-2:]:
            txt = x["text"].replace("\n", " ")[:170]
            print(f"  {x['doc_id']} | {x['issuer_symbol']} | {x['date']} | stage={x['stage']} | labels={x['labels']} | insuff={x['insufficiency_type']} :: {txt}")

    print("\n=== RUMOR DENIAL ITEMS ===")
    for r in rows:
        if r["insufficiency_type"] == "rumor_denial" or r["stage"] == "Denied":
            print(f"  {r['doc_id']} | {r['issuer_symbol']} | {r['label']} | stage={r['stage']} | neg={r['preserves']['negations']} :: {r['text'][:160]}")

    # Check why specific rumor filings were skipped
    curated_ids = {r["doc_id"] for r in rows}
    cands = {r["doc_id"]: r for r in (json.loads(ln) for ln in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if ln.strip())}
    for cid in ["nse:106287871", "nse:106323523", "nse:106501962", "nse:106586607", "nse:106552443", "nse:106646940"]:
        print(f"\nCheck {cid}: in_curated={cid in curated_ids}, bucket={cands.get(cid, {}).get('bucket')}")


if __name__ == "__main__":
    main()
