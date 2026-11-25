"""Download genuine scheduled board meeting notices + same-issuer earnings for pricing issuers."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

from training.download_candidates import CANDIDATES_PATH, MAX_PDF_BYTES, download_and_extract, parse_nse_date, parse_size_bytes
from training.fetch_nse_index import fetch_range


def main() -> None:
    existing = [json.loads(line) for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids = {r["seq_id"] for r in existing}

    rows_2026 = fetch_range("01-01-2026", "25-11-2026", "nse_2026_raw.json")
    rows_2025 = fetch_range("01-07-2025", "31-12-2025", "nse_2025h2_raw.json")
    all_rows = rows_2025 + rows_2026
    valid_rows = [
        r for r in all_rows
        if r.get("seq_id")
        and r.get("symbol")
        and (r.get("attchmntFile") or "").lower().endswith(".pdf")
        and parse_size_bytes(r.get("attFileSize") or r.get("fileSize")) <= MAX_PDF_BYTES
    ]
    valid_rows.sort(key=lambda r: (parse_nse_date(r.get("an_dt", "")), str(r.get("seq_id", ""))))

    to_fetch: list[dict] = []

    # 1. Genuine scheduled board meeting intimations (exclude Outcome of Board Meeting)
    sched_per_sym: Counter[str] = Counter()
    sched_added = 0
    for r in valid_rows:
        seq_id = str(r["seq_id"])
        if seq_id in seen_ids:
            continue
        desc = r.get("desc") or ""
        head = r.get("attchmntText") or ""
        if "outcome" in desc.lower() or "outcome" in head.lower():
            continue
        if re.search(r"(?:intimation.*board\s+meeting|board\s+meeting.*to\s+be\s+held|to\s+consider.*financial\s+results)", head, re.I):
            if sched_per_sym[r["symbol"]] >= 1:
                continue
            seen_ids.add(seq_id)
            sched_per_sym[r["symbol"]] += 1
            to_fetch.append({**r, "_bucket": "scheduled_notice"})
            sched_added += 1
            if sched_added >= 30:
                break
    print(f"Queued {sched_added} scheduled board meeting notices")

    # 2. Earnings filings for pricing issuers so mixed pricing+earnings paraphrases are grounded in real paired filings
    pricing_issuers = {"HYUNDAI", "MARUTI", "M&M", "BLUEDART", "ACLGATI", "IRCTC", "ALKEM", "EXICOM", "NATCOPHARM", "BHARATCOAL", "RTNINDIA", "ESCORTS"}
    earn_per_sym: Counter[str] = Counter()
    earn_added = 0
    for r in valid_rows:
        seq_id = str(r["seq_id"])
        if seq_id in seen_ids or r["symbol"] not in pricing_issuers:
            continue
        desc = r.get("desc") or ""
        head = r.get("attchmntText") or ""
        if re.search(r"Outcome\s+of\s+Board\s+Meeting|Press\s+Release|Integrated\s+Filing", desc, re.I) and re.search(r"financial\s+results|quarter\s+ended|q[1-4]", head, re.I):
            if earn_per_sym[r["symbol"]] >= 2:
                continue
            seen_ids.add(seq_id)
            earn_per_sym[r["symbol"]] += 1
            to_fetch.append({**r, "_bucket": "pricing_issuer_earnings"})
            earn_added += 1
    print(f"Queued {earn_added} earnings filings for pricing issuers")

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(download_and_extract, item) for item in to_fetch]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                new_rows.append(res)

    combined = existing + new_rows
    combined.sort(key=lambda x: (x["date"], x["seq_id"]))
    with CANDIDATES_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added {len(new_rows)} filings -> total {len(combined)}")


if __name__ == "__main__":
    main()
