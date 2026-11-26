"""Download additional targeted NSE filings for underrepresented & hard edge-case buckets."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re

from training.download_candidates import CANDIDATES_PATH, MAX_PDF_BYTES, download_and_extract, parse_nse_date, parse_size_bytes
from training.fetch_nse_index import fetch_range

TARGETED_RULES: list[tuple[str, re.Pattern[str], int]] = [
    ("product_launch", re.compile(r"Product\s+launch", re.I), 40),
    ("pricing_text", re.compile(r"\b(price\s+(?:hike|increase|revision|cut|reduction)|ex-showroom|tariff|introductory\s+price|mrp|discount)\b", re.I), 40),
    ("rumour_reg30", re.compile(r"Rumour\s+Verification", re.I), 35),
    ("disruption", re.compile(r"(?:Disruption\s+of\s+Operations|Strikes/Lockouts/Disturbances|Closure\s+of\s+operations)", re.I), 35),
    ("buyback_split_bonus", re.compile(r"^(?:Buyback|Stock\s+split|Bonus|Rights\s+Issue)$", re.I), 35),
    ("demerger_merger", re.compile(r"^(?:Demerger|Amalgamation/Merger|Scheme\s+of\s+Arrangement)$", re.I), 30),
    ("board_intimation_sched", re.compile(r"board\s+meeting.*?(?:consider\s+and\s+approve|unaudited|audited)\s+financial\s+results", re.I), 30),
]


def main() -> None:
    existing = [json.loads(line) for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids = {r["seq_id"] for r in existing}
    print(f"Existing candidates: {len(existing)}")

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
    for bucket, pat, limit in TARGETED_RULES:
        per_issuer: Counter[str] = Counter()
        added = 0
        for r in valid_rows:
            seq_id = str(r["seq_id"])
            if seq_id in seen_ids:
                continue
            desc = r.get("desc") or ""
            head = r.get("attchmntText") or ""
            target_str = desc if bucket in ("buyback_split_bonus", "demerger_merger") else f"{desc} {head}"
            if not pat.search(target_str):
                continue
            if per_issuer[r["symbol"]] >= 3:
                continue
            seen_ids.add(seq_id)
            per_issuer[r["symbol"]] += 1
            to_fetch.append({**r, "_bucket": bucket})
            added += 1
            if added >= limit:
                break
        print(f"Queued {added:3d} for {bucket} ({len(per_issuer)} issuers)", flush=True)

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
    print(f"Added {len(new_rows)} new filings -> total {len(combined)} across {len({r['issuer_symbol'] for r in combined})} issuers")


if __name__ == "__main__":
    main()
