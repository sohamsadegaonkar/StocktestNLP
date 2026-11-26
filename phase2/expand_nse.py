"""Expand official NSE corpus with 750+ additional real filings (long reports, financials, deals, rumors, etc.)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import re
import logging
import httpx
from pypdf import PdfReader

from training.download_candidates import MAX_PDF_BYTES, parse_nse_date, parse_size_bytes
from training.fetch_nse_index import HEADERS
from training.schema import load_jsonl

logging.getLogger("pypdf").setLevel(logging.ERROR)

PHASE1_DIR = Path("training/data")
PHASE1_CACHE = Path("training/.cache")
RAW_PDF_DIR = Path("phase2/.cache/raw_pdfs")
OUT_PATH = Path("phase2/.cache/nse_expanded_candidates.jsonl")

BUCKETS = [
    ("earnings_results", re.compile(r"(?:Outcome\s+of\s+Board\s+Meeting|Integrated\s+Filing-\s*Financial|financial\s+results)", re.I), 160),
    ("deal_ma", re.compile(r"(?:\bAcquisition\b|Amalgamation/Merger|\bDemerger\b|Scheme\s+of\s+Arrangement|joint\s+venture|stake\s+sale|Takeover)", re.I), 120),
    ("corporate_action", re.compile(r"(?:\bDividend\b|\bBuyback\b|\bBonus\b|Stock\s+split|Rights\s+Issue|\bRecord\s+Date\b)", re.I), 100),
    ("governance", re.compile(r"(?:\bAppointment\b|Change\s+in\s+Management|Resignation\s+of\s+Director|Change\s+in\s+Auditors|\bRetirement\b)", re.I), 100),
    ("regulatory", re.compile(r"(?:Action\(s\)\s+taken|Action\(s\)\s+initiated|Pendency\s+of\s+Litigation|penalty|show\s+cause)", re.I), 100),
    ("operations", re.compile(r"(?:Bagging/Receiving\s+of\s+orders|Commencement\s+of\s+commercial|Capacity\s+addition|Disruption\s+of\s+Operations|Strikes/Lockouts)", re.I), 100),
    ("pricing_launch", re.compile(r"(?:Product\s+launch|price\s+(?:hike|increase|revision|cut)|tariff|discount|ex-showroom|introductory\s+price)", re.I), 70),
    ("rumour_clarify", re.compile(r"(?:Rumour\s+Verification|News\s+Verification|Spurt\s+in\s+Volume|Price\s+movement)", re.I), 80),
    ("routine_sched", re.compile(r"(?:Shareholders\s+meeting|Trading\s+Window|Analysts/Institutional\s+Investor\s+Meet|Board\s+Meeting.*financial\s+results)", re.I), 100),
]


def download_and_extract_long(item: dict) -> dict | None:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    seq_id = str(item["seq_id"])
    url = item["attchmntFile"]
    pdf_path = RAW_PDF_DIR / f"{seq_id}.pdf"
    if not pdf_path.is_file():
        try:
            with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        return None
                    buf = bytearray()
                    for chunk in resp.iter_bytes(32768):
                        buf.extend(chunk)
                        if len(buf) > MAX_PDF_BYTES:
                            return None
                    if len(buf) < 200:
                        return None
                    pdf_path.write_bytes(bytes(buf))
        except Exception:
            return None

    pages: list[str] = []
    err = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_path.read_bytes()))
        for p in reader.pages[:8]:
            pages.append((p.extract_text() or "").strip())
    except Exception as exc:
        err = str(exc)

    joined = "\n".join(p for p in pages if p).strip()
    return {
        "doc_id": f"nse:{seq_id}",
        "seq_id": seq_id,
        "url": url,
        "issuer_symbol": item["symbol"],
        "issuer_name": item.get("sm_name") or item["symbol"],
        "date": parse_nse_date(item.get("an_dt", "")),
        "raw_date": item.get("an_dt", ""),
        "nse_desc": item.get("desc") or "",
        "headline": item.get("attchmntText") or "",
        "bucket": item["_bucket"],
        "page_count": len(pages),
        "pages": pages,
        "extracted_chars": len(joined),
        "alnum_chars": sum(c.isalnum() for c in joined),
        "extract_error": err,
    }


def main() -> None:
    # Exclude ALL Phase 1 doc_ids so Phase 1 splits & Phase 2 stay cleanly linked without duplicating Phase 1 test/cal
    p1_ids = {
        r["doc_id"].replace("nse:", "")
        for fname in ("nse_announcements.jsonl", "train.jsonl", "calibration.jsonl", "test.jsonl")
        for r in load_jsonl(PHASE1_DIR / fname)
    }
    p1_cal_test_issuers = {
        r["issuer_symbol"]
        for fname in ("calibration.jsonl", "test.jsonl")
        for r in load_jsonl(PHASE1_DIR / fname)
    }

    rows = (
        json.loads((PHASE1_CACHE / "nse_2025h2_raw.json").read_text(encoding="utf-8"))
        + json.loads((PHASE1_CACHE / "nse_2026_raw.json").read_text(encoding="utf-8"))
    )
    valid = [
        r for r in rows
        if r.get("seq_id")
        and str(r["seq_id"]) not in p1_ids
        and r.get("symbol")
        and (r.get("attchmntFile") or "").lower().endswith(".pdf")
        and parse_size_bytes(r.get("attFileSize") or r.get("fileSize")) <= MAX_PDF_BYTES
    ]
    valid.sort(key=lambda r: (parse_nse_date(r.get("an_dt", "")), str(r.get("seq_id", ""))))

    seen = set(p1_ids)
    selected: list[dict] = []
    for bname, pat, target in BUCKETS:
        matches = [r for r in valid if pat.search(f"{r.get('desc','')} {r.get('attchmntText','')}")]
        # Spread evenly across the full sorted list for diversity across dates and issuers
        step = max(1, len(matches) // (target * 3))
        per_iss: Counter[str] = Counter()
        added = 0
        for r in matches[::step] + matches:
            sid = str(r["seq_id"])
            if sid in seen:
                continue
            if per_iss[r["symbol"]] >= 2:
                continue
            seen.add(sid)
            per_iss[r["symbol"]] += 1
            selected.append({**r, "_bucket": bname, "_is_p1_heldout_issuer": r["symbol"] in p1_cal_test_issuers})
            added += 1
            if added >= target:
                break
        print(f"Bucket {bname}: queued {added} ({len(per_iss)} issuers)", flush=True)

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(download_and_extract_long, it) for it in selected]
        for idx, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if res is not None:
                out.append(res)
            if idx % 100 == 0:
                print(f"  extracted {idx}/{len(selected)} (ok={len(out)})", flush=True)

    out.sort(key=lambda x: (x["date"], x["seq_id"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(out)} expanded NSE filings across {len({r['issuer_symbol'] for r in out})} issuers to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
