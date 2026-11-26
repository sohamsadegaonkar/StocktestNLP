"""Download a balanced pool of official NSE corporate announcement PDFs across 40+ issuers."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import logging
from pathlib import Path
import re
import httpx
from pypdf import PdfReader

from training.fetch_nse_index import CACHE_DIR, HEADERS, fetch_range

logging.getLogger("pypdf").setLevel(logging.ERROR)

RAW_PDF_DIR = Path(__file__).resolve().parent / "raw_pdfs"
CANDIDATES_PATH = CACHE_DIR / "nse_candidates.jsonl"
MAX_PDF_BYTES = 2_500_000

PRIORITY_ISSUERS = [
    "INFY", "TCS", "WIPRO", "HCLTECH", "TECHM",
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
    "TATAMOTORS", "TMPV", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "HYUNDAI", "TVSMOTOR",
    "RELIANCE", "BHARTIARTL", "LT", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
    "ITC", "HINDUNILVR", "ASIANPAINT", "TITAN", "TRENT", "DMART",
    "JSWSTEEL", "TATASTEEL", "ULTRACEMCO", "INDIGO", "ADANIPORTS",
    "SIEMENS", "ABB", "BEL", "HAL", "DIXON", "VOLTAS", "BLUESTARCO", "GMRAIRPORT",
]

BUCKETS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "pricing",
        re.compile(
            r"\b(price\s+(?:hike|increase|revision|cut|reduction)|tariff|ex-showroom|"
            r"introductory\s+price|discount|mrp|launch(?:ed|es)?\s+.*(?:price|model|variant|plan)|"
            r"revis(?:ed|es|ing)\s+prices?|rate\s+revision)\b",
            re.I,
        ),
        60,
    ),
    (
        "rumor_or_clarification",
        re.compile(
            r"(?:Rumour\s+Verification|News\s+Verification|Spurt\s+in\s+Volume|Price\s+movement|"
            r"clarification\s+on\s+news|clarification\s+regarding\s+media)",
            re.I,
        ),
        65,
    ),
    (
        "earnings",
        re.compile(
            r"(?:Outcome\s+of\s+Board\s+Meeting|Integrated\s+Filing-\s*Financial|Clarification\s*-\s*Financial\s+Results)"
            r"|financial\s+results\s+for\s+the\s+(?:quarter|half|year)",
            re.I,
        ),
        70,
    ),
    (
        "corporate_action",
        re.compile(
            r"(?:\bDividend\b|\bBuyback\b|\bBonus\b|\bRecord\s+Date\b|stock\s+split|sub-division|rights\s+issue)",
            re.I,
        ),
        65,
    ),
    (
        "deal",
        re.compile(
            r"(?:\bAcquisition\b|Amalgamation/Merger|\bDemerger\b|joint\s+venture|stake\s+sale|"
            r"share\s+purchase\s+agreement|Disclosure\s+under\s+SEBI\s+Takeover)",
            re.I,
        ),
        65,
    ),
    (
        "governance",
        re.compile(
            r"(?:\bAppointment\b|Change\s+in\s+Management|Resignation\s+of\s+Director|Change\s+in\s+Director|"
            r"Change\s+in\s+Auditors|\bCessation\b)",
            re.I,
        ),
        65,
    ),
    (
        "regulatory",
        re.compile(
            r"(?:Action\(s\)\s+taken\s+or\s+orders\s+passed|Action\(s\)\s+initiated|"
            r"Pendency\s+of\s+Litigation|show\s+cause|monetary\s+penalty|adjudication\s+order)",
            re.I,
        ),
        65,
    ),
    (
        "operations",
        re.compile(
            r"(?:Bagging/Receiving\s+of\s+orders|Commencement\s+of\s+commercial\s+production|"
            r"notice\s+of\s+strike|fire|outage|shutdown|halted|resumed\s+operations|purchase\s+order)",
            re.I,
        ),
        65,
    ),
    (
        "routine",
        re.compile(
            r"(?:Trading\s+Window|Shareholders\s+meeting|Analysts/Institutional\s+Investor\s+Meet|"
            r"Copy\s+of\s+Newspaper\s+Publication|Board\s+Meeting\s+Intimation)",
            re.I,
        ),
        65,
    ),
]


def parse_nse_date(raw: str) -> str:
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{4})", (raw or "").strip())
    if not m:
        return ""
    return f"{m.group(3)}-{months[m.group(2).lower()]}-{m.group(1)}"


def parse_size_bytes(size_str: str | None) -> int:
    if not size_str:
        return 0
    m = re.match(r"([\d.]+)\s*(KB|MB|B)", str(size_str).strip(), re.I)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "MB":
        return int(val * 1_000_000)
    if unit == "KB":
        return int(val * 1_000)
    return int(val)


def select_candidates(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()
    priority_set = set(PRIORITY_ISSUERS)

    valid_rows = [
        r for r in rows
        if r.get("seq_id")
        and r.get("symbol")
        and (r.get("attchmntFile") or "").lower().endswith(".pdf")
        and parse_size_bytes(r.get("attFileSize") or r.get("fileSize")) <= MAX_PDF_BYTES
    ]
    valid_rows.sort(key=lambda r: (parse_nse_date(r.get("an_dt", "")), str(r.get("seq_id", ""))))

    for bucket_name, pattern, target_count in BUCKETS:
        matches = [
            r for r in valid_rows
            if pattern.search((r.get("desc") or "") + " " + (r.get("attchmntText") or ""))
        ]
        per_issuer: Counter[str] = Counter()
        priority_matches = [r for r in matches if r["symbol"] in priority_set]
        other_matches = [r for r in matches if r["symbol"] not in priority_set]
        # Stride priority matches across time if large, then fallback to others
        if len(priority_matches) > target_count * 2:
            step = max(1, len(priority_matches) // target_count)
            ordered = priority_matches[::step] + priority_matches + other_matches[::max(1, len(other_matches) // target_count)]
        else:
            ordered = priority_matches + other_matches[::max(1, len(other_matches) // max(1, target_count))]

        bucket_added = 0
        for r in ordered:
            seq_id = str(r["seq_id"])
            if seq_id in seen_ids:
                continue
            sym = r["symbol"]
            cap = 6 if bucket_name == "pricing" else 3
            if per_issuer[sym] >= cap:
                continue
            seen_ids.add(seq_id)
            per_issuer[sym] += 1
            selected.append({**r, "_bucket": bucket_name})
            bucket_added += 1
            if bucket_added >= target_count:
                break
        print(f"Selected {bucket_added:3d} candidates for bucket={bucket_name} ({len(per_issuer)} issuers)", flush=True)
    return selected


def download_and_extract(item: dict) -> dict | None:
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
    extract_error = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_path.read_bytes()))
        for page in reader.pages[:4]:
            pages.append((page.extract_text() or "").strip())
    except Exception as exc:
        extract_error = str(exc)

    joined = "\n".join(p for p in pages if p).strip()
    alnum_chars = sum(c.isalnum() for c in joined)
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
        "alnum_chars": alnum_chars,
        "extract_error": extract_error,
        "pdf_bytes": pdf_path.stat().st_size,
    }


def main() -> None:
    rows_2026 = fetch_range("01-01-2026", "25-11-2026", "nse_2026_raw.json")
    rows_2025 = fetch_range("01-07-2025", "31-12-2025", "nse_2025h2_raw.json")
    all_rows = rows_2025 + rows_2026
    print(f"Loaded {len(all_rows)} NSE metadata rows across 2025-H2 and 2026", flush=True)
    candidates = select_candidates(all_rows)
    print(f"Downloading & extracting {len(candidates)} candidate PDFs...", flush=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_and_extract, item) for item in candidates]
        for idx, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if res is not None:
                results.append(res)
            if idx % 50 == 0:
                print(f"  processed {idx}/{len(candidates)} (ok={len(results)})", flush=True)

    results.sort(key=lambda x: (x["date"], x["seq_id"]))
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_bucket = Counter(r["bucket"] for r in results)
    failed = sum(1 for r in results if r["alnum_chars"] < 30 or r["extract_error"])
    issuers = {r["issuer_symbol"] for r in results}
    print(f"Saved {len(results)} extracted filings across {len(issuers)} issuers to {CANDIDATES_PATH}", flush=True)
    print(f"Failed/scanned PDFs (<30 alnum chars): {failed}", flush=True)
    print("Counts by bucket:", dict(by_bucket), flush=True)


if __name__ == "__main__":
    main()
