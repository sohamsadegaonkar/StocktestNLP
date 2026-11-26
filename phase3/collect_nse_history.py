"""Collect 4-5 years (2022-2026) of official NSE corporate announcements with provenance & deduplication."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import time
from typing import Any

import httpx

RAW_CACHE_DIR = Path("phase3/.cache/nse_history")
P1_CACHE_DIR = Path("training/.cache")
OUT_EVENTS = Path("phase3/.cache/nse_events_pool.jsonl")
MANIFEST_PATH = Path("phase3/data/nse_collection_manifest.json")

IST = timezone(timedelta(hours=5, minutes=30))

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}

QUARTERS = [
    ("2022Q1", "01-01-2022", "31-03-2022"),
    ("2022Q2", "01-04-2022", "30-06-2022"),
    ("2022Q3", "01-07-2022", "30-09-2022"),
    ("2022Q4", "01-10-2022", "31-12-2022"),
    ("2023Q1", "01-01-2023", "31-03-2023"),
    ("2023Q2", "01-04-2023", "30-06-2023"),
    ("2023Q3", "01-07-2023", "30-09-2023"),
    ("2023Q4", "01-10-2023", "31-12-2023"),
    ("2024Q1", "01-01-2024", "31-03-2024"),
    ("2024Q2", "01-04-2024", "30-06-2024"),
    ("2024Q3", "01-07-2024", "30-09-2024"),
    ("2024Q4", "01-10-2024", "31-12-2024"),
    ("2025Q1", "01-01-2025", "31-03-2025"),
    ("2025Q2", "01-04-2025", "30-06-2025"),
]

# Boilerplate noise announcements to exclude unless accompanied by material event terms
BOILERPLATE_RE = re.compile(
    r"(?:Loss\s+of\s+Share\s+Certificate|Closure\s+of\s+Trading\s+Window|Trading\s+Window|"
    r"Certificate\s+under\s+Regulation\s+74\(5\)|Newspaper\s+(?:Advertisement|Publication)|"
    r"Copy\s+of\s+Newspaper|Shareholding\s+pattern|Reconciliation\s+of\s+Share\s+Capital|"
    r"Statement\s+of\s+Investor\s+Complaints|Notice\s+of\s+(?:Postal\s+Ballot|Board\s+Meeting\s+only)|"
    r"Annual\s+Secretarial\s+Compliance|Corrigendum\s+to\s+Notice|"
    r"Analysts/Institutional\s+Investor\s+Meet|Earnings\s+Call\s+(?:Invite|Transcript)|"
    r"Audio/Video\s+recording|Transcript\s+of\s+Earnings\s+Call)",
    re.I,
)

# Material event family rules covering Section 3 target event taxonomy
EVENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("rumor_denial", re.compile(r"(?:Rumour\s+Verification|News\s+Verification|Spurt\s+in\s+(?:Volume|Price)|Price\s+movement|Clarification\s+(?:on|regarding)\s+news)", re.I)),
    ("deal_ma", re.compile(r"(?:\bAcquisition\b|Amalgamation/Merger|\bMerger\b|\bDemerger\b|Scheme\s+of\s+Arrangement|Takeover|Slump\s+Sale|Joint\s+Venture|Stake\s+(?:Acquisition|Sale|Divestment))", re.I)),
    ("earnings", re.compile(r"(?:Outcome\s+of\s+Board\s+Meeting.*(?:Financial\s+Results|Unaudited|Audited)|Integrated\s+Filing-\s*Financial|Financial\s+Results|Q[1-4]\s*FY\d{2,4}\s+Results)", re.I)),
    ("corporate_action", re.compile(r"(?:\bBuyback\b|\bBuy-back\b|\bDividend\b|\bBonus\s+Issue\b|\bStock\s+Split\b|Sub-division\s+of\s+Equity|Rights\s+Issue|\bRecord\s+Date\b)", re.I)),
    ("regulatory", re.compile(r"(?:Action\(s\)\s+taken|Action\(s\)\s+initiated|Pendency\s+of\s+Litigation|\bPenalty\b|\bFine\b|Show\s+Cause\s+Notice|\bSEBI\b\s+Order|\bUSFDA\b|\bGST\b\s+(?:Order|Demand)|Tax\s+Demand)", re.I)),
    ("credit_debt", re.compile(r"(?:Credit\s+Rating|CARE|CRISIL|ICRA|India\s+Ratings|Brickwork|Allotment\s+of\s+(?:NCDs|Debentures)|Qualified\s+Institutional\s+Placement|\bQIP\b|Debt\s+Restructuring|Fund\s+Raising.*(?:Debentures|Debt|Bonds))", re.I)),
    ("operations", re.compile(r"(?:Bagging/Receiving\s+of\s+orders|Letter\s+of\s+Award|\bOrder\s+Win\b|Work\s+Order|Contract\s+(?:Award|Bagged|Received)|Commercial\s+Production|Capacity\s+Addition|Plant\s+(?:Outage|Shutdown|Fire|Strike))", re.I)),
    ("pricing_product", re.compile(r"(?:Product\s+Launch|Commercial\s+Launch|Price\s+(?:Hike|Increase|Revision|Cut|Reduction)|Tariff\s+Revision|New\s+Drug\s+Approval|\bANDA\b\s+Approval)", re.I)),
    ("governance", re.compile(r"(?:Change\s+in\s+(?:Management|Directorate|Auditors)|Appointment\s+of\s+(?:Managing\s+Director|CEO|CFO|Director|Auditor)|Resignation\s+of\s+(?:Managing\s+Director|CEO|CFO|Director|Auditor))", re.I)),
]


def parse_nse_timestamp(raw_dt: str) -> tuple[str | None, str | None, str]:
    """Parse DD-Mon-YYYY HH:MM:SS (IST) -> (iso_ist, date_str, timing_bucket)."""
    s = (raw_dt or "").strip()
    if not s:
        return None, None, "unknown"
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=IST)
            if fmt == "%d-%b-%Y":
                return dt.isoformat(), dt.strftime("%Y-%m-%d"), "unknown"
            hhmm = dt.hour * 100 + dt.minute
            if hhmm < 915:
                bucket = "pre_market"
            elif hhmm <= 1530:
                bucket = "intraday"
            else:
                bucket = "post_market"
            return dt.isoformat(), dt.strftime("%Y-%m-%d"), bucket
        except ValueError:
            continue
    return None, None, "unknown"


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def fetch_quarters() -> list[dict[str, Any]]:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    provenance = []
    client = httpx.Client(headers=NSE_HEADERS, timeout=45, follow_redirects=True)
    cookies_ready = False

    for qname, fdate, tdate in QUARTERS:
        cpath = RAW_CACHE_DIR / f"nse_{qname}.json"
        url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={fdate}&to_date={tdate}"
        if cpath.is_file() and cpath.stat().st_size > 1000:
            rows = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            if not cookies_ready:
                client.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements")
                cookies_ready = True
            rows = []
            for attempt in range(3):
                try:
                    resp = client.get(url)
                    if resp.status_code in (401, 403, 301):
                        client.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements")
                        time.sleep(1.0)
                        resp = client.get(url)
                    resp.raise_for_status()
                    rows = resp.json()
                    break
                except Exception as exc:
                    print(f"  Retry {attempt+1} for {qname}: {exc}", flush=True)
                    time.sleep(2.0 * (attempt + 1))
            if rows:
                cpath.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.5)
        if cpath.is_file():
            provenance.append({"slice": qname, "from": fdate, "to": tdate, "url": url, "count": len(rows), "sha256": sha256_file(cpath)})
        print(f"  {qname} ({fdate}..{tdate}): {len(rows)} raw announcements", flush=True)
    client.close()

    # Include Phase 1 cached 2025H2 and 2026 raw dumps
    for name, fname in (("2025H2", "nse_2025h2_raw.json"), ("2026", "nse_2026_raw.json")):
        p = P1_CACHE_DIR / fname
        if p.is_file():
            rows = json.loads(p.read_text(encoding="utf-8"))
            provenance.append({"slice": name, "path": str(p), "count": len(rows), "sha256": sha256_file(p)})
            print(f"  {name} (reused cache): {len(rows)} raw announcements", flush=True)
    return provenance


def classify_material_event(desc: str, headline: str) -> str | None:
    text = f"{desc} | {headline}".strip()
    if not text or len(text) < 12:
        return None
    if BOILERPLATE_RE.search(text) and not re.search(r"(?:Financial\s+Results|Acquisition|Merger|Buyback|Dividend|Penalty|Order\s+Win|Credit\s+Rating)", text, re.I):
        return None
    for family, pat in EVENT_RULES:
        if pat.search(text):
            return family
    if re.search(r"Press\s+Release|Media\s+Release", desc, re.I) and re.search(r"(?:launch|award|order|result|acquir|partner|approv|plant|expansion|contract|growth|revenue|profit)", headline, re.I):
        return "operations"
    return None


def main() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=== Step 1: Collecting 2022-2026 Official NSE Corporate Announcements ===", flush=True)
    prov = fetch_quarters()

    all_paths = sorted(RAW_CACHE_DIR.glob("nse_202*.json")) + [
        p for p in (P1_CACHE_DIR / "nse_2025h2_raw.json", P1_CACHE_DIR / "nse_2026_raw.json") if p.is_file()
    ]

    # Pass 1: deduplicate & filter to material announcements
    seen_seq: set[str] = set()
    seen_text_day: set[str] = set()
    material: list[dict[str, Any]] = []
    raw_total = 0

    for path in all_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        raw_total += len(rows)
        for r in rows:
            seq_id = str(r.get("seq_id") or "").strip()
            sym = (r.get("symbol") or "").strip().upper()
            if not seq_id or not sym or seq_id in seen_seq:
                continue
            desc = (r.get("desc") or "").strip()
            headline = (r.get("attchmntText") or "").strip()
            family = classify_material_event(desc, headline)
            if family is None:
                continue
            iso_ist, date_str, timing_bucket = parse_nse_timestamp(r.get("an_dt") or "")
            if not date_str or timing_bucket == "unknown":
                # Keep conservative track only for valid IST timestamps; skip missing timestamps
                continue
            norm_key = f"{sym}|{date_str}|{sha256(f'{desc}|{headline[:180]}'.lower().encode('utf-8')).hexdigest()[:16]}"
            if norm_key in seen_text_day:
                continue
            seen_seq.add(seq_id)
            seen_text_day.add(norm_key)

            material.append({
                "event_id": f"nse:{seq_id}",
                "seq_id": seq_id,
                "symbol": sym,
                "isin": (r.get("sm_isin") or "").strip(),
                "issuer_name": (r.get("sm_name") or sym).strip(),
                "published_ts_ist": iso_ist,
                "published_date": date_str,
                "timing_bucket": timing_bucket,
                "event_family_hint": family,
                "desc": desc,
                "headline": headline,
                "text": f"{desc}. {headline}".strip(),
                "source_url": (r.get("attchmntFile") or "").strip(),
                "has_xbrl": bool(r.get("hasXbrl")),
            })

    # Rank symbols by material disclosure frequency so we focus market downloads on top 320 liquid active issuers
    # while guaranteeing broad multi-year coverage (target 15,000-25,000 events cleanly)
    sym_counts = Counter(ev["symbol"] for ev in material)
    top_symbols = {s for s, _ in sym_counts.most_common(320)}
    # Also ensure key Phase 1/2 large-cap symbols are included even if lower count
    for core_sym in ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK", "TATAMOTORS", "TMPV", "SUNPHARMA", "DRREDDY", "TITAN", "BAJFINANCE", "MARUTI", "ASIANPAINT", "WIPRO", "HCLTECH"):
        if core_sym in sym_counts:
            top_symbols.add(core_sym)

    curated = [ev for ev in material if ev["symbol"] in top_symbols]
    curated.sort(key=lambda x: (x["published_ts_ist"], x["seq_id"]))

    # Cap total pool at 28,000 stratified evenly across time if larger, preserving chronological density
    if len(curated) > 28000:
        step = len(curated) / 28000.0
        curated = [curated[int(i * step)] for i in range(28000)]

    OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_EVENTS.open("w", encoding="utf-8") as f:
        for ev in curated:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    fam_counts = Counter(ev["event_family_hint"] for ev in curated)
    yr_counts = Counter(ev["published_date"][:4] for ev in curated)
    manifest = {
        "raw_announcements_scanned": raw_total,
        "material_deduplicated_total": len(material),
        "selected_pool_events": len(curated),
        "selected_unique_symbols": len({ev["symbol"] for ev in curated}),
        "selected_unique_isins": len({ev["isin"] for ev in curated if ev["isin"]}),
        "date_min": curated[0]["published_date"] if curated else None,
        "date_max": curated[-1]["published_date"] if curated else None,
        "by_year": dict(sorted(yr_counts.items())),
        "by_event_family": dict(fam_counts),
        "by_timing_bucket": dict(Counter(ev["timing_bucket"] for ev in curated)),
        "raw_slices_provenance": prov,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(curated)} curated NSE events ({manifest['date_min']}..{manifest['date_max']}) across {manifest['selected_unique_symbols']} issuers.", flush=True)
    print("By year:", manifest["by_year"], flush=True)
    print("By family:", manifest["by_event_family"], flush=True)


if __name__ == "__main__":
    main()
