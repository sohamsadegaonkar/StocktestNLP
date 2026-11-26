"""Fetch and inspect official NSE corporate announcement metadata for 2025-2026."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import httpx

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}


def fetch_range(from_date: str, to_date: str, out_name: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / out_name
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = (
        "https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&from_date={from_date}&to_date={to_date}"
    )
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        rows = resp.json()
    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    rows = fetch_range("01-01-2026", "25-11-2026", "nse_2026_raw.json")
    print(f"Total 2026 announcements cached: {len(rows)}")
    patterns = {
        "pricing": r"\b(price\s+(?:hike|increase|revision|cut|reduction)|tariff|ex-showroom|introductory\s+price|discount|mrp)\b",
        "rumour": r"\b(rumour|rumor|news\s+item|clarification.*news|verification|clarification\s+on)\b",
        "buyback_bonus_split": r"\b(buyback|buy-back|bonus\s+share|stock\s+split|sub-division|subdivision|rights\s+issue)\b",
        "outage_fire_resume": r"\b(fire|outage|shutdown|halt|suspended|resumed|restarted|commercial\s+production|commenced\s+production|strike)\b",
        "earnings_results": r"\b(financial\s+results|quarterly\s+results|audited\s+results|unaudited\s+results|q[1-4]\s+fy)\b",
    }
    for tag, pat in patterns.items():
        rx = re.compile(pat, re.I)
        hits = [
            x for x in rows
            if rx.search((x.get("attchmntText") or "") + " " + (x.get("desc") or "") + " " + (x.get("attchmntFile") or ""))
        ]
        syms = {x.get("symbol") for x in hits if x.get("symbol")}
        print(f"  {tag}: {len(hits)} hits across {len(syms)} symbols")
        for h in hits[:3]:
            print(f"    {h.get('symbol')} | {h.get('an_dt')} | {h.get('desc')} | {(h.get('attchmntText') or '')[:110]}")
    top_descs = Counter(x.get("desc") for x in rows).most_common(20)
    print("Top desc values:", top_descs[:10])


if __name__ == "__main__":
    main()
