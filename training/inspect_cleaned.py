"""Test cleaning boilerplate from real NSE PDF extractions and inspect per-bucket samples."""
from __future__ import annotations

import json
from pathlib import Path
import re

CANDIDATES = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"

HEADER_NOISE = re.compile(
    r"(?:BSE\s+Limited|National\s+Stock\s+Exchange|Phiroze\s+Jeejeebhoy|Exchange\s+Plaza|"
    r"Bandra[- ]Kurla\s+Complex|Dalal\s+Street|Scrip\s+Code|Symbol\s*:|ISIN\s*:|"
    r"Regd\.?\s*Office|Corporate\s+Office|CIN\s*:|Tel\s*(?:No)?\s*:|Fax\s*(?:No)?\s*:|"
    r"E-?mail\s*:|Website\s*:|Dear\s+Sir\s*(?:/\s*Madam)?|Yours\s+faithfully|Yours\s+truly|"
    r"For\s+and\s+on\s+behalf\s+of|Company\s+Secretary|Compliance\s+Officer|Digitally\s+signed|"
    r"Thanking\s+you|Encl\s*:?\s*As\s+above)",
    re.I,
)


def is_garbled(text: str) -> bool:
    if not text or sum(c.isalnum() for c in text) < 30:
        return True
    ascii_letters = sum("a" <= c.lower() <= "z" for c in text)
    printable = sum(c.isprintable() for c in text)
    if printable == 0:
        return True
    # Legacy Hindi/Devanagari font mapping or glyph garbage has low ASCII letter ratio or many ^ ~ | symbols
    weird = sum(c in "^~`|\\{}<>" for c in text)
    return (ascii_letters / max(1, len(text))) < 0.35 or (weird / max(1, len(text))) > 0.03


def clean_excerpt(pages: list[str], max_chars: int = 950) -> str:
    raw = "\n".join(pages[:2])
    if is_garbled(raw):
        return ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
    kept: list[str] = []
    started = False
    for ln in lines:
        if not ln:
            continue
        # Start keeping at Subject / Sub / Press Release / Pursuant to / We wish / Ref / body sentence
        if not started:
            if re.match(r"^(?:Sub(?:ject)?\s*:|Ref\s*:|Press\s+Release|Media\s+Release|Pursuant\s+to|We\s+wish|In\s+terms\s+of|With\s+reference|This\s+is\s+to\s+inform|Intimation)", ln, re.I):
                started = True
            elif len(ln) > 75 and not HEADER_NOISE.search(ln):
                started = True
            else:
                continue
        if HEADER_NOISE.search(ln) and len(ln) < 180:
            if re.search(r"Yours\s+(?:faithfully|truly)|Thanking\s+you|For\s+and\s+on\s+behalf", ln, re.I):
                break
            continue
        kept.append(ln)
        if sum(len(x) + 1 for x in kept) >= max_chars:
            break
    out = " ".join(kept).strip()
    if len(out) < 40:
        # Fallback: take longest non-noise lines
        fallback = [ln for ln in lines if len(ln) > 45 and not HEADER_NOISE.search(ln)]
        out = " ".join(fallback)[:max_chars].strip()
    if len(out) > max_chars:
        cut = out.rfind(". ", 200, max_chars)
        out = out[: cut + 1] if cut > 200 else out[:max_chars].rsplit(" ", 1)[0]
    return out


def main() -> None:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)

    for b in ["pricing", "product_launch", "rumour_reg30", "rumor_or_clarification", "disruption", "board_intimation_sched", "earnings", "corporate_action", "deal", "governance", "regulatory", "operations", "routine"]:
        items = buckets.get(b, [])
        garbled = sum(1 for x in items if is_garbled("\n".join(x["pages"][:2])))
        cleaned = [clean_excerpt(x["pages"]) for x in items]
        nonempty = sum(1 for c in cleaned if len(c) >= 60)
        print(f"\n=== {b} (total={len(items)}, garbled/empty={garbled}, clean>=60={nonempty}) ===")
        shown = 0
        for x, c in zip(items, cleaned):
            if c and shown < 4:
                print(f"  [{x['doc_id']} | {x['issuer_symbol']} | {x['date']} | {x['nse_desc']}]")
                print(f"    HEAD: {x['headline'][:120]}")
                print(f"    BODY: {c[:320]}")
                shown += 1


if __name__ == "__main__":
    main()
