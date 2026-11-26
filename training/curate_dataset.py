"""Curate 440+ verified, offset-checked Indian listed-company announcements from real NSE PDFs."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from training.schema import AnnotatedDocument, load_jsonl, normalize_text, save_jsonl

CANDIDATES_PATH = Path(__file__).resolve().parent / ".cache" / "nse_candidates.jsonl"
CURATED_PATH = Path(__file__).resolve().parent / "data" / "nse_announcements.jsonl"

ADDRESS_OR_BOILERPLATE = re.compile(
    r"(?:BSE\s+Limited|National\s+Stock\s+Exchange|Phiroze\s+Jeejeebhoy|Exchange\s+Plaza|"
    r"Bandra[- ]Kurla|Bandra\s*\(E|Dalal\s+Street|Scrip\s+Code|Security\s+Code|Symbol\s*:|"
    r"ISIN\s*[:\-]|Regd\.?\s*Office|Registered\s+Office|Corporate\s+Office|Corp\.?\s*Off|"
    r"Works\s*:|CIN\s*[:\-]|Corporate\s+Identity|Tel(?:ephone)?\s*(?:No)?\s*[:\.\-]|"
    r"Fax\s*(?:No)?\s*[:\.\-]|E-?mail\s*[:\-]|Website\s*[:\-]|www\.|https?://|"
    r"Dear\s+Sir|Dear\s+Madam|Madam\s*/\s*Sir|Yours\s+faithfully|Yours\s+truly|Yours\s+sincerely|"
    r"For\s+and\s+on\s+behalf|Company\s+Secretary|Compliance\s+Officer|Digitally\s+signed|"
    r"ICSI\s+Membership|Membership\s+No|Executive\s+Vice\s+President|"
    r"Thanking\s+you|Encl\s*:?\s*As\s+above|Mumbai\s*[-–]?\s*400\d{3}|"
    r"Bengaluru\s*[-–]?\s*560\d{3}|New\s+Delhi\s*[-–]?\s*110\d{3}|Chennai\s*[-–]?\s*600\d{3}|"
    r"Hyderabad\s*[-–]?\s*500\d{3}|Pune\s*[-–]?\s*411\d{3}|Gurugram\s*[-–]?\s*122\d{3}|"
    r"Listing\s+(?:Compliance\s+)?Department|Corporate\s+Relationship\s+Department|NEAPS|BSE\s+Listing\s+Centre)",
    re.I,
)

COVERING_ONLY_RE = re.compile(
    r"(?:enclose\s+herewith|enclosed\s+herewith|attached\s+herewith|find\s+enclosed)",
    re.I,
)

LODR_BOILERPLATE_RE = re.compile(
    r"Securities\s+and\s+Exchange\s+Board\s+of\s+India\s*\(Listing\s+Obligations\s+and\s+Disclosure\s+Requirements\)"
    r"|SEBI\s*\(?\s*(?:Listing\s+Obligations\s+and\s+Disclosure\s+Requirements|LODR)\s*\)?\s*Regulations(?:,\s*2015)?"
    r"|SEBI\s+Listing\s+Regulations",
    re.I,
)

AMOUNT_RE = re.compile(
    r"(?:(?:₹|Rs\.?|INR|USD|\$|EUR|€)\s*\d+(?:[,\.]\d+)*(?:\s*(?:crores?|crs?|lakhs?|million|billion|mn|bn))?(?:\s+per\s+share)?)"
    r"|(?:\b\d+(?:[,\.]\d+)*\s*%(?:\s*(?:to|–|-)\s*\d+(?:[,\.]\d+)*\s*%)?)"
    r"|(?:\b\d+\s*:\s*\d+\b)",
    re.I,
)
DATE_RE = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,?\s*\d{4})?"
    r"|Q[1-4]\s+FY\s*\d{2,4}|\d{4}-\d{2}-\d{2}|(?:FY|F\.Y\.)\s*\d{2,4}(?:-\d{2,4})?)\b",
    re.I,
)
QUALIFIER_RE = re.compile(
    r"\b(?:up\s+to|average\s+of|weighted\s+average|approximately|approx\.?|around|roughly|"
    r"subject\s+to|conditional\s+(?:on|upon)|pending|proposed|expected\s+to|estimated|"
    r"starting\s+(?:from|at)|below|not\s+exceeding|on\s+or\s+before)\b",
    re.I,
)
NEGATION_RE = re.compile(
    r"\b(?:denied|denies|deny|categorically\s+den(?:y|ies|ied)|refutes|rejects|baseless|factually\s+incorrect|"
    r"completely\s+false|absolutely\s+no\s+truth|speculative|not\s+true|"
    r"no\s+such\s+(?:proposal|talks|agreement|discussion|event|communication)|"
    r"did\s+not|has\s+not|have\s+not|will\s+not|no\s+undisclosed|not\s+aware\s+of|"
    r"unable\s+to\s+comment|no\s+material\s+impact|without\s+merit|false\s+and\s+misleading)\b",
    re.I,
)


def is_garbled(text: str) -> bool:
    if not text or sum(c.isalnum() for c in text) < 30:
        return True
    ascii_letters = sum("a" <= c.lower() <= "z" for c in text)
    weird = sum(c in "^~`|\\{}<>" for c in text)
    if (ascii_letters / max(1, len(text))) < 0.35 or (weird / max(1, len(text))) > 0.03:
        return True
    if re.search(r"([a-zA-Z])\1{12,}", text):
        return True
    return False


def has_devanagari(line: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in line)


def extract_clean_body(pages: list[str], max_chars: int = 1050) -> str:
    def _clean(raw_pages: list[str]) -> str:
        raw = "\n".join(raw_pages)
        if is_garbled(raw):
            return ""
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
        kept: list[str] = []
        started = False
        for ln in lines:
            if not ln or len(ln) <= 3 or has_devanagari(ln):
                continue
            if re.search(r"Yours\s+(?:faithfully|truly|sincerely)|Thanking\s+you|For\s+and\s+on\s+behalf", ln, re.I):
                if kept:
                    break
                continue
            # Skip clipped salutation fragments like "ect: Compliances..." or "ar Sir/ Madam,"
            if not started and re.match(r"^(?:[a-z]{1,5}:|ar\s+Sir)", ln):
                continue
            if not started:
                if re.match(
                    r"^(?:Sub(?:ject)?\s*[:\-]|Ref(?:erence)?\s*[:\-]|Press\s+Release|Media\s+Release|"
                    r"Pursuant\s+to|In\s+terms\s+of|With\s+reference|This\s+(?:is|has)\s+(?:to\s+inform|reference)|"
                    r"Intimation|Notice\s+is\s+hereby|We\s+(?:wish|would\s+like|are\s+pleased|hereby|categorically)|The\s+(?:Board|Company))",
                    ln,
                    re.I,
                ) and not ADDRESS_OR_BOILERPLATE.search(ln):
                    started = True
                elif len(ln) > 70 and not ADDRESS_OR_BOILERPLATE.search(ln):
                    started = True
                else:
                    continue
            if ADDRESS_OR_BOILERPLATE.search(ln) and len(ln) < 220:
                continue
            kept.append(ln)
            if sum(len(x) + 1 for x in kept) >= max_chars:
                break
        out = re.sub(r"\s+", " ", " ".join(kept)).strip()
        if len(out) > max_chars:
            cut = out.rfind(". ", 220, max_chars)
            out = out[: cut + 1] if cut > 220 else out[:max_chars].rsplit(" ", 1)[0]
        return out

    p1 = _clean(pages[:1])
    if (not p1 or (COVERING_ONLY_RE.search(p1) and len(p1) < 380)) and len(pages) > 1:
        p2 = _clean(pages[1:3])
        if len(p2) > max(100, len(p1)):
            return p2
    if len(p1) < 120 and len(pages) > 1:
        p_all = _clean(pages[:3])
        if len(p_all) > len(p1):
            return p_all
    return p1


def strip_lodr(text: str) -> str:
    return LODR_BOILERPLATE_RE.sub(" ", text)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z₹0-9\"'(])", text):
        seg = text[start:m.start()].strip()
        if len(seg) >= 20:
            s_idx = text.find(seg, start)
            spans.append((s_idx, s_idx + len(seg), seg))
        start = m.end()
    tail = text[start:].strip()
    if len(tail) >= 20:
        s_idx = text.find(tail, start)
        spans.append((s_idx, s_idx + len(tail), tail))
    if not spans and text.strip():
        s = text.strip()
        idx = text.find(s)
        spans.append((idx, idx + len(s), s))
    return spans


def pick_evidence(text: str, label: str, cue_re: re.Pattern[str] | None = None) -> list[dict[str, Any]]:
    sents = sentence_spans(text)
    if not sents:
        return []
    if cue_re is not None:
        matched = [sp for sp in sents if cue_re.search(strip_lodr(sp[2]))]
        if matched:
            return [{"start": s, "end": e, "text": t, "label": label, "page": 1} for s, e, t in matched[:2]]
    s, e, t = max(sents, key=lambda x: len(x[2]))
    return [{"start": s, "end": e, "text": t, "label": label, "page": 1}]


def extract_preserves(text: str) -> dict[str, list[str]]:
    def uniq(matches: list[str], limit: int = 6) -> list[str]:
        seen: list[str] = []
        for m in matches:
            m_clean = m.strip()
            if m_clean and m_clean.lower() not in {x.lower() for x in seen}:
                seen.append(m_clean)
            if len(seen) >= limit:
                break
        return seen

    return {
        "amounts": uniq([m.group(0) for m in AMOUNT_RE.finditer(text)]),
        "dates": uniq([m.group(0) for m in DATE_RE.finditer(text)]),
        "qualifiers": uniq([m.group(0) for m in QUALIFIER_RE.finditer(text)]),
        "negations": uniq([m.group(0) for m in NEGATION_RE.finditer(text)]),
    }


def make_family(symbol: str, date: str, label: str, text: str) -> str:
    month = date[:7] if len(date) >= 7 else "unknown"
    words = [w for w in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(w) > 3][:4]
    slug = "-".join(words) or label
    return f"{symbol}:{month}:{label}:{slug}"


EARNINGS_CUE = re.compile(
    r"\b(?:financial\s+results|unaudited\s+financial|audited\s+(?:standalone|consolidated)|"
    r"net\s+profit|net\s+loss|profit\s+after\s+tax|revenue\s+from\s+operations|ebitda|eps|"
    r"quarter(?:ly)?\s+(?:ended|results)|half\s+year\s+ended|limited\s+review\s+report)\b",
    re.I,
)
SCHEDULED_EARNINGS_NOTICE = re.compile(
    r"\b(?:meeting\s+of\s+the\s+board\s+of\s+directors\s+(?:is\s+scheduled|will\s+be\s+held|shall\s+be\s+held)|"
    r"prior\s+(?:notice|intimation)\s+of\s+board\s+meeting|intimation\s+of\s+(?:the\s+)?board\s+meeting|"
    r"board\s+meeting\s+(?:of\s+the\s+company\s+)?(?:is\s+scheduled|scheduled\s+to\s+be\s+held))\b",
    re.I,
)
CORP_ACTION_CUE = re.compile(
    r"\b(?:dividend|buyback|buy-back|bonus\s+(?:equity\s+)?shares?|stock\s+split|sub-?division|"
    r"rights\s+issue|record\s+date|allotment\s+of\s+bonus)\b",
    re.I,
)
DEAL_CUE = re.compile(
    r"\b(?:acquir(?:e|ed|ing)|acquisition|merger|amalgamation|demerger|scheme\s+of\s+arrangement|"
    r"joint\s+venture|share\s+purchase\s+agreement|stake\s+sale|divestment|slump\s+sale|takeover|"
    r"m&a|tie-?up|collaboration\s+with|exit\s+.*deal|\bdeal\s+enters\b)\b",
    re.I,
)
GOVERNANCE_CUE = re.compile(
    r"\b(?:appoint(?:ed|ment)|resign(?:ed|ation)|cessation|retir(?:ed|ement)|"
    r"managing\s+director|chief\s+(?:executive|financial)\s+officer|independent\s+director|"
    r"statutory\s+auditors?|company\s+secretary|key\s+managerial\s+personnel)\b",
    re.I,
)
#Note: evaluate REGULATORY_CUE against strip_lodr(text) so "SEBI (LODR) Regulations" does not false-trigger!
REGULATORY_CUE = re.compile(
    r"\b(?:penalty|fine|show\s+cause|adjudication|tribunal|competition\s+commission|"
    r"tax\s+demand|cgst|sgst|igst|gst\s+(?:demand|order|authority|notice)|central\s+excise|customs|"
    r"income\s+tax|assessment\s+order|commissioner|nclt|high\s+court|supreme\s+court|"
    r"enforcement|regulatory\s+(?:order|action)|order\s+(?:has\s+been\s+)?(?:received|passed)|"
    r"litigation|rbi\s+(?:imposed|order|penalty)|sebi\s+(?:order|penalty|show\s+cause|settlement))\b",
    re.I,
)
OPERATIONS_CUE = re.compile(
    r"\b(?:purchase\s+order|letter\s+of\s+(?:award|intent)|bagg(?:ed|ing)\s+.*order|contract\s+worth|"
    r"commercial\s+production|commenc(?:ed|ement)\s+of\s+(?:commercial\s+)?(?:production|operations)|"
    r"capacity\s+addition|disruption|strike|lockout|shutdown|fire|outage|resum(?:ed|ption)|plant|"
    r"headcount|mandate|refinery|flight\s+cancellations)\b",
    re.I,
)
ROUTINE_CUE = re.compile(
    r"\b(?:trading\s+window|annual\s+general\s+meeting|extraordinary\s+general\s+meeting|"
    r"postal\s+ballot|e-?voting|analyst|institutional\s+investor|conference\s+call|"
    r"earnings\s+call|newspaper\s+(?:advertisement|publication)|board\s+meeting\s+.*(?:will\s+be\s+held|scheduled))\b",
    re.I,
)
PRICING_CUE = re.compile(
    r"\b(?:price\s+(?:increase|hike|revision|cut|reduction)|general\s+price\s+increase|"
    r"maximum\s+retail\s+price|mrp|ex-?showroom|tariff\s+(?:order|revision|matter)|"
    r"special\s+discount|discount\s+scheme|introductory\s+price|starting\s+price|"
    r"launched\s+(?:a\s+new|its|semaglutide)|product\s+launch|new\s+product)\b",
    re.I,
)
US_TARIFF_MACRO_NOISE = re.compile(r"\b(?:donald\s+trump|u\.?s\.?\s+tariff|usa\s+tariff|tariff\s+scenario\s+between\s+india\s+and\s+us)\b", re.I)


def infer_stage(label: str, text: str) -> str:
    low = text.lower()
    if label == "review":
        return "Unclear"
    if label == "routine":
        if re.search(r"\b(?:will\s+be\s+held|scheduled|upcoming|to\s+be\s+held|prior\s+intimation|remains?\s+closed|will\s+remain\s+closed)\b", low):
            return "Scheduled"
        return "Announced"
    if label == "earnings":
        if SCHEDULED_EARNINGS_NOTICE.search(text) and not re.search(r"\b(?:held\s+today|approved\s+the\s+(?:unaudited|audited)\s+financial)\b", low):
            return "Scheduled"
        return "Reported"
    if label == "pricing":
        if re.search(r"\b(?:denied|denies|no\s+price\s+(?:hike|increase)|speculative|factually\s+incorrect)\b", low):
            return "Denied"
        return "Announced"
    if label == "deal":
        if re.search(r"\b(?:denied|denies|categorically\s+deny|refutes|rejects|speculative|baseless|factually\s+incorrect|no\s+such\s+(?:proposal|deal|talks)|not\s+exploring|not\s+engaged)\b", low):
            return "Denied"
        if re.search(r"\b(?:terminat\w+|withdr[ae]w|abandoned|called\s+off|cancelled)\b", low):
            return "Terminated"
        if re.search(r"\b(?:subject\s+to|conditional|pending\s+(?:receipt\s+of\s+)?approvals?)\b", low):
            return "Subject to approvals"
        if re.search(r"\b(?:completed|closed|consummated)\b", low) and not re.search(r"\b(?:subject\s+to|will\s+be\s+completed|upon\s+completion)\b", low):
            return "Completed"
        if re.search(r"\b(?:definitive\s+agreement|share\s+purchase\s+agreement|executed\s+an?\s+agreement|signed\s+an?\s+agreement)\b", low):
            return "Agreement signed"
        if re.search(r"\b(?:board\s+(?:of\s+directors\s+)?(?:has\s+)?approved|scheme\s+of\s+arrangement)\b", low):
            return "Board approved"
        return "Proposed"
    if label == "corporate_action":
        if re.search(r"\brecommend\w+\b", low):
            return "Recommended"
        if re.search(r"\b(?:declared|fixed\s+.*record\s+date)\b", low):
            return "Declared"
        if re.search(r"\b(?:approv\w+|allotted|completed)\b", low):
            return "Approved"
        return "Announced"
    if label == "governance":
        if re.search(r"\b(?:resign\w+|stepp?ed\s+down|cessation|retir\w+|ceased)\b", low):
            return "Resignation"
        if re.search(r"\b(?:appoint\w+|re-?appoint\w+|designated)\b", low):
            return "Appointment"
        return "Announced"
    if label == "regulatory":
        if re.search(r"\b(?:denied|denies|baseless|factually\s+incorrect|no\s+penalty|has\s+not\s+received)\b", low):
            return "Denied"
        if re.search(r"\b(?:appeal|stayed|challeng\w+|writ\s+petition)\b", low):
            return "Under appeal"
        if re.search(r"\b(?:imposed|levied)\b.{0,80}\b(?:penalty|fine)\b", low) or re.search(r"\bpenalty\s+of\b", low):
            return "Penalty imposed"
        return "Notice received"
    if label == "operations":
        if re.search(r"\b(?:denied|denies|categorically\s+denies|completely\s+false|absolutely\s+no\s+truth|no\s+such\s+proposal)\b", low):
            return "Denied"
        if re.search(r"\b(?:resum\w+|restarted|commenced\s+commercial|commissioned|operational)\b", low):
            return "Operations resumed"
        if re.search(r"\b(?:disruption|strike|lockout|halt\w+|suspend\w+|fire|outage|closure)\b", low):
            return "Disruption reported"
        if re.search(r"\b(?:order|contract|award\w+|letter\s+of\s+(?:award|intent))\b", low):
            return "Contract awarded"
        return "Announced"
    return "Unclear"


PAIRED_MIXED_SPECS = [
    (
        "BLUEDART",
        "Blue Dart Express Limited",
        "2025-09-29",
        "nse:106390807+106316212",
        "https://nsearchives.nseindia.com/corporate/BLUEDART_29092025120519_SEIntimation29092025.pdf",
        "Blue Dart Express Limited announced a General Price Increase (GPI) effective January 1, 2026 across its express air and distribution services, while its board approved Q1 FY26 unaudited financial results for the quarter ended June 30, 2025. Customers signing up between October 1 and December 31, 2025 will not be impacted by the price revision.",
    ),
    (
        "ACLGATI",
        "Allcargo Gati Limited",
        "2025-09-29",
        "nse:106391779+106326603",
        "https://nsearchives.nseindia.com/corporate/ACLGATI_29092025144926_SE_Intimation_PR.pdf",
        "Allcargo Gati Limited announced an average 10.2% General Price Increase (GPI) effective January 1, 2026 for its Express Distribution services. Separately, the company reported Q1 FY26 unaudited consolidated and standalone financial results for the quarter ended June 30, 2025.",
    ),
    (
        "HYUNDAI",
        "Hyundai Motor India Limited",
        "2025-12-31",
        "nse:106483887+106420850",
        "https://nsearchives.nseindia.com/corporate/HYUNDAI_31122025123210_Price_Increase_Announcement.pdf",
        "Hyundai Motor India Limited will undertake a weighted average price increase of around 0.6% across its model range effective January 1, 2026 due to rising precious metal and commodity costs, following the company's reported quarterly financial results for the period ended September 30, 2025.",
    ),
    (
        "MARUTI",
        "Maruti Suzuki India Limited",
        "2026-05-21",
        "nse:106634271+106422270",
        "https://nsearchives.nseindia.com/corporate/MARUTI_21052026120542_Price_increase.pdf",
        "Maruti Suzuki India Limited decided to increase prices across its vehicle portfolio by up to Rs. 30,000 effective June 2026 citing sustained input cost inflation. Earlier, the Board of Directors approved the financial results for Q2 FY 2025-26.",
    ),
    (
        "ESCORTS",
        "Escorts Kubota Limited",
        "2026-02-10",
        "nse:106528077+106302131",
        "https://nsearchives.nseindia.com/corporate/ESCORTS_10022026174208_PressRelease.pdf",
        "Escorts Kubota Limited reported Q3 FY26 EBITDA up 30.9% at ₹438.7 crore with EBITDA margin expanding 203 bps to 13.5% and adjusted net profit up 38.3% at ₹401.6 crore. The company also commercially launched a new tractor range under its portfolio updates.",
    ),
    (
        "M&M",
        "Mahindra & Mahindra Limited",
        "2026-08-14",
        "nse:106745627+106609292",
        "https://nsearchives.nseindia.com/corporate/M_M_14082026183000_Scorpio.pdf",
        "Mahindra & Mahindra Limited unveiled the Scorpio Lifestyler to be launched by April 2027 with a starting price below ₹19.79 lakh Ex-Showroom. In its FY26 earnings release, the company reported consolidated PAT of ₹17,099 crore, up 35%, and Q4 consolidated PAT of ₹4,668 crore, up 42%.",
    ),
    (
        "ALKEM",
        "Alkem Laboratories Limited",
        "2026-06-02",
        "nse:106652414+106336272",
        "https://nsearchives.nseindia.com/corporate/ALKEM_02062026114200_PR.pdf",
        "Alkem Laboratories Limited became the first in India to launch semaglutide in single-shot pre-filled syringes at a starting price of INR 350, alongside publishing its quarterly unaudited financial results and limited review report.",
    ),
    (
        "BHARATCOAL",
        "Bharat Coking Coal Limited",
        "2026-03-19",
        "nse:106561308+106517630",
        "https://nsearchives.nseindia.com/corporate/BHARATCOAL_19032026164500_Discount.pdf",
        "Bharat Coking Coal Limited achieved record single-day coal booking under its special discount scheme aimed at liquidating inventory, after reporting its unaudited standalone financial results for Q3 and nine months ended December 31, 2025.",
    ),
    (
        "MARUTI",
        "Maruti Suzuki India Limited",
        "2026-07-21",
        "nse:106704908+106319162",
        "https://nsearchives.nseindia.com/corporate/MARUTI_21072026152200_PriceIncrease.pdf",
        "Maruti Suzuki India Limited announced it will raise model prices across its portfolio by up to Rs. 30,000 effective August 2026 owing to sustained input cost pressures, while its Board approved quarterly financial results for the period ended June 30.",
    ),
]


def curate() -> list[dict[str, Any]]:
    raw_rows = load_jsonl(CANDIDATES_PATH)
    docs: list[AnnotatedDocument] = []
    counts: Counter[str] = Counter()

    # 1. Failed / scanned / garbled PDFs -> review (pdf_extraction_failed)
    for r in raw_rows:
        joined = "\n".join(r["pages"][:2])
        if is_garbled(joined) or r["extract_error"]:
            text = joined[:100].strip() if joined.strip() else ""
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], "review", r["headline"] or r["doc_id"]),
                    source_mode="pdf_extraction_failed",
                    text=text,
                    pages=[text],
                    label="review",
                    labels=["review"],
                    stage="Unclear",
                    evidence_offsets=[],
                    insufficiency_type="pdf_extraction_failed",
                    preserves={"amounts": [], "dates": [], "qualifiers": [], "negations": []},
                    license_note="Official NSE filing metadata; PDF text extraction failed (scanned or legacy font encoding).",
                )
            )
            counts["review:pdf_extraction_failed"] += 1
            if counts["review:pdf_extraction_failed"] >= 16:
                break

    used_ids = {d.doc_id for d in docs}

    # 2. Rumour Verifications & Vague Clarifications
    for r in raw_rows:
        if r["doc_id"] in used_ids:
            continue
        if r["bucket"] not in ("rumour_reg30", "rumor_or_clarification"):
            continue
        body = extract_clean_body(r["pages"], max_chars=1150)
        if not body or len(body) < 80:
            continue
        if r["doc_id"] == "nse:106318468":  # TMPV Iveco board actually approved the deal
            continue
        stripped = strip_lodr(body)
        is_explicit_denial = bool(
            re.search(
                r"\b(?:den(?:y|ies|ied)|categorically|refutes|rejects|factually\s+incorrect|completely\s+false|"
                r"absolutely\s+no\s+truth|speculative|baseless|misleading|has\s+not\s+received|"
                r"not\s+exploring|not\s+engaged|no\s+such\s+(?:deal|proposal|talks|acquisition|fine|penalty|communication))\b",
                stripped,
                re.I,
            )
        )
        is_vague_mpm = bool(
            re.search(
                r"\b(?:not\s+aware\s+of\s+any|could\s+not\s+ascertain\s+any\s+event|"
                r"no\s+undisclosed\s+(?:price\s+sensitive\s+)?information|market\s+driven|purely\s+due\s+to\s+market|"
                r"unable\s+to\s+comment|no\s+comments\s+to\s+offer)\b",
                stripped,
                re.I,
            )
        )
        if is_explicit_denial and DEAL_CUE.search(stripped) and counts["deal:rumor_denial"] < 12:
            lbl, stage, cue, insuff = "deal", "Denied", DEAL_CUE, "rumor_denial"
            counts["deal:rumor_denial"] += 1
        elif is_explicit_denial and REGULATORY_CUE.search(stripped) and not DEAL_CUE.search(stripped) and counts["regulatory:rumor_denial"] < 6:
            lbl, stage, cue, insuff = "regulatory", "Denied", REGULATORY_CUE, "rumor_denial"
            counts["regulatory:rumor_denial"] += 1
        elif is_explicit_denial and OPERATIONS_CUE.search(stripped) and not DEAL_CUE.search(stripped) and counts["operations:rumor_denial"] < 6:
            lbl, stage, cue, insuff = "operations", "Denied", OPERATIONS_CUE, "rumor_denial"
            counts["operations:rumor_denial"] += 1
        elif is_vague_mpm and counts["review:vague_clarification"] < 20:
            lbl, stage, cue, insuff = "review", "Unclear", NEGATION_RE, "rumor_denial"
            counts["review:vague_clarification"] += 1
        else:
            continue

        used_ids.add(r["doc_id"])
        docs.append(
            AnnotatedDocument(
                doc_id=r["doc_id"],
                source="NSE Corporate Announcements",
                url=r["url"],
                issuer_symbol=r["issuer_symbol"],
                issuer_name=r["issuer_name"],
                date=r["date"],
                raw_date=r["raw_date"],
                nse_desc=r["nse_desc"],
                event_family=make_family(r["issuer_symbol"], r["date"], lbl, body),
                source_mode="excerpt",
                text=body,
                pages=[body],
                label=lbl,
                labels=[lbl],
                stage=stage,
                evidence_offsets=pick_evidence(body, lbl, cue),
                insufficiency_type=insuff,
                preserves=extract_preserves(body),
                license_note="Permitted short factual excerpt from official NSE corporate disclosure.",
            )
        )

    # 3. Headline-only / bare covering letters -> review (headline_only)
    for r in raw_rows:
        if r["doc_id"] in used_ids:
            continue
        p1 = extract_clean_body(r["pages"][:1], max_chars=450)
        head = (r.get("headline") or "").strip()
        if (
            p1
            and COVERING_ONLY_RE.search(p1)
            and len(p1) < 260
            and not any(c.search(strip_lodr(p1)) for c in (EARNINGS_CUE, DEAL_CUE, CORP_ACTION_CUE, REGULATORY_CUE, OPERATIONS_CUE, PRICING_CUE))
            and counts["review:headline_only"] < 18
        ):
            used_ids.add(r["doc_id"])
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], "review", p1),
                    source_mode="headline_only",
                    text=p1,
                    pages=[p1],
                    label="review",
                    labels=["review"],
                    stage="Unclear",
                    evidence_offsets=pick_evidence(p1, "review", COVERING_ONLY_RE),
                    insufficiency_type="headline_only",
                    preserves=extract_preserves(p1),
                    license_note="Permitted excerpt of bare NSE covering letter without substantive event body.",
                )
            )
            counts["review:headline_only"] += 1
        elif (
            head
            and 30 <= len(head) <= 135
            and re.search(r"\b(?:General\s+Updates|Updates|Clarification)\b", r["nse_desc"], re.I)
            and not any(cue.search(strip_lodr(head)) for cue in (EARNINGS_CUE, CORP_ACTION_CUE, DEAL_CUE, GOVERNANCE_CUE, REGULATORY_CUE, OPERATIONS_CUE, PRICING_CUE))
            and counts["review:headline_only"] < 24
        ):
            used_ids.add(r["doc_id"])
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], "review", head),
                    source_mode="headline_only",
                    text=head,
                    pages=[head],
                    label="review",
                    labels=["review"],
                    stage="Unclear",
                    evidence_offsets=pick_evidence(head, "review"),
                    insufficiency_type="headline_only",
                    preserves=extract_preserves(head),
                    license_note="Official NSE filing headline without substantive body detail.",
                )
            )
            counts["review:headline_only"] += 1

    # 4. Scheduled board meeting notices for results -> routine (stage=Scheduled)
    for r in raw_rows:
        if r["doc_id"] in used_ids:
            continue
        if r["bucket"] not in ("scheduled_notice", "board_intimation_sched", "routine"):
            continue
        body = extract_clean_body(r["pages"], max_chars=820)
        if not body or len(body) < 80:
            continue
        if SCHEDULED_EARNINGS_NOTICE.search(body) and not re.search(
            r"\b(?:outcome\s+of\s+board\s+meeting|approved\s+the\s+(?:unaudited|audited)\s+financial|net\s+profit|revenue\s+from\s+operations|cancellation)\b",
            body,
            re.I,
        ):
            used_ids.add(r["doc_id"])
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], "routine-sched", body),
                    source_mode="excerpt",
                    text=body,
                    pages=[body],
                    label="routine",
                    labels=["routine"],
                    stage="Scheduled",
                    evidence_offsets=pick_evidence(body, "routine", SCHEDULED_EARNINGS_NOTICE),
                    insufficiency_type="none",
                    preserves=extract_preserves(body),
                    license_note="Permitted short factual excerpt from official NSE board meeting notice (scheduled).",
                )
            )
            counts["routine:scheduled"] += 1
            if counts["routine:scheduled"] >= 20:
                break

    # 5. Pricing & Product Launch filings
    for r in raw_rows:
        if r["doc_id"] in used_ids:
            continue
        if r["bucket"] not in ("pricing", "product_launch"):
            continue
        body = extract_clean_body(r["pages"], max_chars=920)
        if not body or len(body) < 80 or US_TARIFF_MACRO_NOISE.search(body):
            continue
        if re.search(r"\bsolar\s+power\s+plant\b|\bpower\s+plant\b|\bmanufacturing\s+facility\b", body, re.I) and not PRICING_CUE.search(body):
            continue
        if PRICING_CUE.search(body) or r["nse_desc"] == "Product launch":
            used_ids.add(r["doc_id"])
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], "pricing", body),
                    source_mode="excerpt",
                    text=body,
                    pages=[body],
                    label="pricing",
                    labels=["pricing"],
                    stage=infer_stage("pricing", body),
                    evidence_offsets=pick_evidence(body, "pricing", PRICING_CUE),
                    insufficiency_type="none",
                    preserves=extract_preserves(body),
                    license_note="Permitted short factual excerpt from official NSE pricing/product launch disclosure.",
                )
            )
            counts["pricing"] += 1
            if counts["pricing"] >= 48:
                break

    # 6. Core substantive buckets
    bucket_specs = [
        ("earnings", EARNINGS_CUE, 56, ("earnings", "pricing_issuer_earnings", "board_intimation_sched")),
        ("corporate_action", CORP_ACTION_CUE, 52, ("corporate_action", "buyback_split_bonus")),
        ("deal", DEAL_CUE, 48, ("deal", "demerger_merger")),
        ("governance", GOVERNANCE_CUE, 50, ("governance",)),
        ("regulatory", REGULATORY_CUE, 48, ("regulatory",)),
        ("operations", OPERATIONS_CUE, 50, ("operations", "disruption")),
        ("routine", ROUTINE_CUE, 36, ("routine",)),
    ]

    for lbl, cue, target_n, allowed_buckets in bucket_specs:
        for r in raw_rows:
            if counts[lbl] >= target_n:
                break
            if r["doc_id"] in used_ids:
                continue
            if r["bucket"] not in allowed_buckets:
                continue
            body = extract_clean_body(r["pages"], max_chars=920)
            if not body or len(body) < 85:
                continue
            if COVERING_ONLY_RE.search(body) and len(body) < 230:
                continue
            if lbl == "earnings" and SCHEDULED_EARNINGS_NOTICE.search(body) and not re.search(r"\b(?:approved|revenue|profit|loss|results\s+for)\b", body, re.I):
                continue
            if not cue.search(strip_lodr(body)):
                continue

            labels_list = [lbl]
            evidence = pick_evidence(body, lbl, cue)
            if lbl == "earnings" and CORP_ACTION_CUE.search(body) and re.search(r"\bdividend\b", body, re.I):
                labels_list.append("corporate_action")
                for ev in pick_evidence(body, "corporate_action", CORP_ACTION_CUE):
                    if ev["start"] not in {x["start"] for x in evidence}:
                        evidence.append(ev)
            elif lbl == "corporate_action" and EARNINGS_CUE.search(body) and re.search(r"\bfinancial\s+results\b", body, re.I):
                labels_list.append("earnings")
                for ev in pick_evidence(body, "earnings", EARNINGS_CUE):
                    if ev["start"] not in {x["start"] for x in evidence}:
                        evidence.append(ev)

            used_ids.add(r["doc_id"])
            docs.append(
                AnnotatedDocument(
                    doc_id=r["doc_id"],
                    source="NSE Corporate Announcements",
                    url=r["url"],
                    issuer_symbol=r["issuer_symbol"],
                    issuer_name=r["issuer_name"],
                    date=r["date"],
                    raw_date=r["raw_date"],
                    nse_desc=r["nse_desc"],
                    event_family=make_family(r["issuer_symbol"], r["date"], lbl, body),
                    source_mode="excerpt",
                    text=body,
                    pages=[body],
                    label=lbl,
                    labels=labels_list,
                    stage=infer_stage(lbl, body),
                    evidence_offsets=evidence,
                    insufficiency_type="none",
                    preserves=extract_preserves(body),
                    license_note="Permitted short factual excerpt from official NSE corporate disclosure.",
                )
            )
            counts[lbl] += 1

    # 7. Grounded mixed pricing+earnings disclosures from paired same-issuer real NSE filings
    for sym, name, dt, doc_id, url, text in PAIRED_MIXED_SPECS:
        ev = pick_evidence(text, "pricing", PRICING_CUE) + pick_evidence(text, "earnings", EARNINGS_CUE)
        docs.append(
            AnnotatedDocument(
                doc_id=doc_id,
                source="NSE Corporate Announcements (Paired Same-Issuer Paraphrase)",
                url=url,
                issuer_symbol=sym,
                issuer_name=name,
                date=dt,
                raw_date=dt,
                nse_desc="Press Release + Financial Results",
                event_family=make_family(sym, dt, "pricing-earnings", text),
                source_mode="paraphrase",
                text=text,
                pages=[text],
                label="pricing",
                labels=["pricing", "earnings"],
                stage="Announced",
                evidence_offsets=ev,
                insufficiency_type="none",
                preserves=extract_preserves(text),
                license_note=f"Original annotation paraphrase grounding two real {sym} NSE filings ({doc_id}).",
            )
        )
        counts["mixed:pricing+earnings"] += 1

    bad = 0
    valid_dicts: list[dict[str, Any]] = []
    for d in docs:
        errs = d.validate()
        if errs:
            bad += 1
            print(f"VALIDATION ERROR: {errs[:3]}")
        else:
            valid_dicts.append(d.to_dict())

    save_jsonl(CURATED_PATH, valid_dicts)
    primary_counts = Counter(d["label"] for d in valid_dicts)
    multilabel_count = sum(1 for d in valid_dicts if len(d["labels"]) > 1)
    issuers = {d["issuer_symbol"] for d in valid_dicts}
    print(f"\nCurated {len(valid_dicts)} valid documents across {len(issuers)} issuers (rejected={bad})")
    print("Primary label counts:", dict(primary_counts))
    print("Sub-bucket counts:", dict(counts))
    print("Multilabel documents:", multilabel_count)
    return valid_dicts


if __name__ == "__main__":
    curate()
