"""Source-backed paraphrases and explicitly marked synthetic replay packets."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    ticker: str
    pages: tuple[str, ...]
    title: str
    source: str = "Local import"
    date: str = ""
    url: str = ""
    origin: str = "import"

    published_at: str = ""
    retrieved_at: str = ""
    source_scope: str = ""

    def record(self) -> dict:
        return asdict(self)


ACQUISITION = Document(
    "INFY",
    (
        "Infosys announced a definitive agreement to acquire in-tech, a German engineering R&D business.",
        "The announced consideration was up to EUR 450 million, including earn-outs, "
        "and was subject to customary closing adjustments.",
    ),
    "Agreement to acquire in-tech",
    "Infosys investor filing",
    "2024-04-18",
    "https://www.infosys.com/investors/documents/outcome-board-meeting-18apr2024.pdf",
    "historical",
)
EARNINGS = Document(
    "INFY",
    (
        "Infosys reported fiscal first-quarter results. It reported revenue of USD 4,714 million "
        "and 3.6% sequential growth in constant currency. The operating margin was 21.1%. "
        "Management issued annual revenue growth guidance of 3% to 4%.",
    ),
    "Q1 FY25 results and outlook",
    "Infosys investor relations",
    "2024-07-18",
    "https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q1.html",
    "historical",
)
DEMERGER = Document(
    "TATAMOTORS",
    (
        "The Tata Motors board approved a demerger scheme separating commercial vehicles "
        "from passenger vehicles. The transaction remained subject to shareholder, creditor "
        "and regulatory approvals.",
    ),
    "Commercial vehicles demerger scheme",
    "Tata Motors investor relations",
    "2024-08-01",
    "https://www.tatamotors.com/press-releases/demerger-of-cv-business-undertaking-of-tata-motors-ltd-into-a-separate-listed-company/",
    "historical",
)
PENALTY = Document(
    "HDFCBANK",
    (
        "The Reserve Bank of India imposed a monetary penalty of ₹1 crore on HDFC Bank "
        "for non-compliance with certain regulatory directions.",
    ),
    "RBI monetary penalty disclosure",
    "Reserve Bank of India",
    "2024-09-10",
    "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=58673",
    "historical",
)
ROUTINE = Document(
    "RELIANCE",
    ("The company sent instructions for remote voting before its annual general meeting.",),
    "AGM remote voting instructions",
    "Synthetic scenario",
    "2024-09-11",
    origin="synthetic",
)
UNCLEAR = Document(
    "TCS",
    ("An update may be issued in due course. Please refer to the earlier communication.",),
    "Further clarification pending",
    "Synthetic scenario",
    "2024-09-12",
    origin="synthetic",
)

SEED = (ACQUISITION, DEMERGER, EARNINGS)
REPLAY = (PENALTY, PENALTY, ROUTINE, UNCLEAR)
CASE_STUDY = (
    (ACQUISITION, "deal", "Agreement signed"),
    (EARNINGS, "earnings", "Reported"),
    (DEMERGER, "deal", "Subject to approvals"),
    (PENALTY, "regulatory", "Penalty imposed"),
)
