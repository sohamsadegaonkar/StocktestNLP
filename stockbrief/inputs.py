"""Local document parsing with explicit limits and page boundaries."""
from datetime import date
from io import BytesIO
from pathlib import PurePath
from urllib.parse import urlsplit

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .corpus import TICKERS
from .fixtures import Document

MAX_UPLOAD = 5 * 1024 * 1024
MAX_PAGES = 30
MAX_TEXT = 90_000


def read_upload(name: str, payload: bytes) -> tuple[tuple[str, ...], str]:
    if len(payload) > MAX_UPLOAD:
        raise ValueError("Files must be 5 MB or smaller.")
    suffix = PurePath(name).suffix.lower()
    if suffix == ".txt":
        try:
            return (payload.decode("utf-8-sig"),), "text"
        except UnicodeDecodeError as error:
            raise ValueError("Save the TXT file with UTF-8 encoding.") from error
    if suffix != ".pdf" or not payload.startswith(b"%PDF-"):
        raise ValueError("Upload a text-based PDF or a UTF-8 TXT file.")
    try:
        reader = PdfReader(BytesIO(payload))
        if reader.is_encrypted:
            raise ValueError("Open and export an unencrypted copy of the PDF.")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("PDFs must contain 30 pages or fewer.")
        pages = tuple(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, RecursionError, TypeError, KeyError, IndexError, OSError) as error:
        raise ValueError("The PDF could not be read. Try a text-based export.") from error
    if not any(page.strip() for page in pages):
        raise ValueError(
            "No selectable text found (this PDF may be scanned or image-only). "
            "Paste the disclosure text directly in Analyze or export a text-based PDF."
        )
    return pages, "pdf"


def make_document(ticker, title, pages, source="", date_value="", url="", origin="import"):
    if ticker not in TICKERS:
        raise ValueError("Choose a ticker from the watchlist.")
    if not 12 <= sum(len(page.strip()) for page in pages) <= MAX_TEXT:
        raise ValueError("Provide 12 to 90,000 characters of readable disclosure text (current input is too short or incomplete).")
    if date_value:
        try:
            date_value = date.fromisoformat(date_value).isoformat()
        except ValueError as error:
            raise ValueError("Use a valid disclosure date.") from error
    url = url.strip()
    if url:
        try:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or any(c.isspace() for c in url):
                raise ValueError
        except ValueError as error:
            raise ValueError("Use an HTTPS URL for the original source.") from error
    return Document(ticker, pages, title.strip()[:140] or "Untitled disclosure", source.strip()[:100] or "Local import", date_value, url, origin)
