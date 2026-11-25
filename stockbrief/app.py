"""Loopback HTTP interface for the local research workspace."""
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .corpus import LABELS, TICKERS
from .display import display_png
from .evaluation import evaluate, host_metrics
from .fixtures import Document
from .inputs import MAX_UPLOAD, make_document, read_upload
from .nlp import MODEL, EventModel
from .store import Inbox

STATIC = Path(__file__).resolve().parent / "static"
MAX_REQUEST = MAX_UPLOAD + 256 * 1024


class LinkRequest(BaseModel):
    connected: StrictBool


class RequestLimit:
    """Reject oversized bodies before multipart parsing, including streamed requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST:
                return await PlainTextResponse("Request exceeds the local upload limit.", 413)(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body"):
                break

        async def buffered():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        return await self.app(scope, buffered, send)


def create_app(path: Path | None = None, transformer=None):
    model = EventModel()
    inbox = Inbox(path or Path(".runtime/stockbrief.sqlite"), model)
    app = FastAPI(title="StockBrief Edge", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def local_request(request: Request, call_next):
        origin = request.headers.get("origin")
        request_url = request.url
        if request.method == "POST":
            if request.headers.get("sec-fetch-site") == "cross-site":
                return PlainTextResponse("Use the local workspace to send changes.", 403)
            if origin and origin != f"{request_url.scheme}://{request.headers.get('host')}":
                return PlainTextResponse("Use the local workspace to send changes.", 403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/inbox")
    def list_inbox():
        return {
            **inbox.state(), "labels": {**LABELS, "review": "Needs review"},
            "tickers": TICKERS, "transformer": transformer.name if transformer else None,
        }

    @app.get("/api/disclosures/{entry_id}")
    def get_disclosure(entry_id: int):
        entry = inbox.document(entry_id)
        if entry is None:
            raise HTTPException(404, "Disclosure not found.")
        return entry

    @app.get("/api/disclosures/{entry_id}/second-opinion")
    def second_opinion(entry_id: int):
        if transformer is None:
            raise HTTPException(404, "The optional local transformer is not loaded.")
        entry = inbox.document(entry_id)
        if entry is None:
            raise HTTPException(404, "Disclosure not found.")
        document = Document(entry["ticker"], tuple(entry["pages"]), entry["title"])
        category = transformer.classify(document)
        return {"category": category, "name": LABELS.get(category, "Needs review"), "model": transformer.name}

    @app.post("/api/analyze")
    def analyze(
        ticker: str = Form(...),
        title: str = Form(""),
        text: str = Form(""),
        source: str = Form(""),
        disclosure_date: str = Form(""),
        url: str = Form(""),
        file: UploadFile | None = None,
    ):
        try:
            if file and file.filename:
                if text.strip():
                    raise ValueError("Choose either a file or pasted text.")
                pages, origin = read_upload(file.filename, file.file.read(MAX_UPLOAD + 1))
                title = title or Path(file.filename).stem
            else:
                pages, origin = (text,), "text"
            document = make_document(ticker, title, pages, source, disclosure_date, url, origin)
            entry, duplicate = inbox.record(document)
            return {"entry": entry, "duplicate": duplicate}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/replay/step")
    def step():
        return inbox.step()

    @app.post("/api/replay/link")
    def link(request: LinkRequest):
        return inbox.link(request.connected)

    @app.post("/api/replay/reset")
    def reset():
        return inbox.reset()

    @app.get("/api/display.png")
    def display():
        return Response(display_png(inbox.state()), media_type="image/png")

    @app.get("/api/evaluation")
    def evaluation():
        return evaluate(model, transformer)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": MODEL, "inference": "local", "source": "simulated", **host_metrics()}

    return app
