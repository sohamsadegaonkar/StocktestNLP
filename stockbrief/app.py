"""HTTP interface for local and hosted research workspaces."""
import os
from datetime import date
from pathlib import Path
from threading import BoundedSemaphore
from typing import Literal

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .corpus import LABELS, TICKERS
from .display import display_png
from .edge import benchmark
from .evaluation import evaluate, host_metrics
from .fixtures import Document
from .hosted import COOKIE_AGE, COOKIE_NAME, HostedWorkspaces, WorkspaceLimit, public_https_origin, workspace_session
from .inputs import MAX_UPLOAD, make_document, read_upload
from .live_feed import LiveFeed, fetch_bytes
from .market import INSTRUMENTS, MarketData
from .intelligence import build_intelligence
from .market_context import disclosure_context
from .semantic import load_semantic, model_directory
from .store import Inbox

STATIC = Path(__file__).resolve().parent / "static"
MAX_REQUEST = MAX_UPLOAD + 256 * 1024


class EdgeConfig(BaseModel):
    watchlist: list[str]
    policy: Literal["fifo", "adaptive"]


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


def create_app(path: Path | None = None, transformer=None, *, public_origin=None, database_url=None, model=None, live_fetch=None):
    if database_url and public_origin is None:
        raise ValueError("A PostgreSQL URL requires hosted mode.")
    if model is None:
        model = load_semantic(model_directory())
    labels = {**getattr(model, "labels", LABELS), "review": "Needs review"}
    workspaces = None
    if public_origin is None:
        local_inbox = Inbox(path or Path(".runtime/stockbrief.sqlite"), model)
        hosts = ["localhost", "127.0.0.1", "testserver"]
    else:
        public_origin, hostname = public_https_origin(public_origin)
        hosts = [hostname]
        workspaces = HostedWorkspaces(path or Path(".runtime/stockbrief-hosted.sqlite"), model, database_url)
    mode = "hosted" if workspaces is not None else "local"
    storage = workspaces.storage if workspaces is not None else "sqlite"
    hosted_benchmark = evaluate(model, transformer) if workspaces is not None else None
    hosted_imports = BoundedSemaphore(2)
    market = MarketData() if workspaces is None and os.getenv("STOCKBRIEF_DISABLE_MARKET") != "1" else None

    live = LiveFeed(local_inbox, attachments=os.getenv("STOCKBRIEF_NSE_ATTACHMENTS") == "1",
                    fetch=live_fetch or fetch_bytes) if workspaces is None else None

    def get_inbox(request: Request):
        if workspaces is None:
            return local_inbox
        return workspaces.workspace(request.state.workspace_key)

    inbox_dependency = Depends(get_inbox)
    app = FastAPI(title="StockBrief Edge", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestLimit)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def workspace_request(request: Request, call_next):
        origin = request.headers.get("origin")
        request_url = request.url
        if request.method == "POST":
            expected = public_origin or f"{request_url.scheme}://{request.headers.get('host')}"
            bad_origin = origin != expected if public_origin is not None else origin and origin != expected
            if request.headers.get("sec-fetch-site") == "cross-site":
                return PlainTextResponse("Use the workspace website to send changes.", 403)
            if bad_origin:
                return PlainTextResponse("Use the workspace website to send changes.", 403)
        token = None
        if workspaces is not None:
            request.state.workspace_key, token = workspace_session(request.cookies.get(COOKIE_NAME))
        response = await call_next(request)
        if token and response.status_code < 400 and (
            request_url.path == "/" or request_url.path.startswith("/api/")
            and request_url.path not in {"/api/health", "/api/evaluation", "/api/display.png"}
        ):
            response.set_cookie(
                COOKIE_NAME, token, max_age=COOKIE_AGE, path="/", secure=True, httponly=True, samesite="lax",
            )
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
    def list_inbox(inbox=inbox_dependency):
        return {
            **inbox.state(), "labels": labels,
            "tickers": TICKERS, "transformer": transformer.name if transformer else None,
            "live_feed_available": live is not None,
            "mode": mode, "storage": storage,
        }

    @app.post("/api/live/refresh")
    def live_refresh():
        if live is None:
            raise HTTPException(404, "Live feed is available only in local mode.")
        return live.refresh()

    def local_edge():
        if live is None:
            raise HTTPException(404, "Edge scheduler is local-only.")
        return live.edge

    @app.get("/api/edge/status")
    def edge_status():
        return local_edge().status()

    @app.post("/api/edge/config")
    def edge_config(config: EdgeConfig):
        try:
            return local_edge().configure(config.watchlist, config.policy)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/edge/drain")
    def edge_drain():
        local_edge()
        return live.drain_offline()

    @app.post("/api/edge/benchmark")
    def edge_benchmark():
        local_edge()
        if not live.lock.acquire(blocking=False):
            raise HTTPException(409, "The edge processor is busy.")
        try:
            return benchmark(model)
        finally:
            live.lock.release()

    @app.get("/api/disclosures/{entry_id}")
    def get_disclosure(entry_id: int, inbox=inbox_dependency):
        entry = inbox.document(entry_id)
        if entry is None:
            raise HTTPException(404, "Disclosure not found.")
        return entry

    @app.get("/api/disclosures/{entry_id}/intelligence")
    def intelligence(entry_id: int, inbox=inbox_dependency):
        entry = inbox.document(entry_id)
        if entry is None:
            raise HTTPException(404, "Disclosure not found.")
        return build_intelligence(entry)

    @app.get("/api/disclosures/{entry_id}/second-opinion")
    def second_opinion(entry_id: int, inbox=inbox_dependency):
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
        inbox=inbox_dependency,
    ):
        if workspaces is not None and not hosted_imports.acquire(blocking=False):
            raise HTTPException(429, "Analysis is busy. Try again in a moment.")
        try:
            if workspaces is not None and len(url) > 2048:
                raise ValueError("The original source URL must be 2,048 characters or fewer.")
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
        finally:
            if workspaces is not None:
                hosted_imports.release()

    @app.exception_handler(WorkspaceLimit)
    async def workspace_limit(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=429)

    @app.post("/api/replay/step")
    def step(inbox=inbox_dependency):
        return inbox.step()

    @app.post("/api/replay/link")
    def link(request: LinkRequest, inbox=inbox_dependency):
        return inbox.link(request.connected)

    @app.post("/api/replay/reset")
    def reset(inbox=inbox_dependency):
        return inbox.reset()

    @app.post("/api/workspace/clear")
    def clear_workspace(inbox=inbox_dependency):
        if workspaces is None:
            raise HTTPException(404, "Workspace clearing is available only on the hosted website.")
        return inbox.clear()

    @app.get("/api/display.png")
    def display(inbox=inbox_dependency):
        return Response(display_png(inbox.state()), media_type="image/png")

    @app.get("/api/evaluation")
    def evaluation():
        return hosted_benchmark if hosted_benchmark is not None else evaluate(model, transformer)

    @app.get("/api/market/{symbol}")
    def market_chart(symbol: str, request: Request, period: Literal["1D", "1M"] = "1M"):
        if market is None or symbol not in INSTRUMENTS:
            raise HTTPException(404, "Market chart unavailable.")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Open the dashboard to view market charts.")
        return market.chart(symbol, period)

    @app.get("/api/market-context/{symbol}")
    def market_context(symbol: str, filing_date: date, request: Request):
        if market is None or symbol not in INSTRUMENTS or symbol == "NIFTY50":
            raise HTTPException(404, "Disclosure market context unavailable.")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Open the dashboard to view market context.")
        return disclosure_context(market.chart(symbol, "1M"), market.chart("NIFTY50", "1M"), filing_date)

    @app.get("/api/health")
    def health():
        if workspaces is not None:
            healthy = workspaces.healthy()
            return JSONResponse({
                "status": "ok" if healthy else "unavailable", "mode": mode,
                "model": model.name, "inference": "server", "source": "simulated", "storage": storage,
            }, status_code=200 if healthy else 503)
        return {"status": "ok", "model": model.name, "inference": "local", "source": "simulated", **host_metrics()}

    return app
