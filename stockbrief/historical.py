"""Point-in-time historical event comparison engine (analytics-only, zero trading signals).

Loads either the exported Phase 3 local package (.runtime/historical/phase3_package)
or the compact offline demo subset (stockbrief/historical_demo.json). Enforces:
1. Strict prior publication cutoff (published_ts_ist < cutoff_ist; conservative 00:00 IST on date-only disclosures).
2. Strict outcome window completion before cutoff (window_1d_end_date < cutoff_date and window_5d_end_date < cutoff_date).
3. Encoder compatibility checks with explicit category-matching fallback if weights differ or are absent.
4. Explicit separation of raw return, Nifty 50 excess return (BHAR), and market-model CAR plus corporate-action provenance.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import gzip
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .semantic import CHECKSUMS, MODEL_VERSION, WEIGHTS
from .semantic_data import SEMANTIC_LABELS

IST = timezone(timedelta(hours=5, minutes=30))
SCHEMA_VERSION = "stockbrief-historical-v1"
DEFAULT_PACKAGE_DIR = Path(".runtime/historical/phase3_package")
DEFAULT_DEMO_PATH = Path(__file__).resolve().with_name("historical_demo.json")
PRODUCER_COMMAND = ".\\training\\.venv\\Scripts\\python.exe -m stockbrief.historical_export --write-demo"
MISSING_HINT = (
    "Full Phase 3 local package (.runtime/historical/phase3_package/manifest.json) is absent; "
    f"export it locally with: {PRODUCER_COMMAND}"
)
FRAMING_NOTE = (
    "Observed historical market reactions around comparable past NSE announcements. "
    "Historical distributions do not imply future returns or causal proof; no directional probability "
    "or buy/sell trading signal is generated."
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _parse_ist_timestamp(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def _resolve_cutoff(published_at: str = "", disclosure_date: str = "", *, now: datetime | None = None) -> tuple[datetime, str]:
    now_ist = (now.astimezone(IST) if now and now.tzinfo else datetime.now(IST))
    if published_at:
        try:
            return _parse_ist_timestamp(published_at), "Disclosure publication timestamp converted to IST."
        except ValueError:
            pass
    if disclosure_date:
        try:
            day = date.fromisoformat(disclosure_date)
            return datetime.combine(day, time(0, 0), tzinfo=IST), (
                "Date-only disclosure (no time of day): conservative start-of-day IST cutoff "
                "(00:00:00+05:30) excludes same-day filings and unfinished windows."
            )
        except ValueError:
            pass
    return now_ist, "Undated disclosure: using current IST observation timestamp as cutoff."


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _summarize_horizon(values: list[dict[str, float]], pending_count: int) -> dict[str, Any]:
    if not values:
        return {
            "completed_count": 0,
            "pending_after_cutoff_count": pending_count,
            "raw_return_median_pct": None,
            "excess_nifty50_median_pct": None,
            "excess_nifty50_iqr_pct": None,
            "market_model_car_median_pct": None,
            "positive_excess_count": 0,
        }
    raw_arr = np.asarray([v["raw_return_pct"] for v in values], dtype=np.float64)
    exc_arr = np.asarray([v["excess_vs_nifty50_pct"] for v in values], dtype=np.float64)
    car_arr = np.asarray([v["market_model_car_pct"] for v in values], dtype=np.float64)
    q25, q75 = np.percentile(exc_arr, [25, 75])
    return {
        "completed_count": len(values),
        "pending_after_cutoff_count": pending_count,
        "raw_return_median_pct": round(float(np.median(raw_arr)), 2),
        "excess_nifty50_median_pct": round(float(np.median(exc_arr)), 2),
        "excess_nifty50_iqr_pct": [round(float(q25), 2), round(float(q75), 2)],
        "market_model_car_median_pct": round(float(np.median(car_arr)), 2),
        "positive_excess_count": int(np.sum(exc_arr > 0)),
    }


def _extract_query_text(entry: dict[str, Any]) -> str:
    brief = entry.get("brief") or {}
    snippets: list[str] = []
    for item in brief.get("evidence") or []:
        text = str(item.get("text") or "").strip()
        if text and text not in snippets:
            snippets.append(text)
    if not snippets:
        for page in entry.get("pages") or []:
            text = " ".join(str(page).split())
            if text:
                snippets.append(text[:600])
                break
    return " ".join(snippets)[:900]


class HistoricalIndex:
    """Thread-safe read-only point-in-time historical comparator."""

    def __init__(
        self,
        *,
        encoder=None,
        encoder_artifact: dict[str, Any] | None = None,
        package_dir: Path | None = None,
        demo_path: Path | None = None,
        disabled: bool | None = None,
    ):
        self.encoder = encoder
        self.encoder_artifact = encoder_artifact or getattr(encoder, "artifact", None) or {}
        if disabled is None:
            disabled = os.environ.get("STOCKBRIEF_DISABLE_HISTORICAL") == "1"
        self.disabled = disabled
        self.package_dir = package_dir if package_dir is not None else Path(
            os.environ.get("STOCKBRIEF_HISTORICAL_DIR") or DEFAULT_PACKAGE_DIR
        )
        self.demo_path = demo_path if demo_path is not None else DEFAULT_DEMO_PATH
        self.custom_package_requested = package_dir is not None or bool(os.environ.get("STOCKBRIEF_HISTORICAL_DIR"))

        self.mode = "disabled" if disabled else "unavailable"
        self.package_version = ""
        self.schema_version = SCHEMA_VERSION
        self.package_label = "Unavailable"
        self.records: list[dict[str, Any]] = []
        self._timestamps: list[datetime] = []
        self._embs: np.ndarray | None = None
        self.encoder_compatible = False
        self.encoder_note = ""
        self.load_error: str | None = None
        self.manifest: dict[str, Any] = {}

        if not self.disabled:
            self._load()

    def _encoder_weights_sha(self) -> str:
        return str(self.encoder_artifact.get("weights_sha256") or "")

    def _can_use_semantics(self, expected_sha: str, expected_dim: int = 384) -> tuple[bool, str]:
        if self.encoder is None or not hasattr(self.encoder, "encode"):
            return False, "No active sentence encoder supplied; using category matching fallback."
        actual_sha = self._encoder_weights_sha()
        if expected_sha and actual_sha and actual_sha != expected_sha:
            return False, (
                f"Active encoder weights ({actual_sha[:12]}) differ from historical bundle ({expected_sha[:12]}); "
                "using category matching fallback."
            )
        if expected_sha and not actual_sha:
            return False, "Active encoder lacks weights checksum metadata; using category matching fallback."
        if expected_dim != 384:
            return False, f"Unexpected embedding dimension {expected_dim}; using category matching fallback."
        return True, f"Compatible BGE encoder ({expected_sha[:12]})."

    def _load(self) -> None:
        manifest_path = self.package_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                self._load_phase3_package(manifest_path)
                return
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self.load_error = f"Local Phase 3 package invalid ({error})"
                if self.custom_package_requested:
                    self.mode = "unavailable"
                    self.package_label = "Unavailable (invalid custom package)"
                    return

        if self.demo_path.is_file():
            try:
                self._load_demo_package(self.demo_path)
                return
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self.load_error = f"Offline historical demo invalid ({error})"

        self.mode = "unavailable"
        self.package_label = "No usable historical comparison bundle loaded"

    def _load_phase3_package(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')}")
        files = manifest.get("files") or {}
        events_path = self.package_dir / "events.jsonl.gz"
        emb_path = self.package_dir / "embeddings_f16.npy"
        if not events_path.is_file():
            raise ValueError("Missing events.jsonl.gz")
        expected_ev_sha = (files.get("events.jsonl.gz") or {}).get("sha256")
        if expected_ev_sha and _sha256(events_path) != expected_ev_sha:
            raise ValueError("events.jsonl.gz SHA-256 checksum mismatch")

        records: list[dict[str, Any]] = []
        with gzip.open(events_path, "rt", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError("Historical package contains zero records")

        embs: np.ndarray | None = None
        enc_meta = manifest.get("encoder") or {}
        compat, enc_note = self._can_use_semantics(
            str(enc_meta.get("weights_sha256") or CHECKSUMS[WEIGHTS]),
            int(enc_meta.get("dim") or 384),
        )
        if compat and emb_path.is_file():
            expected_emb_sha = (files.get("embeddings_f16.npy") or {}).get("sha256")
            if expected_emb_sha and _sha256(emb_path) != expected_emb_sha:
                compat = False
                enc_note = "Historical embeddings file failed checksum verification; using category matching fallback."
            else:
                arr = np.load(emb_path)
                if arr.ndim != 2 or arr.shape[0] != len(records) or arr.shape[1] != 384:
                    compat = False
                    enc_note = f"Historical embeddings shape {arr.shape} mismatches {len(records)} records; using category fallback."
                else:
                    embs = arr.astype(np.float32)
        elif compat and not emb_path.is_file():
            compat = False
            enc_note = "Historical embeddings file missing; using category matching fallback."

        self.manifest = manifest
        self.records = records
        self._timestamps = [_parse_ist_timestamp(r["published_ts_ist"]) for r in records]
        self._embs = embs
        self.encoder_compatible = compat
        self.encoder_note = enc_note
        self.mode = "phase3_local"
        self.package_version = str(manifest.get("package_version") or "phase3-local")
        self.package_label = (
            f"Phase 3 NSE local archive ({len(records):,} filings, {manifest.get('unique_issuers', '?')} issuers)"
        )

    def _load_demo_package(self, demo_path: Path) -> None:
        payload = json.loads(demo_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported demo schema_version: {payload.get('schema_version')}")
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError("Demo dataset contains zero records")
        compat, enc_note = self._can_use_semantics(CHECKSUMS[WEIGHTS], 384)
        self.manifest = payload
        self.records = records
        self._timestamps = [_parse_ist_timestamp(r["published_ts_ist"]) for r in records]
        self._embs = None  # Encoded lazily on first semantic query if encoder is available & compatible
        self.encoder_compatible = compat
        self.encoder_note = enc_note
        self.mode = "offline_demo"
        self.package_version = str(payload.get("package_version") or "offline-demo-nse-v1")
        self.package_label = f"Offline demo subset ({len(records)} source-backed NSE filings)"

    def _ensure_demo_embeddings(self) -> bool:
        if self._embs is not None:
            return True
        if not self.encoder_compatible or self.encoder is None:
            return False
        texts = [r["excerpt"] for r in self.records]
        vecs = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        self._embs = vecs / norms
        return True

    def status(self) -> dict[str, Any]:
        dates = [r["published_date"] for r in self.records] if self.records else []
        return {
            "available": bool(self.records) and not self.disabled,
            "state": "disabled" if self.disabled else ("available" if self.mode == "phase3_local" else "demo" if self.mode == "offline_demo" else "unavailable"),
            "mode": self.mode,
            "package_version": self.package_version,
            "schema_version": self.schema_version,
            "package_label": self.package_label,
            "record_count": len(self.records),
            "unique_issuers": len({r["symbol"] for r in self.records}),
            "date_range": {"min": min(dates), "max": max(dates)} if dates else None,
            "encoder_compatible": self.encoder_compatible,
            "encoder_note": self.encoder_note,
            "load_error": self.load_error,
            "missing_artifact_hint": None if self.mode == "phase3_local" else MISSING_HINT,
            "framing_note": FRAMING_NOTE,
        }

    def compare(
        self,
        *,
        query_text: str,
        symbol: str = "",
        category: str = "",
        published_at: str = "",
        disclosure_date: str = "",
        exclude_url: str = "",
        exclude_title: str = "",
        origin: str = "",
        limit: int = 6,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        cutoff_dt, cutoff_assumption = _resolve_cutoff(published_at, disclosure_date, now=now)
        cutoff_iso = cutoff_dt.isoformat()
        cutoff_date = cutoff_dt.date().isoformat()
        base = {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "package_mode": self.mode,
            "package_label": self.package_label,
            "cutoff_ist": cutoff_iso,
            "cutoff_date": cutoff_date,
            "cutoff_assumption": cutoff_assumption,
            "encoder_compatible": self.encoder_compatible,
            "encoder_note": self.encoder_note,
            "total_package_events": len(self.records),
            "candidate_pool_before_cutoff": 0,
            "matching_mode": "none",
            "framing_note": FRAMING_NOTE,
            "missing_artifact_hint": None if self.mode == "phase3_local" else MISSING_HINT,
            "summary_distribution": {
                "horizon_1d": _summarize_horizon([], 0),
                "horizon_5d": _summarize_horizon([], 0),
            },
            "comparables": [],
            "limitations": [],
        }

        if self.disabled:
            return {**base, "state": "disabled", "reason": "disabled_by_config",
                    "limitations": ["Historical comparison is disabled via STOCKBRIEF_DISABLE_HISTORICAL=1."]}

        if not self.records:
            return {**base, "state": "unavailable", "reason": "no_historical_package",
                    "limitations": [self.load_error or "No historical comparison bundle is installed.", MISSING_HINT]}

        clean_query = " ".join((query_text or "").split())
        if len(clean_query) < 12:
            return {**base, "state": "insufficient_evidence", "reason": "insufficient_input_text",
                    "limitations": ["Provide at least 12 characters of disclosure text before comparing historical events."]}

        query_tokens = _tokens(clean_query) | _tokens(exclude_title)
        norm_excl_url = (exclude_url or "").strip().lower()
        clean_symbol = (symbol or "").upper().strip()
        if clean_symbol == "TATAMOTORS":
            # Tata Motors passenger vehicles ticker aligned cleanly alongside historical TATAMOTORS / TMPV.
            match_symbols = {"TATAMOTORS", "TMPV"}
        elif clean_symbol == "TMPV":
            match_symbols = {"TMPV", "TATAMOTORS"}
        else:
            match_symbols = {clean_symbol} if clean_symbol and clean_symbol != "OTHER" else set()

        eligible: list[int] = []
        for idx, rec in enumerate(self.records):
            if self._timestamps[idx] >= cutoff_dt:
                continue
            rec_url = (rec.get("source_url") or "").strip().lower()
            if norm_excl_url and rec_url and rec_url == norm_excl_url:
                continue
            if rec["symbol"] in match_symbols and rec["published_date"] == cutoff_date:
                continue
            if rec["symbol"] in match_symbols and _jaccard(query_tokens, _tokens(rec["excerpt"])) >= 0.72:
                continue
            eligible.append(idx)

        base["candidate_pool_before_cutoff"] = len(eligible)
        if not eligible:
            return {
                **base,
                "state": "insufficient_evidence",
                "reason": "no_prior_events_before_cutoff",
                "limitations": [
                    f"No historical NSE disclosures in the active bundle precede cutoff {cutoff_iso}.",
                    *([] if self.mode == "phase3_local" else [MISSING_HINT]),
                ],
            }

        if category == "review":
            return {
                **base,
                "state": "insufficient_evidence",
                "reason": "disclosure_needs_review",
                "limitations": [
                    "The disclosure did not establish a clear corporate event category, so historical return comparisons are withheld.",
                ],
            }

        use_semantic = False
        cos_scores = np.zeros(len(eligible), dtype=np.float32)
        if self.encoder_compatible:
            try:
                if self._ensure_demo_embeddings() and self._embs is not None:
                    q_vec = np.asarray(self.encoder.encode([clean_query[:600]]), dtype=np.float32).reshape(-1)
                    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
                    sub_embs = self._embs[np.asarray(eligible, dtype=np.int64)]
                    cos_scores = sub_embs @ q_vec
                    use_semantic = True
            except Exception as error:
                base["encoder_compatible"] = False
                base["encoder_note"] = f"Encoder execution failed ({error}); using category matching fallback."
                use_semantic = False

        base["matching_mode"] = "semantic_and_category" if use_semantic else "category_fallback"

        scored: list[tuple[float, float, int]] = []
        target_cat = category if category and category != "review" else ""
        for pos, idx in enumerate(eligible):
            rec = self.records[idx]
            cos = float(cos_scores[pos]) if use_semantic else 0.0
            cat_match = 1.0 if (target_cat and rec["category"] == target_cat) else 0.0
            sym_match = 1.0 if (match_symbols and rec["symbol"] in match_symbols) else 0.0
            if use_semantic:
                if cos < 0.54:
                    continue
                score = 0.72 * cos + 0.16 * cat_match + 0.12 * sym_match
            else:
                lex = _jaccard(query_tokens, _tokens(rec["excerpt"]))
                if not cat_match and lex < 0.12 and not sym_match:
                    continue
                # Explicit category/lexical fallback when compatible semantic vectors are absent.
                score = 0.55 * cat_match + 0.25 * lex + 0.20 * sym_match
            scored.append((score, cos, idx))

        if not scored:
            return {
                **base,
                "state": "insufficient_evidence",
                "reason": "no_similar_precedents",
                "limitations": [
                    "No prior disclosures met the similarity threshold before this cutoff date.",
                    *([] if self.mode == "phase3_local" else [MISSING_HINT]),
                ],
            }

        scored.sort(key=lambda item: (item[0], self.records[item[2]]["published_ts_ist"]), reverse=True)
        selected: list[tuple[float, float, dict[str, Any]]] = []
        seen_signatures: set[tuple[str, str, str]] = set()
        seen_excerpts: set[str] = set()
        for score, cos, idx in scored:
            rec = self.records[idx]
            sig = (rec["symbol"], rec["t0_date"], rec["category"])
            short_ex = rec["excerpt"][:90].lower()
            if sig in seen_signatures or short_ex in seen_excerpts:
                continue
            seen_signatures.add(sig)
            seen_excerpts.add(short_ex)
            selected.append((score, cos, rec))
            if len(selected) >= max(1, min(limit, 12)):
                break

        comparables: list[dict[str, Any]] = []
        completed_1d: list[dict[str, float]] = []
        completed_5d: list[dict[str, float]] = []
        pending_1d = 0
        pending_5d = 0
        has_unverified = False
        has_split = False
        has_overlap = False

        for score, cos, rec in selected:
            w1_ready = rec["window_1d_end_date"] < cutoff_date
            w5_ready = rec["window_5d_end_date"] < cutoff_date
            obs_1d = dict(rec["observed_1d"]) if w1_ready else None
            obs_5d = dict(rec["observed_5d"]) if w5_ready else None
            if obs_1d is not None:
                completed_1d.append(obs_1d)
            else:
                pending_1d += 1
            if obs_5d is not None:
                completed_5d.append(obs_5d)
            else:
                pending_5d += 1

            if not rec["corp_action_verified"]:
                has_unverified = True
            if rec.get("has_split_near") or rec.get("has_div_near"):
                has_split = True
            if rec.get("is_overlapping_10d"):
                has_overlap = True

            reasons: list[str] = []
            if use_semantic:
                reasons.append(f"Semantic similarity {cos:.2f}")
            else:
                reasons.append("Category matching fallback (semantic vectors unavailable or incompatible)")
            if target_cat and rec["category"] == target_cat:
                reasons.append(f"Same category ({SEMANTIC_LABELS.get(target_cat, target_cat)})")
            if match_symbols and rec["symbol"] in match_symbols:
                reasons.append(f"Same issuer ({rec['symbol']})")

            comparables.append({
                "event_id": rec["event_id"],
                "symbol": rec["symbol"],
                "sector": rec["sector"],
                "published_ts_ist": rec["published_ts_ist"],
                "t0_date": rec["t0_date"],
                "window_1d_end_date": rec["window_1d_end_date"],
                "window_5d_end_date": rec["window_5d_end_date"],
                "category": rec["category"],
                "category_name": rec.get("category_name") or SEMANTIC_LABELS.get(rec["category"], rec["category"]),
                "stage": rec.get("stage", "Unclear"),
                "source_url": rec["source_url"],
                "excerpt": rec["excerpt"],
                "similarity_score": round(float(score), 4),
                "match_reasons": reasons,
                "corp_action_verified": bool(rec["corp_action_verified"]),
                "market_data_provider": rec["market_data_provider"],
                "has_split_near": bool(rec.get("has_split_near", False)),
                "has_div_near": bool(rec.get("has_div_near", False)),
                "is_overlapping_10d": bool(rec.get("is_overlapping_10d", False)),
                "benchmark": rec.get("benchmark", "NIFTY50"),
                "window_1d_status": "completed_before_cutoff" if w1_ready else "unavailable_before_cutoff",
                "window_5d_status": "completed_before_cutoff" if w5_ready else "unavailable_before_cutoff",
                "observed_1d": obs_1d,
                "observed_5d": obs_5d,
            })

        limitations = [FRAMING_NOTE]
        if self.load_error:
            limitations.append(f"{self.load_error}; fell back to offline demo package.")
        if self.mode == "offline_demo":
            limitations.append(
                f"Showing curated offline demo subset ({len(self.records)} real NSE filings). {MISSING_HINT}"
            )
        if not use_semantic:
            limitations.append(base["encoder_note"])
        if pending_1d or pending_5d:
            limitations.append(
                f"Outcome-window cutoff applied: {pending_1d} 1-day and {pending_5d} 5-day windows closing on or after {cutoff_date} were excluded to prevent future leakage."
            )
        if has_unverified:
            limitations.append(
                "One or more comparables use Upstox unadjusted daily closes fallback (for example TMPV/TCS where Yahoo split/dividend adjustment was unavailable); verify corporate actions separately."
            )
        if has_split:
            limitations.append(
                "At least one comparable occurred near a recorded dividend or stock split boundary."
            )
        if has_overlap:
            limitations.append(
                "Comparables marked 'overlapping <=10d' occurred within 10 trading sessions of another announcement by the same issuer."
            )
        if origin == "synthetic":
            limitations.append("Selected entry is a synthetic replay fixture; comparisons below are illustrative only.")

        return {
            **base,
            "state": "demo" if self.mode == "offline_demo" else "available",
            "reason": None,
            "summary_distribution": {
                "horizon_1d": _summarize_horizon(completed_1d, pending_1d),
                "horizon_5d": _summarize_horizon(completed_5d, pending_5d),
            },
            "comparables": comparables,
            "limitations": limitations,
        }

    def compare_entry(self, entry: dict[str, Any], *, limit: int = 6, now: datetime | None = None) -> dict[str, Any]:
        brief = entry.get("brief") or {}
        query_text = _extract_query_text(entry)
        return self.compare(
            query_text=query_text,
            symbol=str(entry.get("ticker") or ""),
            category=str(brief.get("category") or ""),
            published_at=str(entry.get("published_at") or ""),
            disclosure_date=str(entry.get("date") or ""),
            exclude_url=str(entry.get("url") or ""),
            exclude_title=str(entry.get("title") or ""),
            origin=str(entry.get("origin") or ""),
            limit=limit,
            now=now,
        )
