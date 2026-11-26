"""Reproducible exporter from Phase 3 local research artifacts into the slim StockBrief runtime bundle.

Run under an environment with pandas/pyarrow/numpy (for example training\\.venv):
    .\\training\\.venv\\Scripts\\python.exe -m stockbrief.historical_export --write-demo
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

from .nlp import chunks_for, stage_for
from .semantic import CHECKSUMS, MODEL_NAME, MODEL_VERSION, REPOSITORY, REVISION, WEIGHTS
from .semantic_data import SEMANTIC_LABELS

SCHEMA_VERSION = "stockbrief-historical-v1"
PACKAGE_VERSION = "phase3-nse-2022-2026-v1"
WATCHLIST = ("INFY", "TMPV", "HDFCBANK", "RELIANCE", "TCS")
P1_CATEGORIES = (
    "corporate_action",
    "deal",
    "earnings",
    "governance",
    "operations",
    "pricing",
    "regulatory",
    "routine",
)

DEFAULT_FEATURES = Path("phase3/.cache/features_table.parquet")
DEFAULT_EMBS = Path("phase3/.cache/p1_embs.npy")
DEFAULT_INDICES = Path("phase3/.cache/market/indices.pkl")
DEFAULT_OUT_DIR = Path(".runtime/historical/phase3_package")
DEFAULT_DEMO_PATH = Path("stockbrief/historical_demo.json")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _round_pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return round(num * 100.0, 3)


def _infer_category(row: dict[str, Any]) -> str:
    best_cat = "routine"
    best_val = -1.0
    for cat in P1_CATEGORIES:
        val = float(row.get(f"nlp_p1_cat_{cat}") or 0.0)
        if val > best_val:
            best_val = val
            best_cat = cat
    return best_cat


def _clean_excerpt(text: str, limit: int = 320) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    cut = compact.rfind(" ", 0, limit - 1)
    return (compact[:cut] if cut > 80 else compact[: limit - 1]).rstrip() + "..."


def _require_artifacts(features_path: Path, embs_path: Path, indices_path: Path) -> None:
    missing = [str(p) for p in (features_path, embs_path, indices_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Phase 3 local research artifacts: "
            + ", ".join(missing)
            + ". Run Phase 3 market/feature preparation or use the offline demo dataset."
        )


def _build_record(row: dict[str, Any], trading_dates: list[str], emb_row: int) -> dict[str, Any] | None:
    text = _clean_excerpt(row.get("text", ""), limit=320)
    if len(text) < 12:
        return None
    t0_idx = int(row["t0_idx"])
    if t0_idx < 0 or t0_idx + 4 >= len(trading_dates):
        return None
    raw_1 = _round_pct(row.get("target_raw_ret_1d"))
    bhar_1 = _round_pct(row.get("target_bhar_1d"))
    car_1 = _round_pct(row.get("target_car_1d"))
    raw_5 = _round_pct(row.get("target_raw_ret_5d"))
    bhar_5 = _round_pct(row.get("target_bhar_5d"))
    car_5 = _round_pct(row.get("target_car_5d"))
    if any(val is None for val in (raw_1, bhar_1, car_1, raw_5, bhar_5, car_5)):
        return None

    category = _infer_category(row)
    stage, _ = stage_for(category, chunks_for((text,)))
    verified = bool(int(row.get("corp_action_verified") or 0))
    provider = (
        "Yahoo Finance (split/dividend adjusted)"
        if verified
        else "Upstox fallback (daily unadjusted closes)"
    )
    return {
        "event_id": str(row["event_id"]),
        "emb_row": emb_row,
        "symbol": str(row["symbol"]),
        "sector": str(row.get("sector") or "Unknown"),
        "published_ts_ist": str(row["published_ts_ist"]),
        "published_date": str(row["published_date"]),
        "t0_date": str(row["t0_date"]),
        "window_1d_end_date": trading_dates[t0_idx],
        "window_5d_end_date": trading_dates[t0_idx + 4],
        "timing_bucket": str(row.get("timing_bucket") or "unknown"),
        "category": category,
        "category_name": SEMANTIC_LABELS.get(category, category),
        "event_family": str(row.get("event_family_hint") or category),
        "stage": stage,
        "regime": str(row.get("regime_causal") or "unknown"),
        "source_url": str(row.get("source_url") or ""),
        "excerpt": text,
        "corp_action_verified": verified,
        "market_data_provider": provider,
        "has_split_near": bool(row.get("has_split_near", False)),
        "has_div_near": bool(row.get("has_div_near", False)),
        "is_overlapping_10d": bool(int(row.get("is_overlapping_10d") or 0)),
        "benchmark": "NIFTY50",
        "observed_1d": {
            "raw_return_pct": raw_1,
            "excess_vs_nifty50_pct": bhar_1,
            "market_model_car_pct": car_1,
        },
        "observed_5d": {
            "raw_return_pct": raw_5,
            "excess_vs_nifty50_pct": bhar_5,
            "market_model_car_pct": car_5,
        },
    }


def _select_demo_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Curate a compact offline demo subset covering 2022-2026 across categories & tracked issuers."""
    chosen: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _score_candidate(rec: dict[str, Any]) -> tuple[int, int, int, int, str]:
        return (
            1 if rec["symbol"] in WATCHLIST else 0,
            1 if rec["corp_action_verified"] else 0,
            1 if not rec["is_overlapping_10d"] else 0,
            min(len(rec["excerpt"]), 280),
            rec["published_ts_ist"],
        )

    pre_2024 = [r for r in records if r["published_date"] < "2024-04-01" and len(r["excerpt"]) >= 85]
    post_2024 = [r for r in records if r["published_date"] >= "2024-04-01" and len(r["excerpt"]) >= 85]

    for pool, per_cat in ((pre_2024, 2), (post_2024, 2)):
        for cat in P1_CATEGORIES:
            cands = sorted([r for r in pool if r["category"] == cat], key=_score_candidate, reverse=True)
            taken = 0
            for item in cands:
                if item["event_id"] in seen_ids:
                    continue
                chosen.append(item)
                seen_ids.add(item["event_id"])
                taken += 1
                if taken >= per_cat:
                    break

    # Ensure TMPV and TCS are represented even when corp_action_verified=False (explicitly flagged).
    for sym in WATCHLIST:
        if sum(1 for item in chosen if item["symbol"] == sym) >= 2:
            continue
        sym_cands = sorted([r for r in records if r["symbol"] == sym and len(r["excerpt"]) >= 85],
                           key=_score_candidate, reverse=True)
        for item in sym_cands:
            if item["event_id"] in seen_ids:
                continue
            chosen.append(item)
            seen_ids.add(item["event_id"])
            if sum(1 for r in chosen if r["symbol"] == sym) >= 2:
                break

    chosen.sort(key=lambda r: r["published_ts_ist"], reverse=True)
    cleaned = []
    for item in chosen:
        copy = {k: v for k, v in item.items() if k != "emb_row"}
        cleaned.append(copy)
    return cleaned


def export_historical_package(
    *,
    features_path: Path = DEFAULT_FEATURES,
    embs_path: Path = DEFAULT_EMBS,
    indices_path: Path = DEFAULT_INDICES,
    out_dir: Path = DEFAULT_OUT_DIR,
    demo_path: Path | None = None,
) -> dict[str, Any]:
    """Export Phase 3 cache into compact runtime assets without pandas runtime dependency."""
    _require_artifacts(features_path, embs_path, indices_path)
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Exporting Phase 3 parquet/pickle artifacts requires pandas and pyarrow. "
            "Run: .\\training\\.venv\\Scripts\\python.exe -m stockbrief.historical_export --write-demo"
        ) from error

    df = pd.read_parquet(features_path)
    embs = np.load(embs_path, mmap_mode="r")
    if len(embs.shape) != 2 or embs.shape[1] != 384:
        raise ValueError(f"Expected 384-d embeddings in {embs_path}, got shape {embs.shape}.")

    with indices_path.open("rb") as source:
        indices = pickle.load(source)
    nifty = indices["NIFTY50"]
    trading_dates = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in nifty.index]

    needed_cols = [
        "event_id",
        "symbol",
        "sector",
        "published_ts_ist",
        "published_date",
        "t0_date",
        "t0_idx",
        "timing_bucket",
        "event_family_hint",
        "regime_causal",
        "source_url",
        "text",
        "corp_action_verified",
        "is_overlapping_10d",
        "emb_row_idx",
        "target_raw_ret_1d",
        "target_bhar_1d",
        "target_car_1d",
        "target_raw_ret_5d",
        "target_bhar_5d",
        "target_car_5d",
        *(f"nlp_p1_cat_{cat}" for cat in P1_CATEGORIES),
    ]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Phase 3 table missing expected columns: {missing_cols}")

    optional_aligned_flags: dict[str, tuple[bool, bool]] = {}
    aligned_pkl = features_path.parent / "aligned_events.pkl"
    if aligned_pkl.is_file():
        with aligned_pkl.open("rb") as source:
            aligned_rows = pickle.load(source)
        for item in aligned_rows:
            optional_aligned_flags[str(item["event_id"])] = (
                bool(item.get("has_split_near", False)),
                bool(item.get("has_div_near", False)),
            )

    seen_dedup: set[tuple[str, str, str]] = set()
    records: list[dict[str, Any]] = []
    emb_rows_to_copy: list[int] = []

    for row in df[needed_cols].sort_values("published_ts_ist").to_dict("records"):
        norm_text = " ".join(str(row.get("text") or "").lower().split())
        dedup_key = (str(row["symbol"]), str(row["published_date"]), norm_text)
        if dedup_key in seen_dedup:
            continue
        has_split, has_div = optional_aligned_flags.get(str(row["event_id"]), (False, False))
        row["has_split_near"] = has_split
        row["has_div_near"] = has_div
        src_emb_idx = int(row["emb_row_idx"])
        if not 0 <= src_emb_idx < embs.shape[0]:
            continue
        built = _build_record(row, trading_dates, len(records))
        if built is None:
            continue
        seen_dedup.add(dedup_key)
        records.append(built)
        emb_rows_to_copy.append(src_emb_idx)

    if not records:
        raise ValueError("No valid aligned historical events survived filtering.")

    selected_embs = np.asarray(embs[emb_rows_to_copy], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(selected_embs, axis=1, keepdims=True), 1e-9)
    selected_embs_f16 = (selected_embs / norms).astype(np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl.gz"
    emb_out_path = out_dir / "embeddings_f16.npy"
    manifest_path = out_dir / "manifest.json"

    with gzip.open(events_path, "wt", encoding="utf-8", compresslevel=6) as out:
        for item in records:
            out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    np.save(emb_out_path, selected_embs_f16)

    dates = [r["published_date"] for r in records]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "package_mode": "phase3_local",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer_command": ".\\training\\.venv\\Scripts\\python.exe -m stockbrief.historical_export",
        "encoder": {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "repository": REPOSITORY,
            "revision": REVISION,
            "weights_sha256": CHECKSUMS[WEIGHTS],
            "dim": 384,
            "dtype": "float16",
        },
        "record_count": len(records),
        "unique_issuers": len({r["symbol"] for r in records}),
        "corp_action_verified_count": sum(1 for r in records if r["corp_action_verified"]),
        "non_overlapping_10d_count": sum(1 for r in records if not r["is_overlapping_10d"]),
        "date_range": {"min_published_date": min(dates), "max_published_date": max(dates)},
        "provenance": {
            "text_source": "Public NSE corporate announcements (nsearchives.nseindia.in) short exchange summaries only",
            "market_data": "Yahoo Finance split/dividend-adjusted daily closes (where verified) and Upstox unadjusted daily closes fallback (explicitly flagged); NIFTY 50 benchmark.",
            "excluded_sources": ["Financial PhraseBank (NC-SA)", "FinRED (academic)", "user uploads", "Phase 3 directional predictions / signals"],
            "analytical_scope": "Historical distributions of observed 1d/5d reactions around prior disclosures. Not causal proof or buy/sell signals.",
        },
        "files": {
            "events.jsonl.gz": {"bytes": events_path.stat().st_size, "sha256": _sha256(events_path)},
            "embeddings_f16.npy": {"bytes": emb_out_path.stat().st_size, "sha256": _sha256(emb_out_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if demo_path is not None:
        demo_records = _select_demo_records(records)
        demo_payload = {
            "schema_version": SCHEMA_VERSION,
            "package_version": "offline-demo-nse-v1",
            "package_mode": "offline_demo",
            "created_at": manifest["created_at"],
            "producer_command": ".\\training\\.venv\\Scripts\\python.exe -m stockbrief.historical_export --write-demo",
            "encoder_model_version": MODEL_VERSION,
            "provenance": manifest["provenance"],
            "records": demo_records,
        }
        demo_path.write_text(json.dumps(demo_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["demo_record_count"] = len(demo_records)
        manifest["demo_bytes"] = demo_path.stat().st_size
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Phase 3 local research cache into StockBrief runtime bundle.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--embs", type=Path, default=DEFAULT_EMBS)
    parser.add_argument("--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-demo", action="store_true", help="Also refresh stockbrief/historical_demo.json")
    args = parser.parse_args(argv)
    try:
        manifest = export_historical_package(
            features_path=args.features,
            embs_path=args.embs,
            indices_path=args.indices,
            out_dir=args.out,
            demo_path=DEFAULT_DEMO_PATH if args.write_demo else None,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Historical export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
