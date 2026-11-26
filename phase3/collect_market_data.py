"""Collect 2021-2026 historical OHLCV, Nifty 50, sector indices, India VIX, actions, and fundamentals via yfinance."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import time
from typing import Any

from urllib.parse import quote
import httpx
import pandas as pd
import yfinance as yf

EVENTS_PATH = Path("phase3/.cache/nse_events_pool.jsonl")
MARKET_DIR = Path("phase3/.cache/market")
PROV_PATH = Path("phase3/data/market_provenance.json")

START_DATE = "2021-07-01"
END_DATE = "2026-11-26"

BENCHMARK_INDICES = {
    "NIFTY50": "^NSEI",
    "INDIAVIX": "^INDIAVIX",
    "CNXIT": "^CNXIT",
    "NSEBANK": "^NSEBANK",
    "CNXFIN": "^CNXFIN",
    "CNXAUTO": "^CNXAUTO",
    "CNXPHARMA": "^CNXPHARMA",
    "CNXFMCG": "^CNXFMCG",
    "CNXMETAL": "^CNXMETAL",
    "CNXENERGY": "^CNXENERGY",
}

SECTOR_TO_INDEX = {
    "Technology": "CNXIT",
    "Financial Services": "NSEBANK",
    "Consumer Cyclical": "CNXAUTO",
    "Healthcare": "CNXPHARMA",
    "Consumer Defensive": "CNXFMCG",
    "Basic Materials": "CNXMETAL",
    "Energy": "CNXENERGY",
    "Utilities": "CNXENERGY",
    "Industrials": "NIFTY50",
    "Communication Services": "NIFTY50",
    "Real Estate": "NIFTY50",
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # Filter bad/non-positive prices
    if "Close" in out.columns:
        out = out[out["Close"].notna() & (out["Close"] > 0)]
    if "Adj Close" in out.columns and "Close" in out.columns:
        adj_f = (out["Adj Close"] / out["Close"]).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
        out["Adj_Factor"] = adj_f
        for col in ("Open", "High", "Low", "Close"):
            if col in out.columns:
                out[f"{col}_Adj"] = out[col] * adj_f
    return out


def fetch_symbol_history_and_fundamentals(sym: str) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    yf_sym = f"{sym}.NS"
    df = pd.DataFrame()
    fund: dict[str, Any] = {"symbol": sym, "yf_symbol": yf_sym, "sector": None, "industry": None, "shares": None, "annual": {}, "quarterly": {}}
    try:
        tk = yf.Ticker(yf_sym)
        raw_df = tk.history(start=START_DATE, end=END_DATE, auto_adjust=False, actions=True)
        df = normalize_df(raw_df)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        fund["sector"] = info.get("sector")
        fund["industry"] = info.get("industry")
        fund["shares"] = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")

        def extract_stmt(stmt: pd.DataFrame) -> dict[str, dict[str, float]]:
            res: dict[str, dict[str, float]] = {}
            if stmt is None or stmt.empty:
                return res
            for col in stmt.columns:
                dstr = pd.Timestamp(col).strftime("%Y-%m-%d")
                vals = {}
                for row_key in (
                    "Total Revenue", "Operating Revenue", "Net Income", "Operating Income",
                    "EBITDA", "Diluted EPS", "Basic EPS", "Total Debt", "Stockholders Equity",
                    "Operating Cash Flow", "Free Cash Flow",
                ):
                    if row_key in stmt.index:
                        v = stmt.loc[row_key, col]
                        if pd.notna(v):
                            vals[row_key] = float(v)
                if vals:
                    res[dstr] = vals
            return res

        try:
            fund["annual"] = {
                "inc": extract_stmt(tk.income_stmt),
                "bs": extract_stmt(tk.balance_sheet),
                "cf": extract_stmt(tk.cashflow),
            }
            fund["quarterly"] = {
                "inc": extract_stmt(tk.quarterly_income_stmt),
                "bs": extract_stmt(tk.quarterly_balance_sheet),
                "cf": extract_stmt(tk.quarterly_cashflow),
            }
        except Exception:
            pass
    except Exception:
        pass
    return sym, df, fund


def fetch_upstox_daily(isin: str) -> pd.DataFrame:
    if not isin or not isin.startswith("INE"):
        return pd.DataFrame()
    key = quote(f"NSE_EQ|{isin}", safe="")
    all_rows = []
    # Fetch in 2 windows (Upstox caps per-request range safely)
    for fdt, tdt in (("2021-07-01", "2024-06-30"), ("2024-07-01", "2026-11-26")):
        try:
            resp = httpx.get(f"https://api.upstox.com/v3/historical-candle/{key}/days/1/{tdt}/{fdt}", headers={"Accept": "application/json"}, timeout=10)
            if resp.status_code == 200:
                rows = resp.json().get("data", {}).get("candles") or []
                all_rows.extend(rows)
        except Exception:
            pass
    if not all_rows:
        return pd.DataFrame()
    recs = {}
    for row in all_rows:
        if isinstance(row, list) and len(row) >= 6:
            dt = pd.Timestamp(row[0]).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
            c = float(row[4])
            if c > 0:
                recs[dt] = {
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": c,
                    "Adj Close": c,
                    "Volume": float(row[5]),
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                    "Adj_Factor": 1.0,
                    "Open_Adj": float(row[1]),
                    "High_Adj": float(row[2]),
                    "Low_Adj": float(row[3]),
                    "Close_Adj": c,
                    "Provider": "upstox_fallback",
                }
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    return df


def main() -> None:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    PROV_PATH.parent.mkdir(parents=True, exist_ok=True)

    sym_to_isin: dict[str, str] = {}
    for line in EVENTS_PATH.open("r", encoding="utf-8"):
        if line.strip():
            obj = json.loads(line)
            if obj.get("isin"):
                sym_to_isin[obj["symbol"]] = obj["isin"]
    symbols = sorted({json.loads(line)["symbol"] for line in EVENTS_PATH.open("r", encoding="utf-8") if line.strip()})
    print(f"=== Step 2: Fetching Market Indices & {len(symbols)} NSE Equities ({START_DATE}..{END_DATE}) ===", flush=True)

    eq_path = MARKET_DIR / "equities.pkl"
    fund_path = MARKET_DIR / "fundamentals.pkl"
    idx_path = MARKET_DIR / "indices.pkl"
    if idx_path.is_file():
        with idx_path.open("rb") as f:
            indices = pickle.load(f)
    else:
        indices = {}
        for name, ysym in BENCHMARK_INDICES.items():
            df = normalize_df(yf.Ticker(ysym).history(start=START_DATE, end=END_DATE, auto_adjust=False, actions=False))
            indices[name] = df
            print(f"  Index {name} ({ysym}): {len(df)} trading sessions", flush=True)

    equities: dict[str, pd.DataFrame] = {}
    fundamentals: dict[str, dict[str, Any]] = {}
    if eq_path.is_file() and fund_path.is_file():
        with eq_path.open("rb") as f:
            equities = pickle.load(f)
        with fund_path.open("rb") as f:
            fundamentals = pickle.load(f)

    missing_syms = [s for s in symbols if s not in equities]
    print(f"  Already cached equities: {len(equities)}; recovering {len(missing_syms)} remaining symbols via Upstox ISIN fallback...", flush=True)
    unresolved: list[str] = []
    early_ended: list[dict[str, str]] = []
    upstox_filled = 0

    for s in missing_syms:
        isin = sym_to_isin.get(s, "")
        df = fetch_upstox_daily(isin)
        if not df.empty and len(df) >= 60:
            equities[s] = df
            upstox_filled += 1
        if s not in fundamentals:
            fundamentals[s] = {"symbol": s, "yf_symbol": f"{s}.NS", "sector": None, "industry": None, "shares": None, "annual": {}, "quarterly": {}}
    print(f"  Upstox ISIN fallback filled {upstox_filled}/{len(missing_syms)} symbols (total resolved={len(equities)})", flush=True)

    for s in symbols:
        df = equities.get(s)
        if df is None or df.empty or len(df) < 60:
            unresolved.append(s)
        else:
            last_dt = df.index.max().strftime("%Y-%m-%d")
            if last_dt < "2026-10-01":
                early_ended.append({"symbol": s, "last_date": last_dt, "rows": str(len(df))})

    with (MARKET_DIR / "indices.pkl").open("wb") as f:
        pickle.dump(indices, f)
    with (MARKET_DIR / "equities.pkl").open("wb") as f:
        pickle.dump(equities, f)
    with (MARKET_DIR / "fundamentals.pkl").open("wb") as f:
        pickle.dump(fundamentals, f)

    sector_counts = Counter((fundamentals[s].get("sector") or "Unknown") for s in equities)
    prov = {
        "source": "Yahoo Finance (yfinance daily historical OHLCV + corporate actions + statements)",
        "live_feed_warning": "Cached historical daily bars only; never represented as live intraday prices.",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "indices_count": {k: len(v) for k, v in indices.items()},
        "requested_symbols": len(symbols),
        "resolved_symbols": len(equities),
        "unresolved_or_short_symbols_count": len(unresolved),
        "unresolved_sample": unresolved[:40],
        "early_ended_or_renamed_delisted_count": len(early_ended),
        "early_ended_sample": early_ended[:25],
        "resolved_sector_counts": dict(sector_counts),
        "with_annual_fundamentals": sum(1 for s in equities if fundamentals[s]["annual"].get("inc")),
        "with_quarterly_fundamentals": sum(1 for s in equities if fundamentals[s]["quarterly"].get("inc")),
    }
    PROV_PATH.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"\nMarket collection complete: {len(equities)}/{len(symbols)} symbols resolved ({len(unresolved)} unresolved, {len(early_ended)} early-ended/renamed).", flush=True)
    print(f"Annual fundamentals: {prov['with_annual_fundamentals']}, Quarterly fundamentals: {prov['with_quarterly_fundamentals']}", flush=True)


if __name__ == "__main__":
    main()
