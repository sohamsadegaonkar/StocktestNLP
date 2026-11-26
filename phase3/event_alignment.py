"""IST Exchange Trading-Session Alignment, Market Model Estimation & Event Reaction Computation."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from phase3.collect_market_data import SECTOR_TO_INDEX

EVENTS_PATH = Path("phase3/.cache/nse_events_pool.jsonl")
MARKET_DIR = Path("phase3/.cache/market")
ALIGNED_OUT = Path("phase3/.cache/aligned_events.pkl")
ALIGN_SUMMARY = Path("phase3/data/alignment_summary.json")

HORIZONS = (1, 3, 5, 10, 20)


def fit_market_model(ri: np.ndarray, rm: np.ndarray) -> tuple[float, float]:
    """OLS alpha, beta strictly on pre-event [-120, -11] estimation window."""
    mask = np.isfinite(ri) & np.isfinite(rm)
    if mask.sum() < 60:
        return 0.0, 1.0
    xi = ri[mask]
    xm = rm[mask]
    var_m = float(np.var(xm))
    if var_m < 1e-12:
        return 0.0, 1.0
    beta = float(np.cov(xm, xi, ddof=0)[0, 1] / var_m)
    beta = float(np.clip(beta, -1.0, 3.5))
    alpha = float(np.mean(xi) - beta * np.mean(xm))
    return alpha, beta


def max_drawdown(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.0
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def main() -> None:
    print("=== Step 3: Aligning NSE Events to Actual Exchange Trading Sessions & Estimating Market Models ===", flush=True)
    with (MARKET_DIR / "indices.pkl").open("rb") as f:
        indices: dict[str, pd.DataFrame] = pickle.load(f)
    with (MARKET_DIR / "equities.pkl").open("rb") as f:
        equities: dict[str, pd.DataFrame] = pickle.load(f)
    with (MARKET_DIR / "fundamentals.pkl").open("rb") as f:
        fundamentals: dict[str, dict[str, Any]] = pickle.load(f)

    nifty = indices["NIFTY50"]
    trading_dates = nifty.index.to_numpy()  # sorted datetime64[ns] normalized dates
    nifty_close = nifty["Close"].to_numpy(dtype=np.float64)
    nifty_rets = np.empty(len(nifty_close), dtype=np.float64)
    nifty_rets[0] = np.nan
    nifty_rets[1:] = nifty_close[1:] / nifty_close[:-1] - 1.0

    # Pre-reindex each index onto trading_dates for fast O(1) lookup
    aligned_idx_close: dict[str, np.ndarray] = {}
    for k, idf in indices.items():
        if idf.empty or len(idf) < 200:
            continue
        re = idf["Close"].reindex(nifty.index).ffill()
        aligned_idx_close[k] = re.to_numpy(dtype=np.float64)

    # Pre-reindex each equity onto Nifty trading calendar (keeping mask of actual equity traded rows)
    eq_arrays: dict[str, dict[str, np.ndarray]] = {}
    for sym, df in equities.items():
        re_df = df.reindex(nifty.index)
        has_bar = re_df["Close_Adj"].notna().to_numpy()
        c_adj = re_df["Close_Adj"].ffill().to_numpy(dtype=np.float64)
        o_adj = re_df["Open_Adj"].fillna(re_df["Close_Adj"]).ffill().to_numpy(dtype=np.float64)
        vol = re_df["Volume"].fillna(0.0).to_numpy(dtype=np.float64)
        divs = re_df["Dividends"].fillna(0.0).to_numpy(dtype=np.float64) if "Dividends" in re_df.columns else np.zeros(len(c_adj))
        splits = re_df["Stock Splits"].fillna(0.0).to_numpy(dtype=np.float64) if "Stock Splits" in re_df.columns else np.zeros(len(c_adj))
        rets = np.empty(len(c_adj), dtype=np.float64)
        rets[0] = np.nan
        rets[1:] = c_adj[1:] / np.maximum(c_adj[:-1], 1e-12) - 1.0
        # Mask days where stock wasn't trading (stale ffill) in return series for OLS
        rets[~has_bar] = np.nan
        eq_arrays[sym] = {
            "has_bar": has_bar,
            "c_adj": c_adj,
            "o_adj": o_adj,
            "vol": vol,
            "divs": divs,
            "splits": splits,
            "rets": rets,
            "is_upstox": np.array([df.get("Provider", pd.Series(["yfinance"])).iloc[0] == "upstox_fallback"]),
        }

    raw_events = [json.loads(line) for line in EVENTS_PATH.open("r", encoding="utf-8") if line.strip()]
    last_ts_by_sym: dict[str, int] = {}
    aligned: list[dict[str, Any]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)

    for ev in raw_events:
        sym = ev["symbol"]
        if sym not in eq_arrays:
            skip_reasons["unresolved_symbol"] += 1
            continue
        arr = eq_arrays[sym]
        pub_dt = pd.Timestamp(ev["published_date"])
        bucket = ev["timing_bucket"]

        pos = int(np.searchsorted(trading_dates, pub_dt.to_datetime64(), side="left"))
        is_trading_day = pos < len(trading_dates) and pd.Timestamp(trading_dates[pos]) == pub_dt

        if bucket in ("pre_market", "intraday") and is_trading_day:
            t0_idx = pos
        elif is_trading_day and bucket == "post_market":
            t0_idx = pos + 1
        else:
            # Exchange holiday / weekend -> first trading day strictly after publication date
            t0_idx = int(np.searchsorted(trading_dates, pub_dt.to_datetime64(), side="right"))

        anchor_idx = t0_idx - 1
        if anchor_idx < 120 or t0_idx + 20 >= len(trading_dates):
            skip_reasons["insufficient_history_or_tail"] += 1
            continue
        # Ensure stock actually had valid observed bars at anchor, T0, and T0+20
        if not (arr["has_bar"][anchor_idx] and arr["has_bar"][t0_idx] and arr["has_bar"][t0_idx + 19]):
            skip_reasons["missing_stock_bar_at_boundary"] += 1
            continue

        # Executable entry point (strictly causally after publication timestamp!)
        if bucket == "intraday" and is_trading_day:
            # Market already open when announced: realistic execution at T0 close -> future sessions
            exec_entry_idx = t0_idx
            exec_entry_px = float(arr["c_adj"][t0_idx])
        else:
            # Pre-market or post-market/holiday: realistic execution at T0 open (or close if open missing)
            exec_entry_idx = t0_idx
            o_px = float(arr["o_adj"][t0_idx])
            exec_entry_px = o_px if o_px > 0 else float(arr["c_adj"][t0_idx])

        if exec_entry_px <= 0 or arr["c_adj"][anchor_idx] <= 0:
            skip_reasons["invalid_price"] += 1
            continue

        # Fit market model strictly on pre-event [-120, -11] before T0
        est_s = t0_idx - 120
        est_e = t0_idx - 10
        alpha, beta = fit_market_model(arr["rets"][est_s:est_e], nifty_rets[est_s:est_e])

        # Sector mapping
        sector = (fundamentals.get(sym) or {}).get("sector") or "Unknown"
        sec_key = SECTOR_TO_INDEX.get(sector, "NIFTY50")
        if sec_key not in aligned_idx_close:
            sec_key = "NIFTY50"
        sec_close = aligned_idx_close[sec_key]

        # Event-window reactions over H in (1, 3, 5, 10, 20)
        targets: dict[str, float] = {}
        anchor_px = float(arr["c_adj"][anchor_idx])
        anchor_m = float(nifty_close[anchor_idx])
        anchor_sec = float(sec_close[anchor_idx])

        for h in HORIZONS:
            end_idx = t0_idx + h - 1
            raw_r = float(arr["c_adj"][end_idx] / anchor_px - 1.0)
            m_r = float(nifty_close[end_idx] / anchor_m - 1.0)
            s_r = float(sec_close[end_idx] / anchor_sec - 1.0)
            bhar = raw_r - m_r
            sec_adj = raw_r - s_r
            # Market-model CAR[T0 .. T0+H-1]
            ri_win = arr["rets"][t0_idx : end_idx + 1]
            rm_win = nifty_rets[t0_idx : end_idx + 1]
            ar_win = np.where(np.isfinite(ri_win) & np.isfinite(rm_win), ri_win - (alpha + beta * rm_win), 0.0)
            car = float(np.sum(ar_win))

            # Realistic executable forward return starting strictly after publication
            if bucket == "intraday" and is_trading_day:
                ex_end = t0_idx + h
                ex_ret = float(arr["c_adj"][ex_end] / exec_entry_px - 1.0)
                ex_bench = float(nifty_close[ex_end] / nifty_close[t0_idx] - 1.0)
            else:
                ex_end = t0_idx + h - 1
                ex_ret = float(arr["c_adj"][ex_end] / exec_entry_px - 1.0)
                ex_bench = float(nifty_close[ex_end] / anchor_m - 1.0)
            # Clip extreme data anomalies (>300% single-window artifact) out? Better: flag & preserve
            targets[f"raw_ret_{h}d"] = round(raw_r, 6)
            targets[f"bench_ret_{h}d"] = round(m_r, 6)
            targets[f"bhar_{h}d"] = round(bhar, 6)
            targets[f"sec_adj_{h}d"] = round(sec_adj, 6)
            targets[f"car_{h}d"] = round(car, 6)
            targets[f"exec_ret_{h}d"] = round(ex_ret, 6)
            targets[f"exec_excess_{h}d"] = round(ex_ret - ex_bench, 6)
            targets[f"max_dd_{h}d"] = round(max_drawdown(arr["c_adj"][anchor_idx : end_idx + 1]), 6)

        pre_vol_20 = float(np.nanstd(arr["rets"][t0_idx - 20 : t0_idx]))
        post_vol_5 = float(np.nanstd(arr["rets"][t0_idx : t0_idx + 5]))
        post_vol_10 = float(np.nanstd(arr["rets"][t0_idx : t0_idx + 10]))
        mean_v20 = float(np.mean(arr["vol"][t0_idx - 20 : t0_idx]))
        vol_ratio_0 = float(arr["vol"][t0_idx] / mean_v20) if mean_v20 > 0 else 1.0
        vol_exp_5d = float(post_vol_5 / max(pre_vol_20, 1e-5))

        has_split = bool(np.any(arr["splits"][max(0, t0_idx - 5) : t0_idx + 20] != 0))
        has_div = bool(np.any(arr["divs"][max(0, t0_idx - 5) : t0_idx + 20] > 0))
        prev_t0 = last_ts_by_sym.get(sym, -999)
        is_overlap = (t0_idx - prev_t0) <= 10
        last_ts_by_sym[sym] = t0_idx

        # Filter obvious unadjusted corporate-action/data glitches (e.g. >75% single-day drop with split/upstox)
        if abs(targets["raw_ret_1d"]) > 0.65:
            skip_reasons["extreme_1d_jump_suspected_unadjusted_split"] += 1
            continue

        aligned.append({
            **ev,
            "sector": sector,
            "sector_index": sec_key,
            "t0_idx": t0_idx,
            "t0_date": pd.Timestamp(trading_dates[t0_idx]).strftime("%Y-%m-%d"),
            "anchor_idx": anchor_idx,
            "anchor_date": pd.Timestamp(trading_dates[anchor_idx]).strftime("%Y-%m-%d"),
            "exec_entry_idx": exec_entry_idx,
            "exec_entry_px": round(exec_entry_px, 4),
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "pre_vol_20d": round(pre_vol_20, 6),
            "post_vol_5d": round(post_vol_5, 6),
            "post_vol_10d": round(post_vol_10, 6),
            "vol_ratio_t0": round(vol_ratio_0, 4),
            "vol_expansion_5d": round(vol_exp_5d, 4),
            "has_split_near": has_split,
            "has_div_near": has_div,
            "corp_action_verified": not bool(arr["is_upstox"][0]),
            "is_overlapping_10d": is_overlap,
            "targets": targets,
        })

    with ALIGNED_OUT.open("wb") as f:
        pickle.dump(aligned, f)

    summary = {
        "input_events": len(raw_events),
        "aligned_events": len(aligned),
        "unique_symbols": len({a["symbol"] for a in aligned}),
        "t0_min": aligned[0]["t0_date"] if aligned else None,
        "t0_max": aligned[-1]["t0_date"] if aligned else None,
        "non_overlapping_10d_count": sum(not a["is_overlapping_10d"] for a in aligned),
        "corp_action_verified_count": sum(a["corp_action_verified"] for a in aligned),
        "skipped": dict(skip_reasons),
    }
    ALIGN_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Aligned {len(aligned)}/{len(raw_events)} events ({summary['t0_min']}..{summary['t0_max']}, {summary['unique_symbols']} issuers).", flush=True)
    print(f"Non-overlapping 10d clean events: {summary['non_overlapping_10d_count']}, Skipped: {dict(skip_reasons)}", flush=True)


if __name__ == "__main__":
    main()
