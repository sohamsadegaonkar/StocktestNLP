"""Multimodal Point-in-Time Feature Engineering (Phase 1+2 ONNX INT8 NLP + Technical + Lagged Fundamentals + Context)."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import json
from pathlib import Path
import pickle
import re
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from phase2.numeric_engine import parse_quantity
from phase2.schema import EVENT_MATURITY_LEVELS, IMPACT_IMPLICATIONS, MATERIALITY_LEVELS, STAKEHOLDER_GROUPS, UNCERTAINTY_LEVELS
from phase3.text_signals import MONEY_RE, NEG_TONE_RE, PCT_RE, POS_TONE_RE, sigmoid
from stockbrief.semantic_data import LABELS

ALIGNED_PATH = Path("phase3/.cache/aligned_events.pkl")
MARKET_DIR = Path("phase3/.cache/market")
FEATURES_OUT = Path("phase3/.cache/features_table.parquet")
SIM_RECS_OUT = Path("phase3/.cache/similarity_records.json")
EMB_OUT = Path("phase3/.cache/bge_embeddings.npy")
FEAT_META_OUT = Path("phase3/data/feature_manifest.json")


def ort_sess(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def batch_encode_ort(model_dir: Path, texts: list[str], max_length: int, batch_size: int = 64, take_cls: bool = False) -> np.ndarray:
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pad_id = tok.token_to_id("[PAD]") or 0
    tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    tok.enable_truncation(max_length=max_length)
    onnx_file = model_dir / "model_quantized.onnx"
    if not onnx_file.is_file():
        onnx_file = model_dir / "onnx" / "model_quantized.onnx"
    sess = ort_sess(onnx_file)
    in_names = {i.name for i in sess.get_inputs()}

    outs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for b_idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        encs = tok.encode_batch(texts[i : i + batch_size])
        feed = {
            "input_ids": np.asarray([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encs], dtype=np.int64),
        }
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.asarray([e.type_ids for e in encs], dtype=np.int64)
        res = sess.run(None, feed)[0]
        if take_cls and res.ndim == 3:
            res = res[:, 0, :]
        outs.append(res)
        if b_idx % 100 == 0 or b_idx == n_batches:
            print(f"    [{model_dir.name}] batch {b_idx}/{n_batches}", flush=True)
    return np.concatenate(outs, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = pd.Series(gains).rolling(period, min_periods=period).mean().to_numpy()
    avg_l = pd.Series(losses).rolling(period, min_periods=period).mean().to_numpy()
    rs = avg_g / np.maximum(avg_l, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def causal_regime(nifty_ret_60d: float, vix_lvl: float) -> str:
    if vix_lvl >= 18.0:
        return "high_vol"
    if nifty_ret_60d >= 0.04:
        return "bull"
    if nifty_ret_60d <= -0.04:
        return "bear"
    return "sideways"


def extract_pit_fundamentals(sym: str, anchor_date: str, anchor_px: float, fund_entry: dict[str, Any]) -> dict[str, float]:
    """Point-in-time financial statement join using strict +60 calendar day statutory reporting lag."""
    out = {
        "has_pit_fundamentals": 0.0,
        "fund_rev_yoy": 0.0,
        "fund_ni_yoy": 0.0,
        "fund_op_margin": 0.0,
        "fund_net_margin": 0.0,
        "fund_debt_to_equity": 0.0,
        "fund_eps": 0.0,
        "fund_ocf_to_ni": 0.0,
        "fund_pe_proxy": 0.0,
    }
    if not fund_entry:
        return out
    cutoff = pd.Timestamp(anchor_date)

    # Combine quarterly & annual snapshots with conservative +60d availability lag
    snapshots: list[tuple[pd.Timestamp, dict[str, float], str]] = []
    for kind in ("quarterly", "annual"):
        st = fund_entry.get(kind) or {}
        inc = st.get("inc") or {}
        bs = st.get("bs") or {}
        cf = st.get("cf") or {}
        for p_str, ivals in inc.items():
            p_dt = pd.Timestamp(p_str)
            avail_dt = p_dt + timedelta(days=60)
            if avail_dt <= cutoff:
                merged = {**ivals, **(bs.get(p_str) or {}), **(cf.get(p_str) or {})}
                snapshots.append((p_dt, merged, kind))
    if not snapshots:
        return out

    snapshots.sort(key=lambda x: x[0])
    latest_dt, latest, latest_kind = snapshots[-1]
    out["has_pit_fundamentals"] = 1.0

    rev = latest.get("Total Revenue") or latest.get("Operating Revenue") or 0.0
    ni = latest.get("Net Income") or 0.0
    oi = latest.get("Operating Income") or latest.get("EBITDA") or 0.0
    debt = latest.get("Total Debt") or 0.0
    eq = latest.get("Stockholders Equity") or 0.0
    ocf = latest.get("Operating Cash Flow") or 0.0
    eps = latest.get("Diluted EPS") or latest.get("Basic EPS") or 0.0

    if abs(rev) > 1e-6:
        out["fund_op_margin"] = float(np.clip(oi / rev, -2.0, 2.0))
        out["fund_net_margin"] = float(np.clip(ni / rev, -2.0, 2.0))
    if abs(eq) > 1e-6:
        out["fund_debt_to_equity"] = float(np.clip(debt / abs(eq), 0.0, 20.0))
    if abs(ni) > 1e-6:
        out["fund_ocf_to_ni"] = float(np.clip(ocf / abs(ni), -5.0, 5.0))
    out["fund_eps"] = float(np.clip(eps, -500.0, 2000.0))
    if eps > 0.5 and anchor_px > 0:
        ann_eps = eps * (4.0 if latest_kind == "quarterly" else 1.0)
        out["fund_pe_proxy"] = float(np.clip(anchor_px / ann_eps, 0.0, 200.0))

    # YoY growth from matching ~1 year earlier period of same kind
    target_prev = latest_dt - timedelta(days=365)
    for p_dt, prev, k in reversed(snapshots[:-1]):
        if k == latest_kind and abs((p_dt - target_prev).days) <= 45:
            prev_rev = prev.get("Total Revenue") or prev.get("Operating Revenue") or 0.0
            prev_ni = prev.get("Net Income") or 0.0
            if prev_rev > 1e-6:
                out["fund_rev_yoy"] = float(np.clip((rev - prev_rev) / prev_rev, -1.0, 3.0))
            if prev_ni > 1e-6:
                out["fund_ni_yoy"] = float(np.clip((ni - prev_ni) / prev_ni, -2.0, 5.0))
            break
    return out


def main() -> None:
    t0 = time.perf_counter()
    FEAT_META_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ALIGNED_PATH.open("rb") as f:
        aligned: list[dict[str, Any]] = pickle.load(f)
    with (MARKET_DIR / "indices.pkl").open("rb") as f:
        indices: dict[str, pd.DataFrame] = pickle.load(f)
    with (MARKET_DIR / "equities.pkl").open("rb") as f:
        equities: dict[str, pd.DataFrame] = pickle.load(f)
    with (MARKET_DIR / "fundamentals.pkl").open("rb") as f:
        fundamentals: dict[str, dict[str, Any]] = pickle.load(f)

    print(f"=== Step 4: Engineering Multimodal Point-in-Time Features for {len(aligned)} Events ===", flush=True)

    # 1. Neural NLP inference using Phase 1 semantic + Phase 2 Domain-FT BGE + Phase 2 Impact ONNX INT8
    texts = [ev["text"][:800] for ev in aligned]
    if EMB_OUT.is_file():
        embs = np.load(EMB_OUT)
        if len(embs) == len(texts):
            print(f"  1a. Reusing cached Domain-FT BGE embeddings {embs.shape}.", flush=True)
        else:
            embs = batch_encode_ort(Path("models/phase2/bge_retrieval_ft"), texts, max_length=128, batch_size=64)
            embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
            np.save(EMB_OUT, embs.astype(np.float32))
    else:
        print("  1a. Extracting Domain-FT BGE 384-d embeddings (ONNX INT8)...", flush=True)
        embs = batch_encode_ort(Path("models/phase2/bge_retrieval_ft"), texts, max_length=128, batch_size=64)
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        np.save(EMB_OUT, embs.astype(np.float32))

    # Phase 1 semantic category inference
    print("  1b. Extracting Phase 1 event category probabilities + Phase 2 impact logits (ONNX INT8)...", flush=True)
    p1_cache = Path("phase3/.cache/p1_embs.npy")
    if p1_cache.is_file() and len(np.load(p1_cache, mmap_mode="r")) == len(texts):
        p1_embs = np.load(p1_cache)
    else:
        p1_embs = batch_encode_ort(Path("models/semantic"), texts, max_length=128, batch_size=64, take_cls=True)
        p1_embs = p1_embs / np.maximum(np.linalg.norm(p1_embs, axis=1, keepdims=True), 1e-12)
        np.save(p1_cache, p1_embs.astype(np.float32))

    from stockbrief.semantic import load_semantic
    sem_model = load_semantic(Path("models/semantic"))
    p1_probs = sem_model.head.predict_proba(p1_embs)
    p1_classes = list(sem_model.head.classes_)
    p1_top_cat = [p1_classes[int(i)] for i in np.argmax(p1_probs, axis=1)]

    # Impact ONNX INT8 inputs formatted identically to train_impact: "[category | stage] text"
    imp_cache = Path("phase3/.cache/imp_logits.npy")
    if imp_cache.is_file() and len(np.load(imp_cache, mmap_mode="r")) == len(texts):
        imp_logits = np.load(imp_cache)
    else:
        imp_texts = [f"[{cat} | Reported] {ev['text'][:600]}" for cat, ev in zip(p1_top_cat, aligned)]
        imp_logits = batch_encode_ort(Path("models/phase2/impact_bge"), imp_texts, max_length=128, batch_size=64)
        np.save(imp_cache, imp_logits.astype(np.float32))

    i1 = len(IMPACT_IMPLICATIONS)
    i2 = i1 + len(MATERIALITY_LEVELS)
    i3 = i2 + len(EVENT_MATURITY_LEVELS)
    i4 = i3 + len(UNCERTAINTY_LEVELS)

    imp_probs = sigmoid(imp_logits[:, :i1])
    mat_exp = np.exp(imp_logits[:, i1:i2] - np.max(imp_logits[:, i1:i2], axis=1, keepdims=True))
    mat_probs = mat_exp / mat_exp.sum(axis=1, keepdims=True)
    evm_exp = np.exp(imp_logits[:, i2:i3] - np.max(imp_logits[:, i2:i3], axis=1, keepdims=True))
    evm_probs = evm_exp / evm_exp.sum(axis=1, keepdims=True)
    unc_exp = np.exp(imp_logits[:, i3:i4] - np.max(imp_logits[:, i3:i4], axis=1, keepdims=True))
    unc_probs = unc_exp / unc_exp.sum(axis=1, keepdims=True)
    stk_probs = sigmoid(imp_logits[:, i4:])

    # 2. Precompute Nifty, Sector & Equity technical series cleanly
    print("  2. Computing pre-event technical, market-context & lagged fundamental features...", flush=True)
    nifty = indices["NIFTY50"]
    n_dates = nifty.index
    n_close = nifty["Close"].to_numpy(dtype=np.float64)
    n_rets = pd.Series(n_close).pct_change().fillna(0.0).to_numpy()
    n_vol20 = pd.Series(n_rets).rolling(20, min_periods=10).std().fillna(0.01).to_numpy()
    vix_close = indices["INDIAVIX"]["Close"].reindex(n_dates).ffill().bfill().to_numpy(dtype=np.float64)

    sec_close_map = {
        k: idf["Close"].reindex(n_dates).ffill().bfill().to_numpy(dtype=np.float64)
        for k, idf in indices.items()
        if not idf.empty and len(idf) >= 200 and k not in ("NIFTY50", "INDIAVIX")
    }

    eq_tech: dict[str, dict[str, np.ndarray]] = {}
    for sym, df in equities.items():
        re = df.reindex(n_dates)
        c = re["Close_Adj"].ffill().to_numpy(dtype=np.float64)
        v = re["Volume"].fillna(0.0).to_numpy(dtype=np.float64)
        r = pd.Series(c).pct_change().fillna(0.0).to_numpy()
        sma20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma50 = pd.Series(c).rolling(50, min_periods=20).mean().to_numpy()
        sma200 = pd.Series(c).rolling(200, min_periods=60).mean().to_numpy()
        hi252 = pd.Series(c).rolling(252, min_periods=60).max().to_numpy()
        vol20 = pd.Series(r).rolling(20, min_periods=10).std().fillna(0.02).to_numpy()
        vol60 = pd.Series(r).rolling(60, min_periods=20).std().fillna(0.02).to_numpy()
        vmean5 = pd.Series(v).rolling(5, min_periods=1).mean().to_numpy()
        vmean20 = pd.Series(v).rolling(20, min_periods=5).mean().to_numpy()
        turn20 = pd.Series(c * v).rolling(20, min_periods=5).mean().to_numpy()
        eq_tech[sym] = {
            "c": c,
            "rsi14": compute_rsi(c, 14),
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "hi252": hi252,
            "vol20": vol20,
            "vol60": vol60,
            "vmean5": vmean5,
            "vmean20": vmean20,
            "turn20": turn20,
            "rets": r,
        }

    # Precompute causal news flow count per symbol (events strictly prior to current event timestamp)
    sym_history_dates: defaultdict[str, list[pd.Timestamp]] = defaultdict(list)

    rows_out: list[dict[str, Any]] = []
    for idx, ev in enumerate(aligned):
        sym = ev["symbol"]
        a_idx = ev["anchor_idx"]
        t0_idx = ev["t0_idx"]
        t = eq_tech[sym]
        c_a = float(t["c"][a_idx])
        if c_a <= 0:
            continue

        cur_ts = pd.Timestamp(ev["published_ts_ist"])
        prev_dates = sym_history_dates[sym]
        prior_5d = sum(1 for d in prev_dates if 0 < (cur_ts - d).total_seconds() <= 5 * 86400)
        prior_30d = sum(1 for d in prev_dates if 0 < (cur_ts - d).total_seconds() <= 30 * 86400)
        prev_dates.append(cur_ts)

        def safe_ret(arr: np.ndarray, lag: int) -> float:
            if a_idx - lag < 0 or arr[a_idx - lag] <= 0:
                return 0.0
            return float(arr[a_idx] / arr[a_idx - lag] - 1.0)

        # Market & sector context strictly at anchor_idx (<= T0 - 1)
        m_r5 = safe_ret(n_close, 5)
        m_r20 = safe_ret(n_close, 20)
        m_r60 = safe_ret(n_close, 60)
        vix_lvl = float(vix_close[a_idx])
        vix_chg5 = safe_ret(vix_close, 5)
        vix_chg20 = safe_ret(vix_close, 20)
        reg = causal_regime(m_r60, vix_lvl)

        sec_key = ev["sector_index"]
        sec_arr = sec_close_map.get(sec_key, n_close)
        s_r5 = safe_ret(sec_arr, 5)
        s_r20 = safe_ret(sec_arr, 20)

        # Pre-event AR[-10, -2] drift strictly before T0
        pre_ar_win = t["rets"][max(0, t0_idx - 10) : max(0, t0_idx - 1)] - (ev["alpha"] + ev["beta"] * n_rets[max(0, t0_idx - 10) : max(0, t0_idx - 1)])
        pre_car_10_2 = float(np.sum(pre_ar_win))

        # Deterministic numeric & lexical signals from disclosure text
        txt = ev["text"]
        money_hits = MONEY_RE.findall(txt)
        max_inr_crore = 0.0
        for mh in money_hits[:5]:
            q = parse_quantity(mh)
            if q and q.normalized_unit == "INR_crore":
                max_inr_crore = max(max_inr_crore, abs(q.normalized_value))
        pct_hits = len(PCT_RE.findall(txt))
        neg_hits = len(NEG_TONE_RE.findall(txt))
        pos_hits = len(POS_TONE_RE.findall(txt))

        # Point-in-time lagged fundamentals
        pit_fund = extract_pit_fundamentals(sym, ev["anchor_date"], c_a, fundamentals.get(sym) or {})

        rec: dict[str, Any] = {
            "event_id": ev["event_id"],
            "symbol": sym,
            "sector": ev["sector"],
            "sector_index": sec_key,
            "published_ts_ist": ev["published_ts_ist"],
            "published_date": ev["published_date"],
            "t0_date": ev["t0_date"],
            "anchor_date": ev["anchor_date"],
            "t0_idx": t0_idx,
            "timing_bucket": ev["timing_bucket"],
            "event_family_hint": ev["event_family_hint"],
            "source_url": ev["source_url"],
            "text": ev["text"][:500],
            "corp_action_verified": int(ev["corp_action_verified"]),
            "is_overlapping_10d": int(ev["is_overlapping_10d"]),
            "regime_causal": reg,
            "emb_row_idx": idx,
            # Targets (never used as input features!)
            **{f"target_{k}": v for k, v in ev["targets"].items()},
            "target_vol_expansion_5d": float(ev["vol_expansion_5d"]),
            "target_vol_expand_cls": int(ev["vol_expansion_5d"] > 1.25),
            # --- Feature Group 1: Financial NLP ---
            "nlp_p1_top_conf": float(np.max(p1_probs[idx])),
            "nlp_has_xbrl": float(ev["has_xbrl"]),
            "nlp_log_inr_crore": float(np.log1p(max_inr_crore)),
            "nlp_money_count": float(len(money_hits)),
            "nlp_pct_count": float(pct_hits),
            "nlp_pos_lex": float(pos_hits),
            "nlp_neg_lex": float(neg_hits),
            "nlp_headline_len": float(len(txt)),
            "nlp_mat_score": float(np.dot(mat_probs[idx], np.array([3.0, 2.0, 1.0, 0.0]))),
            "nlp_unc_score": float(np.dot(unc_probs[idx], np.array([0.0, 1.0, 2.0]))),
            # --- Feature Group 2: Technical Market ---
            "tech_ret_1d": safe_ret(t["c"], 1),
            "tech_ret_3d": safe_ret(t["c"], 3),
            "tech_ret_5d": safe_ret(t["c"], 5),
            "tech_ret_10d": safe_ret(t["c"], 10),
            "tech_ret_20d": safe_ret(t["c"], 20),
            "tech_ret_60d": safe_ret(t["c"], 60),
            "tech_pre_car_m10_m2": pre_car_10_2,
            "tech_dist_sma20": float(c_a / max(t["sma20"][a_idx], 1e-6) - 1.0) if np.isfinite(t["sma20"][a_idx]) else 0.0,
            "tech_dist_sma50": float(c_a / max(t["sma50"][a_idx], 1e-6) - 1.0) if np.isfinite(t["sma50"][a_idx]) else 0.0,
            "tech_dist_sma200": float(c_a / max(t["sma200"][a_idx], 1e-6) - 1.0) if np.isfinite(t["sma200"][a_idx]) else 0.0,
            "tech_dist_52w_hi": float(c_a / max(t["hi252"][a_idx], 1e-6) - 1.0) if np.isfinite(t["hi252"][a_idx]) else 0.0,
            "tech_rsi14": float(t["rsi14"][a_idx]),
            "tech_vol20": float(t["vol20"][a_idx]),
            "tech_vol60": float(t["vol60"][a_idx]),
            "tech_vol_ratio_20_60": float(t["vol20"][a_idx] / max(t["vol60"][a_idx], 1e-5)),
            "tech_vol_surge_5_20": float(t["vmean5"][a_idx] / max(t["vmean20"][a_idx], 1.0)),
            "tech_log_turnover20": float(np.log1p(max(0.0, t["turn20"][a_idx]))),
            "tech_alpha": float(ev["alpha"]),
            "tech_beta": float(ev["beta"]),
            # --- Feature Group 3: Point-in-Time Fundamentals ---
            **pit_fund,
            # --- Feature Group 4: Market Context ---
            "ctx_nifty_ret_5d": m_r5,
            "ctx_nifty_ret_20d": m_r20,
            "ctx_nifty_ret_60d": m_r60,
            "ctx_nifty_vol_20d": float(n_vol20[a_idx]),
            "ctx_sector_ret_5d": s_r5,
            "ctx_sector_ret_20d": s_r20,
            "ctx_stock_vs_sector_20d": safe_ret(t["c"], 20) - s_r20,
            "ctx_stock_vs_nifty_20d": safe_ret(t["c"], 20) - m_r20,
            "ctx_vix_level": vix_lvl,
            "ctx_vix_chg_5d": vix_chg5,
            "ctx_vix_chg_20d": vix_chg20,
            "ctx_prior_events_5d": float(prior_5d),
            "ctx_prior_events_30d": float(prior_30d),
            "ctx_timing_pre": 1.0 if ev["timing_bucket"] == "pre_market" else 0.0,
            "ctx_timing_intraday": 1.0 if ev["timing_bucket"] == "intraday" else 0.0,
            "ctx_timing_post": 1.0 if ev["timing_bucket"] == "post_market" else 0.0,
            "ctx_is_overlap_10d": float(ev["is_overlapping_10d"]),
        }

        for j, cname in enumerate(p1_classes):
            rec[f"nlp_p1_cat_{cname}"] = float(p1_probs[idx, j])
        for j, cname in enumerate(IMPACT_IMPLICATIONS):
            rec[f"nlp_imp_{cname}"] = float(imp_probs[idx, j])
        for j, cname in enumerate(MATERIALITY_LEVELS):
            rec[f"nlp_mat_{cname}"] = float(mat_probs[idx, j])
        for j, cname in enumerate(EVENT_MATURITY_LEVELS):
            rec[f"nlp_evm_{cname}"] = float(evm_probs[idx, j])
        for j, cname in enumerate(UNCERTAINTY_LEVELS):
            rec[f"nlp_unc_{cname}"] = float(unc_probs[idx, j])
        for j, cname in enumerate(STAKEHOLDER_GROUPS):
            rec[f"nlp_stk_{cname}"] = float(stk_probs[idx, j])

        rows_out.append(rec)

    df_feat = pd.DataFrame(rows_out)
    df_feat.to_parquet(FEATURES_OUT, index=False)
    sim_cols = [
        "event_id", "symbol", "sector", "published_ts_ist", "t0_date", "event_family_hint",
        "regime_causal", "source_url", "text", "emb_row_idx", "nlp_log_inr_crore", "nlp_mat_score",
        "target_raw_ret_1d", "target_bhar_1d", "target_raw_ret_5d", "target_bhar_5d",
        "target_car_5d", "target_bhar_10d", "target_bhar_20d", "target_vol_expansion_5d",
    ]
    SIM_RECS_OUT.write_text(json.dumps(df_feat[sim_cols].to_dict(orient="records"), ensure_ascii=False), encoding="utf-8")

    manifest = {
        "rows": len(df_feat),
        "unique_symbols": int(df_feat["symbol"].nunique()),
        "pit_fundamentals_coverage": round(float(df_feat["has_pit_fundamentals"].mean()), 4),
        "regime_counts": df_feat["regime_causal"].value_counts().to_dict(),
        "nlp_feature_count": len([c for c in df_feat.columns if c.startswith("nlp_")]),
        "tech_feature_count": len([c for c in df_feat.columns if c.startswith("tech_")]),
        "fund_feature_count": len([c for c in df_feat.columns if c.startswith("fund_") or c == "has_pit_fundamentals"]),
        "ctx_feature_count": len([c for c in df_feat.columns if c.startswith("ctx_")]),
        "target_columns": [c for c in df_feat.columns if c.startswith("target_")],
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    FEAT_META_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Features built: {manifest['rows']} rows in {manifest['elapsed_sec']}s | PIT fundamentals coverage={manifest['pit_fundamentals_coverage']}", flush=True)
    print("Feature counts:", {k: manifest[k] for k in ("nlp_feature_count", "tech_feature_count", "fund_feature_count", "ctx_feature_count")}, flush=True)
    print("Regimes:", manifest["regime_counts"], flush=True)


if __name__ == "__main__":
    main()
