"""Phase 3 Edge Intelligence Adapter (Pure ONNX Runtime, Strict Fact/Prediction Separation)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from phase2.numeric_engine import parse_quantity
from phase2.schema import EVENT_MATURITY_LEVELS, IMPACT_IMPLICATIONS, MATERIALITY_LEVELS, STAKEHOLDER_GROUPS, UNCERTAINTY_LEVELS
from phase3.text_signals import NEG_TONE_RE, PCT_RE, POS_TONE_RE, MONEY_RE, sigmoid
from phase3.similarity_engine import HistoricalSimilarityEngine
from stockbrief.semantic import load_semantic

IST = timezone(timedelta(hours=5, minutes=30))


class MarketIntelligenceEngine:
    """Offline-capable CPU edge engine combining Phase 1/2 ONNX NLP + Historical Similarity + Phase 3 ONNX reaction models."""

    version = "stockbrief-phase3-market-intel-v1"

    def __init__(self, models_root: Path = Path("models"), results_root: Path = Path("phase3/results")):
        p3 = models_root / "phase3"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        # Phase 1 semantic classifier (also supplies base BGE embedding)
        self.sem_model = load_semantic(models_root / "semantic")

        # Phase 2 Domain-FT BGE & Contextual Impact ONNX INT8
        bge_dir = models_root / "phase2/bge_retrieval_ft"
        self.bge_tok = Tokenizer.from_file(str(bge_dir / "tokenizer.json"))
        self.bge_tok.enable_padding(pad_id=self.bge_tok.token_to_id("[PAD]") or 0)
        self.bge_tok.enable_truncation(max_length=128)
        self.bge_sess = ort.InferenceSession(str(bge_dir / "model_quantized.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

        imp_dir = models_root / "phase2/impact_bge"
        self.imp_tok = Tokenizer.from_file(str(imp_dir / "tokenizer.json"))
        self.imp_tok.enable_padding(pad_id=self.imp_tok.token_to_id("[PAD]") or 0)
        self.imp_tok.enable_truncation(max_length=128)
        self.imp_sess = ort.InferenceSession(str(imp_dir / "model_quantized.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

        # Phase 3 CatBoost ONNX models & pure NumPy/JSON SVD/column metadata (zero catboost/pickle dependency)
        self.dir_sess = ort.InferenceSession(str(p3 / "catboost_direction_combined.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.vol_sess = ort.InferenceSession(str(p3 / "catboost_volatility_combined.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        schema = json.loads((p3 / "feature_schema.json").read_text(encoding="utf-8"))
        self.svd_components: np.ndarray = np.load(p3 / "svd_components.npy").astype(np.float32)
        self.mkt_cols: list[str] = schema["mkt_cols"]
        self.nlp_cols: list[str] = schema["nlp_cols"]
        self.train_cent: np.ndarray = np.load(p3 / "train_centroid.npy").astype(np.float32)

        self.similarity = HistoricalSimilarityEngine()
        mod_rep_path = results_root / "modeling_report.json"
        self.nlp_lift_validated = False
        if mod_rep_path.is_file():
            self.nlp_lift_validated = bool(json.loads(mod_rep_path.read_text(encoding="utf-8")).get("nlp_lift_validated", json.loads(mod_rep_path.read_text(encoding="utf-8")).get("nlp_lift_established", False)))

    def _encode_bge(self, tok: Tokenizer, sess: ort.InferenceSession, text: str) -> np.ndarray:
        enc = tok.encode(text)
        feed = {"input_ids": np.asarray([enc.ids], dtype=np.int64), "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64)}
        in_names = {i.name for i in sess.get_inputs()}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.asarray([enc.type_ids], dtype=np.int64)
        out = sess.run(None, feed)[0][0]
        return out

    def analyze(
        self,
        *,
        text: str,
        symbol: str,
        sector: str = "Unknown",
        event_family_hint: str = "",
        source_url: str = "",
        as_of_ts_ist: str | None = None,
        market_snapshot: dict[str, float] | None = None,
        fundamentals_snapshot: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter_ns()
        ts_now = as_of_ts_ist or datetime.now(IST).isoformat()
        mkt = market_snapshot or {}
        fnd = fundamentals_snapshot or {}

        # 1. Phase 1 + Phase 2 NLP features
        p1_emb = self.sem_model.encoder.encode([text[:800]])
        p1_probs = self.sem_model.head.predict_proba(p1_emb)[0]
        p1_classes = list(self.sem_model.head.classes_)
        top_cat = p1_classes[int(np.argmax(p1_probs))]

        bge_vec = self._encode_bge(self.bge_tok, self.bge_sess, text[:800])
        bge_vec = bge_vec / max(float(np.linalg.norm(bge_vec)), 1e-12)

        imp_lg = self._encode_bge(self.imp_tok, self.imp_sess, f"[{top_cat} | Reported] {text[:600]}")
        i1 = len(IMPACT_IMPLICATIONS)
        i2 = i1 + len(MATERIALITY_LEVELS)
        i3 = i2 + len(EVENT_MATURITY_LEVELS)
        i4 = i3 + len(UNCERTAINTY_LEVELS)
        p_imp = sigmoid(imp_lg[:i1])
        p_mat = np.exp(imp_lg[i1:i2] - np.max(imp_lg[i1:i2])); p_mat /= p_mat.sum()
        p_evm = np.exp(imp_lg[i2:i3] - np.max(imp_lg[i2:i3])); p_evm /= p_evm.sum()
        p_unc = np.exp(imp_lg[i3:i4] - np.max(imp_lg[i3:i4])); p_unc /= p_unc.sum()
        p_stk = sigmoid(imp_lg[i4:])

        money_hits = MONEY_RE.findall(text)
        max_inr_crore = 0.0
        for mh in money_hits[:5]:
            q = parse_quantity(mh)
            if q and q.normalized_unit == "INR_crore":
                max_inr_crore = max(max_inr_crore, abs(q.normalized_value))

        # 2. Retrieve strict prior historical precedents
        regime = "high_vol" if mkt.get("ctx_vix_level", 14.0) >= 18.0 else ("bull" if mkt.get("ctx_nifty_ret_60d", 0.0) >= 0.04 else ("bear" if mkt.get("ctx_nifty_ret_60d", 0.0) <= -0.04 else "sideways"))
        sim_res = self.similarity.retrieve(
            bge_vec,
            as_of_ts_ist=ts_now,
            event_family=event_family_hint or top_cat,
            sector=sector,
            regime_causal=regime,
            log_inr_crore=float(np.log1p(max_inr_crore)),
            top_k=8,
        )

        # 3. Assemble combined feature vector aligned strictly to mkt_cols + nlp_cols
        feat_map: dict[str, float] = {**mkt, **fnd}
        feat_map.update({
            "nlp_p1_top_conf": float(np.max(p1_probs)),
            "nlp_has_xbrl": 0.0,
            "nlp_log_inr_crore": float(np.log1p(max_inr_crore)),
            "nlp_money_count": float(len(money_hits)),
            "nlp_pct_count": float(len(PCT_RE.findall(text))),
            "nlp_pos_lex": float(len(POS_TONE_RE.findall(text))),
            "nlp_neg_lex": float(len(NEG_TONE_RE.findall(text))),
            "nlp_headline_len": float(len(text)),
            "nlp_mat_score": float(np.dot(p_mat, np.array([3.0, 2.0, 1.0, 0.0]))),
            "nlp_unc_score": float(np.dot(p_unc, np.array([0.0, 1.0, 2.0]))),
        })
        for j, cname in enumerate(p1_classes):
            feat_map[f"nlp_p1_cat_{cname}"] = float(p1_probs[j])
        for j, cname in enumerate(IMPACT_IMPLICATIONS):
            feat_map[f"nlp_imp_{cname}"] = float(p_imp[j])
        for j, cname in enumerate(MATERIALITY_LEVELS):
            feat_map[f"nlp_mat_{cname}"] = float(p_mat[j])
        for j, cname in enumerate(EVENT_MATURITY_LEVELS):
            feat_map[f"nlp_evm_{cname}"] = float(p_evm[j])
        for j, cname in enumerate(UNCERTAINTY_LEVELS):
            feat_map[f"nlp_unc_{cname}"] = float(p_unc[j])
        for j, cname in enumerate(STAKEHOLDER_GROUPS):
            feat_map[f"nlp_stk_{cname}"] = float(p_stk[j])
        svd_vals = bge_vec @ self.svd_components.T
        for j, val in enumerate(svd_vals):
            feat_map[f"nlp_svd_{j}"] = float(val)

        x_vec = np.asarray([[feat_map.get(c, 0.0) for c in (self.mkt_cols + self.nlp_cols)]], dtype=np.float32)

        # 4. Run ONNX directional & volatility models
        dir_out = self.dir_sess.run(None, {self.dir_sess.get_inputs()[0].name: x_vec})
        prob_dict = dir_out[1][0] if isinstance(dir_out[1][0], dict) else {0: float(dir_out[1][0][0]), 1: float(dir_out[1][0][1]), 2: float(dir_out[1][0][2])}
        p_neg, p_neut, p_pos = float(prob_dict[0]), float(prob_dict[1]), float(prob_dict[2])

        vol_out = self.vol_sess.run(None, {self.vol_sess.get_inputs()[0].name: x_vec})
        vol_dict = vol_out[1][0] if isinstance(vol_out[1][0], dict) else {0: float(vol_out[1][0][0]), 1: float(vol_out[1][0][1])}
        p_vol_exp = float(vol_dict[1])

        # Uncertainty / abstention check
        cent_cos = float(bge_vec @ self.train_cent)
        sorted_p = sorted([p_neg, p_neut, p_pos])
        margin = sorted_p[-1] - sorted_p[-2]
        abstain = False
        abstain_reason = None
        if cent_cos < 0.30:
            abstain = True
            abstain_reason = "out_of_distribution_text"
        elif sim_res["historical_distribution"] is None or sim_res["historical_distribution"]["sample_size"] < 3:
            abstain = True
            abstain_reason = "sparse_historical_evidence"
        elif margin < 0.03:
            abstain = True
            abstain_reason = "low_directional_confidence_margin"

        hist_dist = sim_res["historical_distribution"]
        unc_range = (
            {"q10_excess_5d": hist_dist["bhar_5d_q10"], "q90_excess_5d": hist_dist["bhar_5d_q90"]}
            if hist_dist
            else {"q10_excess_5d": None, "q90_excess_5d": None}
        )

        implications = [IMPACT_IMPLICATIONS[i] for i, pv in enumerate(p_imp) if pv >= 0.45] or [IMPACT_IMPLICATIONS[int(np.argmax(p_imp))]]
        stakeholders = [STAKEHOLDER_GROUPS[i] for i, pv in enumerate(p_stk) if pv >= 0.45] or ["shareholders"]

        return {
            "model_version": self.version,
            "analysis_timestamp": ts_now,
            "event_summary": {
                "symbol": symbol,
                "sector": sector,
                "predicted_category": top_cat,
                "category_confidence": round(float(np.max(p1_probs)), 4),
                "extracted_max_inr_crore": round(max_inr_crore, 3),
                "headline_excerpt": text[:240],
            },
            "affected_companies": [{"symbol": symbol, "sector": sector, "stakeholders": stakeholders}],
            "potential_implications": {
                "implications": implications,
                "materiality": MATERIALITY_LEVELS[int(np.argmax(p_mat))],
                "event_maturity": EVENT_MATURITY_LEVELS[int(np.argmax(p_evm))],
                "uncertainty": UNCERTAINTY_LEVELS[int(np.argmax(p_unc))],
            },
            "historical_facts": {
                "historical_comparable_events": sim_res["comparables"],
                "observed_historical_return_distribution": hist_dist,
                "supporting_sources": [{"url": source_url}] + [{"url": c["source_url"], "event_id": c["event_id"]} for c in sim_res["comparables"] if c.get("source_url")],
                "market_data_notice": "Historical OHLCV & comparables come from cached daily datasets only; never represented as live prices or causal effects.",
            },
            "predictions": {
                "prediction_horizon": "5 trading sessions (benchmark-adjusted excess return, +/-1.5% thresholds)",
                "directional_probabilities_status": "calibrated_conditional" if self.nlp_lift_validated else "experimental_unvalidated_above_market_baseline",
                "directional_probabilities": None if abstain else {
                    "negative_lt_-1.5pct": round(p_neg, 4),
                    "neutral_-1.5_to_+1.5pct": round(p_neut, 4),
                    "positive_gt_+1.5pct": round(p_pos, 4),
                },
                "volatility_expansion_gt_1_25x_prob": round(p_vol_exp, 4),
                "expected_uncertainty_range": unc_range,
                "embedding_centroid_similarity": round(cent_cos, 4),
                "abstain": abstain,
                "abstain_reason": abstain_reason,
            },
            "chart_data_contracts": {
                "probability_distribution": [
                    {"bucket": "Negative (< -1.5%)", "prob": round(p_neg, 4)},
                    {"bucket": "Neutral (-1.5% to +1.5%)", "prob": round(p_neut, 4)},
                    {"bucket": "Positive (> +1.5%)", "prob": round(p_pos, 4)},
                ],
                "historical_event_distribution": hist_dist,
            },
            "important_limitations": [
                "Observed historical reactions reflect statistical associations, never proven causation.",
                "Cached historical market closes may lag live exchange ticks and exclude intraday slippage beyond simulation bounds.",
                "Outputs are conditional analytical estimates, never investment advice, price targets, or buy/sell recommendations.",
            ],
            "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
        }
