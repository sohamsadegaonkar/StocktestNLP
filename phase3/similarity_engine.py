"""Point-in-Time Historical Event Similarity Engine (Zero-Pandas Pure-NumPy Edge Runtime)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SIM_RECS_PATH = Path("phase3/.cache/similarity_records.json")
EMB_PATH = Path("phase3/.cache/bge_embeddings.npy")


class HistoricalSimilarityEngine:
    """Retrieves strictly prior comparable NSE disclosures with observed return distributions."""

    def __init__(self, records_path: Path = SIM_RECS_PATH, emb_path: Path = EMB_PATH):
        self.records: list[dict[str, Any]] = json.loads(records_path.read_text(encoding="utf-8"))
        self.embs = np.load(emb_path).astype(np.float32)
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        self.embs = self.embs / np.maximum(norms, 1e-12)

        self.event_ids = np.asarray([r["event_id"] for r in self.records])
        self.emb_idx_arr = np.asarray([int(r["emb_row_idx"]) for r in self.records], dtype=np.int64)
        self.timestamps = np.asarray([str(r["published_ts_ist"]) for r in self.records])
        self.families = np.asarray([str(r["event_family_hint"]) for r in self.records])
        self.sectors = np.asarray([str(r["sector"]) for r in self.records])
        self.regimes = np.asarray([str(r["regime_causal"]) for r in self.records])
        self.log_mag = np.asarray([float(r["nlp_log_inr_crore"]) for r in self.records], dtype=np.float32)
        self.mat_score = np.asarray([float(r["nlp_mat_score"]) for r in self.records], dtype=np.float32)

    def retrieve(
        self,
        query_emb: np.ndarray,
        *,
        as_of_ts_ist: str,
        event_family: str = "",
        sector: str = "",
        regime_causal: str = "",
        log_inr_crore: float = 0.0,
        mat_score: float = 1.5,
        exclude_event_id: str | None = None,
        top_k: int = 8,
        min_similarity: float = 0.42,
    ) -> dict[str, Any]:
        q_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

        # Strict point-in-time mask: exclude all events at or after query timestamp
        valid_mask = self.timestamps < as_of_ts_ist
        if exclude_event_id is not None:
            valid_mask = valid_mask & (self.event_ids != exclude_event_id)

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return {"comparables": [], "historical_distribution": None, "warning": "no_prior_events"}

        cos_sim = self.embs[self.emb_idx_arr[valid_idx]] @ q_vec

        fam_match = (self.families[valid_idx] == event_family).astype(np.float32) if event_family else np.zeros(len(valid_idx), dtype=np.float32)
        sec_match = (self.sectors[valid_idx] == sector).astype(np.float32) if sector else np.zeros(len(valid_idx), dtype=np.float32)
        reg_match = (self.regimes[valid_idx] == regime_causal).astype(np.float32) if regime_causal else np.zeros(len(valid_idx), dtype=np.float32)
        mag_sim = np.exp(-np.abs(self.log_mag[valid_idx] - float(log_inr_crore)) / 2.5)
        mat_sim = np.exp(-np.abs(self.mat_score[valid_idx] - float(mat_score)) / 1.5)

        hybrid = (
            0.58 * cos_sim
            + 0.14 * fam_match
            + 0.10 * sec_match
            + 0.08 * reg_match
            + 0.05 * mag_sim
            + 0.05 * mat_sim
        )

        qual = np.where((hybrid >= min_similarity) & (cos_sim >= 0.35))[0]
        if len(qual) == 0:
            return {"comparables": [], "historical_distribution": None, "warning": "insufficient_historical_precedents"}

        order = qual[np.argsort(-hybrid[qual])[:top_k]]
        comps = []
        r1s, bhar5s, cars5s, r10s, r20s, vexp = [], [], [], [], [], []

        for loc_i in order:
            r = self.records[int(valid_idx[loc_i])]
            diffs = []
            if event_family and r["event_family_hint"] != event_family:
                diffs.append(f"family:{r['event_family_hint']}")
            if sector and r["sector"] != sector:
                diffs.append(f"sector:{r['sector']}")
            if regime_causal and r["regime_causal"] != regime_causal:
                diffs.append(f"regime:{r['regime_causal']}")
            comps.append({
                "event_id": r["event_id"],
                "symbol": r["symbol"],
                "sector": r["sector"],
                "published_ts_ist": r["published_ts_ist"],
                "t0_date": r["t0_date"],
                "event_family": r["event_family_hint"],
                "regime": r["regime_causal"],
                "source_url": r["source_url"],
                "text_excerpt": str(r["text"])[:220],
                "hybrid_similarity": round(float(hybrid[loc_i]), 4),
                "semantic_cosine": round(float(cos_sim[loc_i]), 4),
                "differences": diffs,
                "observed_reactions": {
                    "raw_1d": float(r["target_raw_ret_1d"]),
                    "bhar_1d": float(r["target_bhar_1d"]),
                    "raw_5d": float(r["target_raw_ret_5d"]),
                    "bhar_5d": float(r["target_bhar_5d"]),
                    "car_5d": float(r["target_car_5d"]),
                    "bhar_10d": float(r["target_bhar_10d"]),
                    "bhar_20d": float(r["target_bhar_20d"]),
                    "vol_expansion_5d": float(r["target_vol_expansion_5d"]),
                },
            })
            r1s.append(float(r["target_bhar_1d"]))
            bhar5s.append(float(r["target_bhar_5d"]))
            cars5s.append(float(r["target_car_5d"]))
            r10s.append(float(r["target_bhar_10d"]))
            r20s.append(float(r["target_bhar_20d"]))
            vexp.append(float(r["target_vol_expansion_5d"]))

        b5 = np.asarray(bhar5s, dtype=np.float64)
        dist = {
            "sample_size": len(comps),
            "as_of_cutoff_ist": as_of_ts_ist,
            "bhar_1d_mean": round(float(np.mean(r1s)), 5),
            "bhar_5d_mean": round(float(np.mean(b5)), 5),
            "bhar_5d_median": round(float(np.median(b5)), 5),
            "bhar_5d_q10": round(float(np.quantile(b5, 0.10)), 5),
            "bhar_5d_q25": round(float(np.quantile(b5, 0.25)), 5),
            "bhar_5d_q75": round(float(np.quantile(b5, 0.75)), 5),
            "bhar_5d_q90": round(float(np.quantile(b5, 0.90)), 5),
            "bhar_5d_positive_rate": round(float(np.mean(b5 > 0)), 4),
            "car_5d_mean": round(float(np.mean(cars5s)), 5),
            "bhar_10d_mean": round(float(np.mean(r10s)), 5),
            "bhar_20d_mean": round(float(np.mean(r20s)), 5),
            "vol_expansion_5d_median": round(float(np.median(vexp)), 4),
        }
        return {"comparables": comps, "historical_distribution": dist, "warning": None}
