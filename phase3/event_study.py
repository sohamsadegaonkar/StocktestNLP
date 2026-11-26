"""Financial Event Study with Market-Model CAR, Cluster-Robust Inference, Overlapping Control & Regimes."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy import stats

ALIGNED_PATH = Path("phase3/.cache/aligned_events.pkl")
EVENT_STUDY_OUT = Path("phase3/results/event_study_report.json")
HORIZONS = (1, 3, 5, 10, 20)


def clustered_mean_ci(values: np.ndarray, clusters: list[str]) -> dict[str, Any]:
    """Compute sample mean, OLS t-stat, and cluster-robust (by issuer) SE/t-stat/95% CI."""
    n = len(values)
    if n < 5:
        return {"n": n, "mean": 0.0, "median": 0.0, "t_cluster": 0.0, "p_cluster": 1.0}
    mu = float(np.mean(values))
    med = float(np.median(values))
    by_c: defaultdict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        by_c[c].append(float(v))
    g = len(by_c)
    if g < 3:
        se_ols = float(np.std(values, ddof=1) / np.sqrt(n))
        t_ols = mu / se_ols if se_ols > 1e-12 else 0.0
        return {"n": n, "clusters": g, "mean": round(mu, 6), "median": round(med, 6), "t_cluster": round(t_ols, 3), "p_cluster": round(float(2 * stats.t.sf(abs(t_ols), max(1, n - 1))), 5)}
    # Liang-Zeger one-way cluster-robust variance of the mean
    cluster_sums = np.array([sum(v - mu for v in vs) for vs in by_c.values()], dtype=np.float64)
    scale = g / max(1, g - 1)
    var_mu = scale * float(np.sum(cluster_sums ** 2)) / (n ** 2)
    se_c = float(np.sqrt(max(var_mu, 1e-18)))
    t_c = mu / se_c if se_c > 1e-12 else 0.0
    p_c = float(2 * stats.t.sf(abs(t_c), df=g - 1))
    return {
        "n": n,
        "clusters": g,
        "mean": round(mu, 6),
        "median": round(med, 6),
        "se_cluster": round(se_c, 6),
        "t_cluster": round(t_c, 3),
        "p_cluster": round(p_c, 5),
        "ci95_low": round(mu - 1.96 * se_c, 6),
        "ci95_high": round(mu + 1.96 * se_c, 6),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    syms = [r["symbol"] for r in rows]
    horizons_out = {}
    for h in HORIZONS:
        car = np.array([r["targets"][f"car_{h}d"] for r in rows], dtype=np.float64)
        bhar = np.array([r["targets"][f"bhar_{h}d"] for r in rows], dtype=np.float64)
        raw = np.array([r["targets"][f"raw_ret_{h}d"] for r in rows], dtype=np.float64)
        dd = np.array([r["targets"][f"max_dd_{h}d"] for r in rows], dtype=np.float64)
        horizons_out[f"{h}d"] = {
            "car": clustered_mean_ci(car, syms),
            "bhar": clustered_mean_ci(bhar, syms),
            "raw_mean": round(float(np.mean(raw)), 6),
            "pos_bhar_rate": round(float(np.mean(bhar > 0)), 4),
            "abs_bhar_mean": round(float(np.mean(np.abs(bhar))), 6),
            "mean_max_dd": round(float(np.mean(dd)), 6),
        }
    vol_r = np.array([r["vol_ratio_t0"] for r in rows], dtype=np.float64)
    vol_exp = np.array([r["vol_expansion_5d"] for r in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "unique_issuers": len(set(syms)),
        "mean_t0_volume_ratio": round(float(np.mean(np.clip(vol_r, 0, 20))), 3),
        "median_t0_volume_ratio": round(float(np.median(vol_r)), 3),
        "vol_expansion_gt_1_25_rate": round(float(np.mean(vol_exp > 1.25)), 4),
        "horizons": horizons_out,
    }


def main() -> None:
    EVENT_STUDY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ALIGNED_PATH.open("rb") as f:
        aligned: list[dict[str, Any]] = pickle.load(f)

    clean_non_overlap = [r for r in aligned if not r["is_overlapping_10d"]]
    verified_only = [r for r in aligned if r["corp_action_verified"]]

    by_family = {}
    families = sorted({r["event_family_hint"] for r in aligned})
    for fam in families:
        sub_all = [r for r in aligned if r["event_family_hint"] == fam]
        sub_clean = [r for r in clean_non_overlap if r["event_family_hint"] == fam]
        by_family[fam] = {
            "all_events": summarize_group(sub_all),
            "non_overlapping_10d_clean": summarize_group(sub_clean) if len(sub_clean) >= 10 else {"count": len(sub_clean)},
        }

    by_timing = {}
    for tb in ("pre_market", "intraday", "post_market"):
        sub = [r for r in aligned if r["timing_bucket"] == tb]
        by_timing[tb] = summarize_group(sub)

    report = {
        "note": "All figures measure observed historical association around IST disclosure timestamps using pre-event OLS market models [-120, -11]. Observed CAR/volume/volatility patterns do not prove causal price effects.",
        "overall": summarize_group(aligned),
        "non_overlapping_10d_overall": summarize_group(clean_non_overlap),
        "corp_action_verified_overall": summarize_group(verified_only),
        "by_event_family": by_family,
        "by_timing_bucket": by_timing,
    }
    EVENT_STUDY_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Event Study complete ({len(aligned)} events, {len(clean_non_overlap)} clean non-overlapping).", flush=True)
    for fam in families:
        s5 = by_family[fam]["all_events"]["horizons"]["5d"]
        s1 = by_family[fam]["all_events"]["horizons"]["1d"]
        print(
            f"  {fam:18s} n={by_family[fam]['all_events']['count']:5d} | "
            f"1d CAR={s1['car']['mean']*100:+.3f}% (t={s1['car']['t_cluster']:+.2f}) | "
            f"5d CAR={s5['car']['mean']*100:+.3f}% (t={s5['car']['t_cluster']:+.2f}) | "
            f"|5d BHAR|={s5['abs_bhar_mean']*100:.2f}% | VolExp>1.25={by_family[fam]['all_events']['vol_expansion_gt_1_25_rate']*100:.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
