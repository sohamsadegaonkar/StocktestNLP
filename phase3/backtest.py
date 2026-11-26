"""Research-Only Event-Driven Backtest & Regime Stress Testing (Zero Real-Money Execution)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TEST_PREDS_PATH = Path("phase3/.cache/test_predictions.parquet")
WF_PREDS_PATH = Path("phase3/.cache/wf_predictions.parquet")
BACKTEST_OUT = Path("phase3/results/backtest_report.json")


def simulate_strategy(
    df: pd.DataFrame,
    prob_pos_col: str,
    prob_neg_col: str | None,
    *,
    long_short: bool = False,
    prob_thresh: float = 0.40,
    min_turnover_log: float = float(np.log1p(5_000_000)),
    round_trip_cost_bps: float = 30.0,
    max_daily_positions: int = 5,
) -> dict[str, Any]:
    """Simulate realistic 5-day slotted holding with execution delay, liquidity screen, and costs."""
    if df.empty:
        return {"trades": 0}
    sub = df[df["tech_log_turnover20"] >= min_turnover_log].copy()
    sub = sub.sort_values(["t0_date", prob_pos_col], ascending=[True, False])

    cost = round_trip_cost_bps / 10_000.0
    trades = []
    for dt, grp in sub.groupby("t0_date", sort=True):
        day_slots = []
        # Long signals
        longs = grp[grp[prob_pos_col] >= prob_thresh].head(max_daily_positions)
        for _, r in longs.iterrows():
            gross = float(r["target_exec_ret_5d"]) if not long_short else float(r["target_exec_excess_5d"])
            net = gross - cost
            day_slots.append({"date": dt, "side": 1, "gross": gross, "net": net, "bench": float(r["target_bench_ret_5d"]), "regime": r["regime_causal"]})

        if long_short and prob_neg_col is not None:
            shorts = grp[grp[prob_neg_col] >= prob_thresh].head(max_daily_positions)
            for _, r in shorts.iterrows():
                gross = -float(r["target_exec_excess_5d"])
                net = gross - cost
                day_slots.append({"date": dt, "side": -1, "gross": gross, "net": net, "bench": float(r["target_bench_ret_5d"]), "regime": r["regime_causal"]})

        if day_slots:
            trades.extend(day_slots)

    if not trades:
        return {"trades": 0, "net_total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    tdf = pd.DataFrame(trades)
    # Aggregate by entry date, allocating 1/5th capital per concurrent 5-day cohort to prevent overlapping leverage
    daily = tdf.groupby("date", sort=True).agg(
        net_mean=("net", "mean"),
        gross_mean=("gross", "mean"),
        bench_mean=("bench", "mean"),
        n=("net", "size"),
    )
    # Approximate daily portfolio P&L from 5-day cohort returns divided across 5 holding days
    daily_port_ret = daily["net_mean"].to_numpy(dtype=np.float64) / 5.0
    daily_gross_ret = daily["gross_mean"].to_numpy(dtype=np.float64) / 5.0
    daily_bench_ret = daily["bench_mean"].to_numpy(dtype=np.float64) / 5.0

    eq_curve = np.cumprod(1.0 + daily_port_ret)
    peak = np.maximum.accumulate(eq_curve)
    mdd = float(np.min((eq_curve - peak) / np.maximum(peak, 1e-12)))

    mu = float(np.mean(daily_port_ret))
    sd = float(np.std(daily_port_ret, ddof=1)) if len(daily_port_ret) > 1 else 0.0
    downside = float(np.std(np.minimum(daily_port_ret, 0.0), ddof=1)) if len(daily_port_ret) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(252.0) if sd > 1e-9 else 0.0
    sortino = (mu / downside) * np.sqrt(252.0) if downside > 1e-9 else 0.0

    # Regime breakdown (trade-level mean net excess / return)
    by_reg = {}
    for reg, rgrp in tdf.groupby("regime"):
        by_reg[str(reg)] = {
            "trades": int(len(rgrp)),
            "mean_net_5d": round(float(rgrp["net"].mean()), 5),
            "hit_rate": round(float((rgrp["net"] > 0).mean()), 4),
        }

    return {
        "trades": int(len(tdf)),
        "active_days": int(len(daily)),
        "round_trip_cost_bps": round_trip_cost_bps,
        "gross_cum_return": round(float(np.prod(1.0 + daily_gross_ret) - 1.0), 4),
        "net_cum_return": round(float(eq_curve[-1] - 1.0), 4),
        "benchmark_cum_return": round(float(np.prod(1.0 + daily_bench_ret) - 1.0), 4),
        "sharpe_annualized": round(float(sharpe), 3),
        "sortino_annualized": round(float(sortino), 3),
        "max_drawdown": round(float(mdd), 4),
        "trade_hit_rate": round(float((tdf["net"] > 0).mean()), 4),
        "mean_net_5d_per_trade": round(float(tdf["net"].mean()), 5),
        "avg_daily_turnover_pct": round(float(min(100.0, len(tdf) / max(1, len(daily)) * 20.0)), 2),
        "by_regime": by_reg,
    }


def main() -> None:
    BACKTEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    te_df = pd.read_parquet(TEST_PREDS_PATH)
    wf_df = pd.read_parquet(WF_PREDS_PATH) if WF_PREDS_PATH.is_file() else pd.DataFrame()

    results = {
        "disclaimer": "Research-only historical simulation with zero real-money execution. Uses executable post-announcement entry (Open[T0] or Close[T0] for intraday), liquidity floor >= INR 50 lakh daily turnover, and 5-day non-overlapping capital slots.",
        "frozen_test_horizon": f"{te_df['t0_date'].min()}..{te_df['t0_date'].max()}",
        "frozen_test_strategies": {
            "combined_long_only_20bps": simulate_strategy(te_df, "p_pos", None, long_short=False, round_trip_cost_bps=20.0),
            "combined_long_only_30bps": simulate_strategy(te_df, "p_pos", None, long_short=False, round_trip_cost_bps=30.0),
            "combined_long_only_50bps": simulate_strategy(te_df, "p_pos", None, long_short=False, round_trip_cost_bps=50.0),
            "combined_long_short_excess_30bps": simulate_strategy(te_df, "p_pos", "p_neg", long_short=True, round_trip_cost_bps=30.0),
            "market_only_long_30bps": simulate_strategy(te_df, "p_mkt_pos", None, long_short=False, round_trip_cost_bps=30.0),
        },
        "walk_forward_oos_strategies": (
            {
                "combined_long_only_30bps": simulate_strategy(wf_df, "p_pos", None, long_short=False, round_trip_cost_bps=30.0),
                "combined_long_short_excess_30bps": simulate_strategy(wf_df, "p_pos", "p_neg", long_short=True, round_trip_cost_bps=30.0),
                "market_only_long_30bps": simulate_strategy(wf_df, "p_mkt_pos", None, long_short=False, round_trip_cost_bps=30.0),
            }
            if not wf_df.empty
            else {}
        ),
    }
    BACKTEST_OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    c30 = results["frozen_test_strategies"]["combined_long_only_30bps"]
    m30 = results["frozen_test_strategies"]["market_only_long_30bps"]
    ls30 = results["frozen_test_strategies"]["combined_long_short_excess_30bps"]
    print(
        f"Backtest Frozen Test (30bps): Combined Long Net={c30['net_cum_return']*100:+.2f}% Sharpe={c30['sharpe_annualized']} MDD={c30['max_drawdown']*100:.2f}% "
        f"| Market-Only Long Net={m30['net_cum_return']*100:+.2f}% Sharpe={m30['sharpe_annualized']} "
        f"| L/S Excess Net={ls30['net_cum_return']*100:+.2f}% Sharpe={ls30['sharpe_annualized']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
