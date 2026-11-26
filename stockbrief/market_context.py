"""Conservative daily market context around an exchange disclosure.

The filing date is not a timestamp. Exclude its entire IST date and today's
potentially incomplete candle; do not infer an intraday reaction or causality.
"""
from datetime import date, datetime, timedelta, timezone
from math import isfinite

IST = timezone(timedelta(hours=5, minutes=30))
MAX_SESSION_GAP = timedelta(days=7)


def _closes(chart: dict, today: date) -> dict[date, float]:
    if chart.get("state") not in {"available", "cached"} or chart.get("period") != "1M":
        return {}
    result = {}
    for row in chart.get("candles", []):
        try:
            when = datetime.fromisoformat(row["time"])
            close = float(row["close"])
            if when.tzinfo is None or not isfinite(close) or close <= 0:
                continue
            day = when.astimezone(IST).date()
            if day < today:
                result[day] = close
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return result


def disclosure_context(symbol_chart: dict, benchmark_chart: dict, filing_date: date, *,
                       now: datetime | None = None) -> dict:
    """Compare close-to-close movement using strictly surrounding sessions."""
    now = now if now is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Use an offset-aware observation time.")
    today = now.astimezone(IST).date()
    result = {
        "symbol": symbol_chart.get("symbol"),
        "filing_date": filing_date.isoformat(),
        "observed_at": now.isoformat(),
        "source": "Upstox",
        "benchmark": "NIFTY50",
        "symbol_data_state": symbol_chart.get("state", "unavailable"),
        "benchmark_data_state": benchmark_chart.get("state", "unavailable"),
        "symbol_checked_at": symbol_chart.get("checked_at"),
        "benchmark_checked_at": benchmark_chart.get("checked_at"),
        "state": "unavailable",
        "reason": None,
        "before": None, "after": None,
        "symbol_return_pct": None, "benchmark_return_pct": None,
        "relative_return_pp": None,
        "method": "Last completed daily close before filing date to first completed daily close after filing date (IST).",
        "limitations": [
            "Observed price movement, not proof of news causation or a forecast.",
            "Filing-day and current-day candles excluded; the filing time is unknown.",
            "Price returns may reflect corporate actions; dividend or split adjustment is not verified.",
        ],
    }
    if filing_date > today:
        result["reason"] = "future_filing_date"
        return result
    closes = _closes(symbol_chart, today)
    if not closes:
        result["reason"] = "symbol_data_unavailable"
        return result
    before = max((d for d in closes if d < filing_date), default=None)
    after = min((d for d in closes if d > filing_date), default=None)
    if before is None:
        result["reason"] = "no_prior_completed_session"
        return result
    if after is None:
        result["state"] = "pending" if filing_date >= today - MAX_SESSION_GAP else "unavailable"
        result["reason"] = "no_later_completed_session"
        return result
    if filing_date - before > MAX_SESSION_GAP or after - filing_date > MAX_SESSION_GAP:
        result["reason"] = "session_gap_too_large"
        return result

    change = (closes[after] / closes[before] - 1) * 100
    result.update(state="available", before={"date": before.isoformat(), "close": closes[before]},
                  after={"date": after.isoformat(), "close": closes[after]},
                  symbol_return_pct=round(change, 4))
    benchmark = _closes(benchmark_chart, today)
    if before in benchmark and after in benchmark:
        bench_change = (benchmark[after] / benchmark[before] - 1) * 100
        result.update(benchmark_return_pct=round(bench_change, 4),
                      relative_return_pp=round(change - bench_change, 4))
    else:
        result["limitations"].append("Benchmark lacks the exact matching sessions; relative movement is unavailable.")
    if "cached" in (result["symbol_data_state"], result["benchmark_data_state"]):
        result["state"] = "cached"
        result["limitations"].append("At least one provider response is cached; check the source fetch timestamps.")
    return result
