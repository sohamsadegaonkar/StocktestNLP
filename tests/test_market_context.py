from datetime import date, datetime
import unittest

from stockbrief.market_context import disclosure_context

NOW = datetime.fromisoformat("2026-11-26T10:00:00+05:30")
FILING = date(2026, 11, 23)


def chart(rows, symbol="INFY", state="available", period="1M"):
    return {"symbol": symbol, "state": state, "period": period, "checked_at": NOW.isoformat(),
            "candles": [{"time": day + "T09:15:00+05:30", "close": close} for day, close in rows]}


class DisclosureContextTests(unittest.TestCase):
    def test_exact_surrounding_sessions_and_relative_return(self):
        stock = chart([("2026-11-20", 100), ("2026-11-23", 999),
                       ("2026-11-24", 105), ("2026-11-25", 200)])
        index = chart([("2026-11-20", 1000), ("2026-11-24", 1020)], "NIFTY50")
        result = disclosure_context(stock, index, FILING, now=NOW)
        self.assertEqual(result["state"], "available")
        self.assertEqual(result["before"]["date"], "2026-11-20")
        self.assertEqual(result["after"]["date"], "2026-11-24")
        self.assertEqual(result["symbol_return_pct"], 5)
        self.assertEqual(result["benchmark_return_pct"], 2)
        self.assertEqual(result["relative_return_pp"], 3)

    def test_no_incomplete_current_day_or_filing_day(self):
        stock = chart([("2026-11-24", 100), ("2026-11-25", 105), ("2026-11-26", 110)])
        result = disclosure_context(stock, {}, date(2026, 11, 25), now=NOW)
        self.assertEqual(result["state"], "pending")
        self.assertIsNone(result["symbol_return_pct"])

    def test_benchmark_must_have_matching_sessions(self):
        stock = chart([("2026-11-20", 100), ("2026-11-24", 95)])
        index = chart([("2026-11-21", 1000), ("2026-11-25", 1020)], "NIFTY50")
        result = disclosure_context(stock, index, FILING, now=NOW)
        self.assertEqual(result["symbol_return_pct"], -5)
        self.assertIsNone(result["relative_return_pp"])

    def test_stale_prices_are_explicit(self):
        stock = chart([("2026-11-20", 100), ("2026-11-24", 101)], state="cached")
        result = disclosure_context(stock, {}, FILING, now=NOW)
        self.assertEqual(result["state"], "cached")
        self.assertEqual(result["symbol_checked_at"], NOW.isoformat())

    def test_future_date_does_not_produce_return(self):
        result = disclosure_context({}, {}, date(2026, 11, 27), now=NOW)
        self.assertEqual(result["reason"], "future_filing_date")
        self.assertIsNone(result["symbol_return_pct"])

    def test_large_gap_is_not_treated_as_reaction(self):
        stock = chart([("2026-11-01", 100), ("2026-11-24", 105)])
        result = disclosure_context(stock, {}, FILING, now=NOW)
        self.assertEqual(result["reason"], "session_gap_too_large")

    def test_intraday_data_not_used_as_daily_closes(self):
        stock = chart([("2026-11-20", 100), ("2026-11-24", 105)], period="1D")
        result = disclosure_context(stock, {}, FILING, now=NOW)
        self.assertEqual(result["reason"], "symbol_data_unavailable")

    def test_invalid_rows_and_naive_observation_time(self):
        stock = chart([("2026-11-20", 100), ("2026-11-24", float("nan"))])
        self.assertEqual(disclosure_context(stock, {}, FILING, now=NOW)["state"], "pending")
        with self.assertRaises(ValueError):
            disclosure_context(stock, {}, FILING, now=datetime(2026, 11, 26))
