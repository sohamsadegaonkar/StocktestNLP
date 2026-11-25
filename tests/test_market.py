from datetime import datetime
import unittest
from unittest.mock import Mock, patch

import httpx

from stockbrief.market import MarketData, read_candles


def candle(stamp, close):
    return [stamp, close, close, close, close, 0, 0]


class MarketDataTests(unittest.TestCase):
    def test_provider_rows_are_validated_deduplicated_and_sorted(self):
        early = "2026-11-24T09:15:00+05:30"
        late = "2026-11-25T09:15:00+05:30"
        payload = {"status": "success", "data": {"candles": [
            candle(late, 100), candle(early, 97), candle(late, 101),
            candle("invalid", 800), candle("2026-11-26T09:15:00+05:30", float("nan")),
        ]}}
        self.assertEqual(read_candles(payload), [
            (datetime.fromisoformat(early), 97.0), (datetime.fromisoformat(late), 101.0),
        ])

    def test_intraday_falls_back_to_the_latest_completed_session(self):
        first = datetime.fromisoformat("2026-11-24T09:15:00+05:30")
        recent = datetime.fromisoformat("2026-11-25T09:15:00+05:30")
        provider = Mock(side_effect=[[], [(first, 100.0), (recent, 108.0)]])
        with patch.object(MarketData, "_get", provider):
            chart = MarketData().chart("NIFTY50", "1D")
        self.assertEqual(chart["candles"], [{"time": recent.isoformat(), "close": 108.0}])
        self.assertEqual(chart["unit"], "points")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(provider.call_args_list[0].args[0], "intraday/NSE_INDEX%7CNifty%2050/minutes/5")
        self.assertIn("NSE_INDEX%7CNifty%2050/minutes/5/", provider.call_args_list[1].args[0])

    def test_failed_refresh_preserves_previous_candle_and_fetch_time(self):
        provider = Mock(side_effect=[
            [(datetime.fromisoformat("2026-11-25T09:15:00+05:30"), 420.5)],
            httpx.ConnectError("offline"),
        ])
        with patch.object(MarketData, "_load", provider), patch(
            "stockbrief.market.monotonic", side_effect=[0, 1, 61, 62, 63],
        ):
            market = MarketData()
            fresh = market.chart("TMPV", "1M")
            self.assertEqual(market.chart("TMPV", "1M"), fresh)
            stale = market.chart("TMPV", "1M")
            self.assertEqual(market.chart("TMPV", "1M"), stale)
        self.assertEqual(fresh["state"], "available")
        self.assertEqual(stale["state"], "cached")
        self.assertEqual(stale["checked_at"], fresh["checked_at"])
        self.assertEqual(stale["candles"], fresh["candles"])
        self.assertEqual(provider.call_count, 2)
