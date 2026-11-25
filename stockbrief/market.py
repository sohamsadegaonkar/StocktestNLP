"""Public Upstox candle data for the local NSE watchlist."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from threading import Lock
from time import monotonic
from urllib.parse import quote

import httpx

UPSTOX = "https://api.upstox.com/v3/historical-candle"
IST = timezone(timedelta(hours=5, minutes=30))
REFRESH_SECONDS = 60


@dataclass(frozen=True)
class Instrument:
    name: str
    key: str
    unit: str = "INR"


INSTRUMENTS = {
    "NIFTY50": Instrument("Nifty 50", "NSE_INDEX|Nifty 50", "points"),
    "TMPV": Instrument("Tata Motors Passenger Vehicles", "NSE_EQ|INE155A01022"),
    "INFY": Instrument("Infosys", "NSE_EQ|INE009A01021"),
    "RELIANCE": Instrument("Reliance Industries", "NSE_EQ|INE002A01018"),
    "TCS": Instrument("Tata Consultancy Services", "NSE_EQ|INE467B01029"),
    "HDFCBANK": Instrument("HDFC Bank", "NSE_EQ|INE040A01034"),
}


def read_candles(payload: dict) -> list[tuple[datetime, float]]:
    """Validate provider shape and return distinct, ascending candle closes."""
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError("Invalid market-data response")
    data = payload.get("data")
    rows = data.get("candles") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Missing candles")
    candles = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        try:
            when = datetime.fromisoformat(row[0])
            close = float(row[4])
        except (TypeError, ValueError, OverflowError):
            continue
        if when.tzinfo is not None and isfinite(close) and 0 < close < 1e12:
            candles[when] = close
    if rows and not candles:
        raise ValueError("No usable candle rows")
    return sorted(candles.items())[-2_000:]


class MarketData:
    """One upstream check per instrument and period per minute."""
    def __init__(self):
        self._locks = {(symbol, period): Lock() for symbol in INSTRUMENTS for period in ("1D", "1M")}
        self._cache = {}

    @staticmethod
    def _get(path: str) -> list[tuple[datetime, float]]:
        response = httpx.get(f"{UPSTOX}/{path}", headers={"Accept": "application/json"}, timeout=5)
        response.raise_for_status()
        return read_candles(response.json())

    def _load(self, instrument: Instrument, period: str) -> list[tuple[datetime, float]]:
        key = quote(instrument.key, safe="")
        today = datetime.now(IST).date()
        if period == "1M":
            return self._get(f"{key}/days/1/{today}/{today - timedelta(days=30)}")
        candles = self._get(f"intraday/{key}/minutes/5")
        if not candles:
            candles = self._get(f"{key}/minutes/5/{today}/{today - timedelta(days=7)}")
        if not candles:
            return []
        last_session = candles[-1][0].astimezone(IST).date()
        return [(when, price) for when, price in candles if when.astimezone(IST).date() == last_session]

    def chart(self, symbol: str, period: str) -> dict:
        instrument = INSTRUMENTS[symbol]
        key = (symbol, period)
        with self._locks[key]:
            previous = self._cache.get(key)
            if previous and monotonic() < previous[0]:
                return previous[1]

            response = {
                "symbol": symbol, "name": instrument.name, "unit": instrument.unit, "period": period,
                "source": "Upstox", "state": "empty", "checked_at": None, "candles": [],
            }
            try:
                candles = self._load(instrument, period)
                response.update(
                    state="available" if candles else "empty",
                    checked_at=datetime.now(timezone.utc).isoformat(),
                    candles=[{"time": when.isoformat(), "close": close} for when, close in candles],
                )
            except (httpx.HTTPError, ValueError, ImportError):
                old = previous[1] if previous else None
                if old and old["candles"]:
                    response = {**old, "state": "cached"}
                else:
                    response["state"] = "unavailable"
            self._cache[key] = (monotonic() + REFRESH_SECONDS, response)
            return response
