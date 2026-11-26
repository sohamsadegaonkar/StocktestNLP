"""Recorded exchange closes for the offline Infosys earnings chart."""
from .fixtures import EARNINGS
from .market import INSTRUMENTS

CLOSES = (
    (16, 1726.05, 24613.00),
    (19, 1792.95, 24530.90),
    (22, 1810.85, 24509.25),
    (23, 1836.90, 24479.05),
    (24, 1833.95, 24413.50),
    (25, 1824.85, 24406.10),
)


def recorded_example() -> dict:
    instrument = INSTRUMENTS[EARNINGS.ticker]
    _, stock_base, index_base = CLOSES[0]
    comparisons = []
    for session, (day, stock, index) in enumerate(CLOSES[1:], start=1):
        stock_return = 100 * (stock / stock_base - 1)
        index_return = 100 * (index / index_base - 1)
        comparisons.append({"session": session, "date": f"2024-07-{day}", "stock_pct": stock_return,
                            "index_pct": index_return, "difference_pp": stock_return - index_return})
    return {
        "symbol": EARNINGS.ticker, "name": instrument.name, "unit": instrument.unit, "period": "1M",
        "source": "Recorded Infosys example · EquityPandit", "state": "recorded", "checked_at": None,
        "title": EARNINGS.title, "filing_date": EARNINGS.date, "baseline_date": "2024-07-16",
        "candles": [{"time": f"2024-07-{day}T00:00:00+05:30", "close": stock} for day, stock, _ in CLOSES],
        "comparisons": comparisons,
        "source_links": {
            "Infosys earnings release": EARNINGS.url,
            "Infosys historical closes": "https://www.equitypandit.com/historical-data/INFY",
            "Nifty 50 historical closes": "https://www.equitypandit.com/share-price/today/nifty-50-historical-data",
        },
    }
