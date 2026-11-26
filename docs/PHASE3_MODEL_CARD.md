# Phase 3 Model Card — StocktestNLP Market Intelligence Engine (`stockbrief-phase3-market-intel-v1`)

## Model Overview
- **System:** `MarketIntelligenceEngine` ([adapter.py](file:///d:/StocktestNLP/phase3/adapter.py)) combining Phase 1/2 ONNX INT8 NLP models, strict point-in-time `HistoricalSimilarityEngine` (`26,186` events, `2022–2026`), and CatBoost-ONNX 5-session directional & volatility models.
- **Runtime:** Pure `onnxruntime` + `tokenizers` + `numpy` inside `.venv` (`393.9 MB` peak RSS, `1.372 s` cold load, `55.05 ms` p50 / `82.35 ms` p95).
- **Promotion Status:** **Analytics + Volatility Engine Only.** Directional excess-return probabilities are marked `experimental_unvalidated_above_market_baseline` because Combined NLP+Market directional lift (`+0.00205` balanced accuracy, 95% CI `[-0.01454, +0.02132]`) did not clear statistical significance over Market-Only baselines.

## Strict Safety & Non-Causality Contracts
- **Historical Facts vs Predictions:** `historical_facts` (comparable historical filings, URLs, observed CAR/BHAR distributions) are strictly separated from `predictions`.
- **Zero Future Leakage:** `HistoricalSimilarityEngine` masks out all records with `published_ts_ist >= as_of_ts_ist`. Financial fundamentals apply a `+60` calendar-day filing lag.
- **Prohibited Uses:** Never emits buy/sell advice, guaranteed return forecasts, automated trading execution, or causal claims. Cached OHLCV data is never represented as live market prices.
