# Phase 3 — Historical Market Reaction & Predictive Intelligence Report

**Branch:** `ml/phase3-market-reaction` (branched from `ml/phase2-financial-reasoning`)
**Promotion Decision:** **`RETAIN_HISTORICAL_ANALYTICS_ENGINE_ONLY_INCONCLUSIVE_NLP_LIFT`**
*(Preserves negative experimental result honestly: directional NLP features improve volatility prediction `AUC=0.7028 vs 0.6898`, but do NOT establish statistically significant directional excess-return lift above market-only baselines out-of-sample.)*

---

## 1. Executive Summary & Scientific Conclusion

Phase 3 evaluates whether adding Phase 1 & Phase 2 financial NLP representations to point-in-time market technicals, VIX/sector regimes, and lagged fundamentals improves out-of-sample predictions of post-disclosure Indian stock behavior across **`26,186` aligned NSE events (`2022-01-03` to `2026-10-23`, `310` issuers)**.

### Core Empirical Findings
1. **Structural Event Reactions Exist in Historical Associations:** Cluster-robust event studies show significant observed reactions in contract wins/operations (`+0.535%` 1d CAR, `t=+7.46`; `+0.613%` 5d CAR, `t=+4.66`), pricing/launches (`+1.310%` 5d CAR, `t=+2.67`), M&A deals (`+0.126%` 1d CAR, `t=+2.55`), and earnings (`-0.544%` 5d CAR, `t=-4.56` alongside `34.7%` post-event volatility expansion `> 1.25x`).
2. **Directional NLP Lift Is Inconclusive Over Strong Market-Only Baselines:** On the embargoed frozen holdout (`Test = 4,035` events, `2026-03-17..2026-10-23`), `CatBoost Combined` achieved **`0.3906` Balanced Accuracy (`MCC=0.0802`)** vs **`0.3898` Balanced Accuracy (`MCC=0.0785`)** for `CatBoost Market-Only`. Grouped block-bootstrap 95% confidence intervals straddle zero by both month (**`+0.00205`, 95% CI `[-0.01454, +0.02132]`**) and issuer (**`+0.00034`, 95% CI `[-0.01542, +0.01558]`**).
3. **Volatility Expansion & Return Dispersion Are Predictable:** Post-event `5d` volatility expansion (`> 1.25x` pre-event 20d vol) improves with multimodal NLP + market features (**`ROC-AUC = 0.7028`** vs `0.6898` Market-Only; `Balanced Acc = 0.6352`, `Brier = 0.2046`).
4. **Classical GBDTs Outperform Deep Tabular Neural Net:** `CatBoost` (`0.3906`), `XGBoost` (`0.3869`), and `LightGBM` (`0.3840`) all beat `CompactTabularMLP` (`0.3711`) and `LogisticRegression` (`0.3653`) on the exact same frozen holdout.
5. **Backtest Reality Check:** Under realistic post-announcement execution (`Open[T0]` or `Close[T0]` for intraday), `>= INR 50 lakh` daily turnover floor, and `30 bps` round-trip transaction costs, `Combined Long-Only` returned **`-1.01%` (`Sharpe = -0.171`, `MDD = -10.14%`)** on Frozen Test vs **`+2.43%` (`Sharpe = +0.601`)** for `Market-Only Long-Only`, while `Long/Short Excess` lost `-14.41%` (`Sharpe = -2.48`).
6. **Edge Deployment:** Exporting CatBoost directional (`657.6 KB`) and volatility (`995.3 KB`) models to native ONNX enables pure-`onnxruntime` inference (zero PyTorch/CatBoost/Pandas) in **`55.05 ms` warm p50 / `82.35 ms` warm p95** (`17.33 docs/s`, **`393.9 MB` peak RSS**, `1.372 s` cold load) including the `26,186`-event historical similarity engine.

---

## 2. Historical Event Collection, Market Data & Session Alignment

### 2.1 Corpus & Market Data Provenance
- **Official NSE Disclosures (`phase3/collect_nse_history.py`):** Scanned **`829,588`** official NSE announcements across `16` quarterly/semi-annual slices (`2022-01-01` to `2026-11-25`), deduplicated **`193,784`** material filings, and curated **`28,000`** events across **`323` issuers** (`2022: 4,517`, `2023: 4,880`, `2024: 5,877`, `2025: 6,551`, `2026: 6,175`).
- **Timing Breakdown:** `pre_market`: `678`; `intraday`: `9,831`; `post_market`: `17,491` (second-level IST timestamps from `an_dt`; missing timestamps excluded).
- **Market Data (`phase3/collect_market_data.py`):**
  - **Benchmarks (`2021-07-01..2026-11-23`):** `^NSEI` (`1,333` sessions), `^INDIAVIX` (`1,326` sessions), and NSE sector indices (`^CNXIT`, `^NSEBANK`, `^CNXAUTO`, `^CNXPHARMA`, `^CNXFMCG`, `^CNXMETAL`, `^CNXENERGY`; `1,326–1,332` sessions). Note: `^CNXFIN` returned only 1 bar, so `Financial Services` maps cleanly to `NSEBANK`.
  - **Equities & Survivorship Audit:** Resolved **`310 / 323` (`96.0%`)** tickers (`274` via Yahoo Finance `Adj Close` + splits/dividends + `270` annual / `266` quarterly statements; `36` Yahoo-rate-limited tickers recovered via official Upstox v3 ISIN historical daily candles; `13` unresolved/delisted/short tickers excluded and logged in [market_provenance.json](file:///d:/StocktestNLP/phase3/data/market_provenance.json)).
  - **Live-Feed Guardrail:** All OHLCV & financial statement series are explicitly flagged as cached historical daily datasets only, never live quotes.

### 2.2 IST Trading-Session Alignment (`phase3/event_alignment.py`)
- **`26,186 / 28,000`** events aligned cleanly to NSE trading sessions (`2022-01-03..2026-10-23`, `310` issuers; `1,153` skipped for unresolved symbol, `660` for `<120` pre-history or `<20` tail days, `1` missing boundary bar):
  - **Pre-market (`< 09:15 IST`):** `T0 = same trading day`, anchor `T0-1`, executable entry `Open_Adj[T0]`.
  - **Intraday (`09:15–15:30 IST`):** event study anchor `T0-1`, **executable entry `Close_Adj[T0]`** (never entering at `Open[T0]` before an intraday announcement).
  - **Post-market (`> 15:30 IST`) / Holiday:** `T0 = next trading session`, anchor `last session < T0`, executable entry `Open_Adj[T0]`.
- **Market Model:** OLS `R_i = alpha + beta * R_m` estimated strictly over pre-event **`[-120, -11]`** trading sessions (`22,842` events corporate-action-verified via Yahoo `Adj Close`; **`10,636`** clean non-overlapping events separated by `> 10` sessions per issuer).

---

## 3. Financial Event Study (`phase3/results/event_study_report.json`)

Standard errors are cluster-robust by issuer (`310` clusters) to account for repeated announcements per company:

| Event Family | Count (`n`) | `1d` Mean CAR (`t_cluster`) | `5d` Mean CAR (`t_cluster`) | `|5d BHAR|` Dispersion | `VolExp > 1.25x` Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Aligned Events** | `26,186` | `+0.038%` (`+1.20`) | `-0.007%` (`-0.12`) | `4.30%` | `26.62%` |
| **Clean Non-Overlapping (`>10d`)** | `10,636` | `+0.078%` (`+2.41`) | — | — | `27.60%` |
| `operations` (contracts/orders) | `2,985` | **`+0.535%` (`+7.46`)** | **`+0.613%` (`+4.66`)** | `4.06%` | `23.2%` |
| `pricing_product` (launches/tariffs) | `123` | `+0.428%` (`+1.69`) | **`+1.310%` (`+2.67`)** | `3.49%` | `20.3%` |
| `deal_ma` (M&A / demergers) | `4,421` | `+0.126%` (`+2.55`) | `+0.273%` (`+2.21`) | `3.75%` | `21.2%` |
| `earnings` (quarterly/annual results) | `6,510` | `-0.165%` (`-2.45`) | **`-0.544%` (`-4.56`)** | **`5.06%`** | **`34.7%`** |
| `corporate_action` (div/buyback/split) | `3,958` | `-0.066%` (`-0.83`) | `+0.073%` (`+0.56`) | `4.51%` | `28.9%` |
| `credit_debt` (ratings/NCD/QIP) | `4,245` | `+0.044%` (`+0.73`) | `-0.020%` (`-0.15`) | `4.08%` | `23.9%` |
| `governance` (CEO/CFO/auditors) | `1,426` | `-0.052%` (`-0.61`) | `-0.098%` (`-0.64`) | `3.64%` | `24.3%` |
| `regulatory` (SEBI/tax/litigation) | `1,612` | `-0.040%` (`-0.58`) | `+0.067%` (`+0.46`) | `3.64%` | `23.1%` |
| `rumor_denial` (clarifications) | `906` | `+0.068%` (`+0.52`) | `-0.022%` (`-0.09`) | `4.60%` | `20.0%` |

*(Overall `10d` BHAR `= +0.343%` (`t=+3.71`), `20d` BHAR `= +0.747%` (`t=+5.24`), `20d` CAR `= -0.259%` (`t=-2.51`). Observed associations are descriptive, not causal proof.)*

---

## 4. Zero-Leakage Splits, Features & Predictive Evaluation (`phase3/results/modeling_report.json`)

### 4.1 Leakage Prevention (`phase3/data/leakage_audit.json`)
- **Chronological + Purged 20-Session Embargo:**
  - **Train:** `17,489` events (`2022-01-03..2025-05-30`)
  - **Embargo 1:** `21` trading sessions gap
  - **Val (Dev):** `3,879` events (`2025-06-30..2026-02-13`)
  - **Embargo 2:** `21` trading sessions gap
  - **Frozen Test:** `4,035` events (`2026-03-17..2026-10-23`; class balance `Neg=1,534`, `Neut=1,116`, `Pos=1,385` at `±1.5%` 5d executable excess return)
- **Zero-Overlap Guarantees:** `0` event-ID overlap, `0` near-duplicate `(symbol, date, text)` overlap, `TruncatedSVD(16)` fit strictly inside Train only, and `+60` calendar-day statutory lag on all fundamental statement joins (`65.48%` PIT coverage).

### 4.2 Directional 3-Class Model & Feature Ablation Results (Frozen Test, `n=4,035`)

| Model & Feature Set | Balanced Acc | Macro-F1 | MCC | Brier Score | Log Loss | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_majority_prior` | 0.3333 | 0.1836 | 0.0000 | 0.6618 | 1.0911 | 0.0211 |
| `baseline_event_family_mean` | 0.3325 | 0.1848 | -0.0112 | 0.6608 | 1.0895 | 0.0155 |
| `logistic_regression_combined` | 0.3653 | 0.3497 | 0.0402 | 0.6707 | 1.1033 | 0.0668 |
| `pytorch_mlp_combined` | 0.3711 | 0.3585 | 0.0493 | 0.6663 | 1.0965 | 0.0583 |
| `lightgbm_nlp_only` | 0.3520 | 0.3365 | 0.0275 | 0.6671 | 1.0993 | 0.0252 |
| `xgboost_nlp_only` | 0.3533 | 0.3388 | 0.0277 | 0.6667 | 1.0987 | 0.0213 |
| `catboost_nlp_only` | 0.3542 | 0.3396 | 0.0301 | 0.6672 | 1.0994 | 0.0187 |
| `lightgbm_market_only` | 0.3840 | 0.3715 | 0.0694 | 0.6620 | 1.0914 | 0.0123 |
| `xgboost_market_only` | 0.3873 | 0.3763 | 0.0759 | 0.6606 | 1.0891 | 0.0202 |
| `catboost_market_only` | 0.3898 | **0.3802** | 0.0785 | **0.6574** | **1.0837** | 0.0183 |
| `lightgbm_combined` | 0.3840 | 0.3695 | 0.0701 | 0.6609 | 1.0895 | **0.0068** |
| `xgboost_combined` | 0.3869 | 0.3712 | 0.0739 | 0.6614 | 1.0905 | 0.0094 |
| **`catboost_combined`** | **0.3906** | 0.3738 | **0.0802** | 0.6584 | 1.0852 | 0.0157 |
| `catboost_combined_calibrated` *(Val isotonic)* | 0.3584 | 0.2655 | 0.0654 | 0.6605 | 1.0952 | 0.0693 |

*(Note: CatBoost's native balanced probabilities were already well calibrated (`ECE=0.0157`–`0.0183`); isotonic recalibration on Val hurt ECE (`0.0693`) under regime shift into 2026.)*

### 4.3 Walk-Forward CV, Issuer Holdout, Quantiles, Volatility & Uncertainty
- **Grouped Bootstrap (`1,000` resamples, `Combined - Market-Only`):**
  - By calendar month (`8` months): `bacc_delta = +0.00205` (`95% CI [-0.01454, +0.02132]`), `mcc_delta = +0.00328` (`[-0.02229, +0.03273]`).
  - By issuer (`310` issuers): `bacc_delta = +0.00034` (`95% CI [-0.01542, +0.01558]`).
- **3-Fold Expanding Purged Walk-Forward CV (`20d` embargo):**
  - Fold 1 (`2024-04-02..2024-10-24`, `3,415` val): Combined `0.3661` vs Market-Only `0.3707`
  - Fold 2 (`2025-01-24..2025-08-18`, `4,132` val): Combined **`0.3918`** vs Market-Only `0.3812`
  - Fold 3 (`2025-10-31..2026-05-29`, `3,825` val): Combined **`0.3981`** vs Market-Only `0.3857`
- **Unseen Issuer Holdout (`70` held-out issuers, `958` test events):** `bacc = 0.3788`, `MCC = 0.0705` (no severe issuer overfitting).
- **Regime Performance (Test):** `bull` (`0.4171`), `high_vol` (`0.3960`), `sideways` (`0.3840`), `bear` (`0.3541`).
- **Return Quantile Regression (`LightGBM`, 5d executable excess):** Combined `pinball_q10 = 0.00895`, `pinball_q50 = 0.01989`, `pinball_q90 = 0.01054`, `MAE_q50 = 0.03979`, **`80% Interval Coverage = 0.7879`** (vs Market-Only `0.7792` coverage, `0.03984` MAE).
- **Volatility Expansion (`vol_expansion_5d > 1.25`):** Combined **`ROC-AUC = 0.7028`, `Balanced Acc = 0.6352`, `Brier = 0.2046`** (vs Market-Only `0.6898` AUC, `0.6271` BAcc, `0.2112` Brier).
- **Abstention Curve:** Directional accuracy stays flat (`0.3506`–`0.3584`) across `100%`..`30%` coverage on calibrated probabilities, reinforcing that strong directional confidence cannot be artificially forced out of noisy equity drift.

---

## 5. Research Backtesting & Transaction-Cost Sensitivity (`phase3/results/backtest_report.json`)

Strict post-announcement entry (`Open[T0]` or `Close[T0]` for intraday), liquidity filter `>= INR 50 lakh` 20d avg daily turnover, max 5 daily positions, 1/5th capital per 5-day cohort:

| Strategy & Cost | Split | Trades | Gross Return | Net Return | Nifty Bench | Sharpe | Sortino | Max DD | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `combined_long_only_20bps` | Frozen Test | 86 | +1.95% | -0.03% | -1.33% | +0.087 | +0.152 | -9.56% | 50.0% |
| `combined_long_only_30bps` | Frozen Test | 86 | +1.95% | **-1.01%** | -1.33% | **-0.171** | -0.294 | -10.14% | 48.8% |
| `combined_long_only_50bps` | Frozen Test | 86 | +1.95% | -2.93% | -1.33% | -0.686 | -1.164 | -11.29% | 45.4% |
| `market_only_long_30bps` | Frozen Test | 136 | +5.93% | **+2.43%** | -0.24% | **+0.601** | +1.014 | -12.39% | 50.0% |
| `combined_long_short_excess_30bps` | Frozen Test | 831 | -6.40% | **-14.41%** | -2.27% | **-2.480** | -3.881 | -17.31% | 49.0% |
| `combined_long_only_30bps` | Walk-Forward | 1,070 | +25.77% | **+3.81%** | +5.86% | +0.249 | +0.523 | -27.41% | 46.0% |
| `market_only_long_30bps` | Walk-Forward | 1,184 | -8.73% | **-24.86%** | +7.72% | -1.269 | -2.250 | -31.48% | 45.3% |

*(Combined outperformed Market-Only on 2024–2026 Walk-Forward CV `+3.81% vs -24.86%`, but trailed Market-Only `+2.43% vs -1.01%` on the Frozen Test `2026-03..2026-10` window, confirming regime instability of directional alpha after costs.)*

---

## 6. Edge Deployment & Artifact Manifest (`models/phase3/`)

| Artifact | Size (Bytes) | SHA-256 | Edge Latency / Memory |
| :--- | :---: | :--- | :--- |
| `models/phase3/catboost_direction_combined.onnx` | `657,647` | `de0f2edc4de2cc261e5a65d9d20bb394e32a85da216d1023da0347a68ab0357e` | Standalone pure `.venv` end-to-end (`Phase 1+2 ONNX + 26,186-row similarity + Phase 3 ONNX`): **`55.05 ms` p50 / `82.35 ms` p95 (`17.33 docs/s`)**, **`393.9 MB` peak RSS**, `1.372 s` cold load |
| `models/phase3/catboost_volatility_combined.onnx` | `995,301` | `e803984b5b4b5984c2a50703effa83cc93ed0541850a0c5fe0d46daa18257aba` | Included inside above combined end-to-end ORT benchmark (`0.053 ms` p50 model-only) |

---

## 7. Reproducible PowerShell Commands

```powershell
training\.venv\Scripts\python.exe -m phase3.collect_nse_history
training\.venv\Scripts\python.exe -m phase3.collect_market_data
training\.venv\Scripts\python.exe -m phase3.event_alignment
training\.venv\Scripts\python.exe -m phase3.event_study
training\.venv\Scripts\python.exe -m phase3.build_features
training\.venv\Scripts\python.exe -m phase3.train_and_eval
training\.venv\Scripts\python.exe -m phase3.backtest
.\.venv\Scripts\python.exe -m phase3.measure_edge_runtime
.\.venv\Scripts\python.exe -m unittest phase3\tests\test_phase3_pipeline.py -v
```
