# Historical Event Comparisons (Phase 3 Integration)

StockBrief Edge surfaces read-only **Similar past events** beside each analyzed or saved disclosure. This feature integrates the curated historical NSE alignment archive produced on `ml/phase3-market-reaction` without placing directional probabilities, price targets, or trading signals in the product interface.

## Research Handoff Decision

Phase 3 research evaluated event-driven directional and return prediction models (`0.3906` vs `0.3898` market-only directional accuracy on held-out windows, confidence intervals spanning zero, and worse post-cost test return). Consistent with that finding, **StockBrief Edge retains historical analytics only**:

- Shows matched prior NSE announcements alongside their **observed** 1-day (`T0`) and 5-day (`T0..T0+4`) market reactions.
- Separates **Raw return**, **Excess vs Nifty 50 (BHAR)**, and **Market-model CAR** into distinct labeled metrics.
- Explicitly flags **Upstox unadjusted daily closes fallback** (for example `TMPV` and `TCS` where Yahoo split/dividend-adjusted series were unavailable), near-event split/dividend boundaries, and overlapping filings within 10 trading sessions.
- Withholds comparisons (`insufficient_evidence`) whenever an analyzed text falls into `Needs review` or fails similarity thresholds.

## Research Commit Traceability

Research branches are preserved untouched; only clean app-facing artifacts and runtime loaders live on `feat/stockbrief-localhost-complete`:

| Branch | Verified Commit | Role |
| --- | --- | --- |
| `ml/domain-event-training` | `44935c81736686787ecf34b1b1fdc8f1e3a762ff` | Phase 1 event dataset and benchmark analysis |
| `ml/phase2-financial-reasoning` | `01b1f6a5a166c9a6d7b32ecab847280cd4f556b5` | Phase 2 multi-task financial reasoning experiments |
| `ml/phase3-market-reaction` | `3e1dc187ee2aa3be50dcb007eb379b8f6de62eb0` | Phase 3 event-study alignment and analytics |

Excluded non-redistributable datasets (`Financial PhraseBank` NC-SA, `FinRED` academic), raw PDFs, large checkpoints, and private inputs are **not** included in the runtime repository or historical bundle.

## Reproducing the Local Historical Bundle (Windows PowerShell)

1. The offline redistributable demo subset (`stockbrief/historical_demo.json`, 32 source-backed public NSE announcements, ~46.0 KiB) is included in Git so a fresh checkout works without downloading or exporting Phase 3 caches.
2. When local Phase 3 research caches exist (`phase3/.cache/features_table.parquet`, `phase3/.cache/p1_embs.npy`, `phase3/.cache/market/indices.pkl`), export the slim runtime archive (~22.28 MiB across `26,185` deduplicated events and `310` issuers from `2022-01-01` to `2026-10-23`):

~~~powershell
.\training\.venv\Scripts\python.exe -m stockbrief.historical_export --write-demo
~~~

Or via the CLI entry point (when run in an environment with `pandas` and `pyarrow`):

~~~powershell
.\training\.venv\Scripts\python.exe -m stockbrief export-historical --write-demo
~~~

This generates gitignored files under `.runtime/historical/phase3_package/`:
- `manifest.json` (schema `stockbrief-historical-v1`, SHA-256 checksums, encoder metadata)
- `events.jsonl.gz` (~3.10 MiB compressed event records)
- `embeddings_f16.npy` (~19.18 MiB float16 L2-normalized vectors from `Xenova/bge-small-en-v1.5` ONNX INT8 `6c9c6101...`)

## Leakage Prevention & Compatibility Guards

`stockbrief/historical.py` (`HistoricalIndex`) enforces:

1. **Publication cutoff**: All retrieved filings satisfy `published_ts_ist < cutoff_ist`. For date-only disclosures (`disclosure_date` without time of day), StockBrief applies a conservative start-of-day IST cutoff (`00:00:00+05:30`), excluding same-day filings.
2. **Outcome-window cutoff**: Each comparable's 1-day end date (`window_1d_end_date`) and 5-day end date (`window_5d_end_date`) must finish strictly before `cutoff_date` (`< cutoff_date`). Windows finishing on or after `cutoff_date` are returned as `null` (`unavailable_before_cutoff`) and excluded from horizon summary medians/IQRs.
3. **Self/duplicate exclusion**: Excludes matching source URLs, same-issuer same-day entries, and high-overlap texts (`Jaccard >= 0.72`).
4. **Encoder compatibility check**: Verifies active BGE ONNX `weights_sha256` and `384`-d vector shape against `manifest.json`; degrades cleanly to explicit **category matching fallback** if weights differ or `.npy` vectors are missing/corrupt.

## Configuration Flags

- `$env:STOCKBRIEF_DISABLE_HISTORICAL = "1"` — Disables historical comparisons (`state: "disabled"`).
- `$env:STOCKBRIEF_HISTORICAL_DIR = "PATH"` — Points to an alternate exported `phase3_package` directory.
- `$env:STOCKBRIEF_DISABLE_MARKET = "1"` — Runs without upstream Upstox chart/market-context network requests (recorded Infosys example remains available).
- `$env:STOCKBRIEF_NSE_ATTACHMENTS = "1"` — Enables optional NSE attachment PDF download on RSS polling.

## HTTP API Endpoints

- `GET /api/historical/status` — Active bundle mode (`phase3_local`, `offline_demo`, `disabled`, or `unavailable`), record count, date range, and encoder compatibility.
- `GET /api/disclosures/{entry_id}/comparables?limit=6` — Point-in-time historical comparisons, completed-window horizon summaries, and provenance limitations for a saved disclosure.
