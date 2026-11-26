# StockBrief Edge

**A stock-market disclosure triage workbench with local NLP and a simulated edge device.**

NSE watchlist researchers often need to decide which announcements to open and verify. StockBrief classifies a disclosure, extracts the stated event stage and figures, and shows the exact passages used. It keeps working on local documents while its simulated inbound feed is disconnected.

[Product and system design](docs/DESIGN.md) · [Historical comparisons & Phase 3 handoff](docs/HISTORICAL_COMPARISONS.md) · [Evaluation and jury demo](docs/DEMO.md) · [Editable Figma design](https://www.figma.com/design/bIwqXWeBhLKpGqO1IIj07s)

## Run locally

Requires Python 3.11 or 3.12. Install dependencies and download the verified semantic model once while online. The Analyze workflow runs on CPU from local model files after setup.

### Windows PowerShell

Open PowerShell in the repository folder:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m stockbrief fetch-semantic
.\.venv\Scripts\python.exe -m stockbrief serve
~~~

### macOS and Linux

~~~bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m stockbrief fetch-semantic
./.venv/bin/python -m stockbrief serve
~~~

Open [http://127.0.0.1:5050](http://127.0.0.1:5050). The app stores imports, analyses, and simulation state in `.runtime/stockbrief.sqlite` and reopens that file at the next launch. Pass `--db PATH` for an isolated database or `--port 5051` to use a different local port. It is a single-user, loopback application.

| View | What you can do |
| --- | --- |
| Inbox | View NSE charts, fetch official filings, and inspect a local brief with source evidence. |
| Analyze | Paste English text or import a UTF-8 TXT or text-based PDF. Choose an issuer and optionally provide the disclosure date and source link. |
| Edge lab | Disconnect or reconnect the source link, advance replay packets, and see queue state on the virtual 128 × 64 display. |
| Evaluation | Compare the learned model with keyword matching. Inspect class-level results, matrix, abstention, timings, and source-backed examples. |

PDF import is limited to 5 MB and 30 pages. OCR is outside this release. Historical fixtures use short, attributed paraphrases with the original URLs. The display preserves the text of those fixtures or of your imported pages. Reading a historical fixture’s original URL requires a connection.

### NSE feed and charts

Open **Inbox → Fetch now** to check official NSE announcements. When the exchange snapshot has zero tracked companies or the feed cannot be reached, the UI shows which happened and offers **Find recent publisher headlines**. This optional Google News RSS lookup retrieves the last seven days of linked headlines by company. Those results stay out of the official filing queue.

The chart appears at the top of the Inbox and is reachable from **Charts**. The localhost chart requests Upstox candle data for Nifty 50, Tata Motors Passenger Vehicles, Infosys, Reliance, TCS, and HDFC Bank. The chart uses daily closes for **1M** and five-minute closes for **1D**. Outside the current trading session, **1D** can show the latest available session. The candle timestamp is displayed in IST. The chart refreshes once a minute while the dashboard is visible and labels cached data after a connection failure. **Recorded Infosys case** opens an attributed historical chart and first- and fifth-session comparisons against Nifty 50, even without an internet connection. On the initial load, the recorded case is shown with a clear label if the selected market feed returns no prices. **Refresh prices** retries the latest chart. Open a saved or pasted disclosure with a supported ticker and publication date to request **Compare market closes**.

An internet connection is needed to fetch candles. The chart calls the provider with the selected instrument and period only. For an entirely offline demo, set `$env:STOCKBRIEF_DISABLE_MARKET = "1"` in PowerShell before startup (or `$env:STOCKBRIEF_DISABLE_HISTORICAL = "1"` to disable historical comparisons). Disclosure analysis, the saved inbox, and replay run locally.

### Similar past events (historical comparisons)

Opening any analyzed or saved disclosure renders a read-only **Similar past events** panel below **Observed market context** (analytics only; no directional probabilities or trading advice):
- Out of the box, StockBrief uses `stockbrief/historical_demo.json` (32 source-backed public NSE announcements, ~46.0 KiB).
- When local Phase 3 caches are present (`phase3/.cache/`), export the compact local archive (`26,185` filings across `310` issuers, ~22.28 MiB under `.runtime/historical/phase3_package`) with `.\training\.venv\Scripts\python.exe -m stockbrief.historical_export --write-demo`.
- Queries enforce strict publication cutoffs (conservative `00:00:00+05:30` when time of day is absent) and exclude any 1-day or 5-day outcome window that finishes on or after the cutoff date. See [Historical comparisons](docs/HISTORICAL_COMPARISONS.md) for compatibility checks and provenance flags.

[Upstox historical candles](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/) · [Upstox intraday candles](https://upstox.com/developer/api-documentation/v3/get-intra-day-candle-data/)

## Models

The default pipeline encodes disclosure passages with **BGE-small-en-v1.5 INT8** via ONNX Runtime on CPU and fits a supervised classification head on 122 labeled examples. It groups supported passages into event cards by category so mixed disclosures can show multiple categories. It routes unclear inputs to review with source excerpts. The app extracts stated stages, figures, and dates with their original wording.

The one-time download verifies a pinned copy of the roughly 34 MB encoder and its tokenizer. The model files are saved under `models/semantic`. Resubmitting saved text can refresh its analysis using the current model and the existing inbox entry.

The optional transformer runs quantized MobileBERT MNLI through ONNX Runtime on CPU. It scores seven event hypotheses against up to the first six chunks as a second opinion. The downloader uses a pinned model revision and the app verifies the weights against the published SHA-256 before loading them. The served comparator reads only local model files.

[Antigravity implementation prompt for model training and mixed disclosures](docs/ANTIGRAVITY_MODEL_PROMPT.md)

~~~bash
python -m pip install -r requirements-transformer.txt
python -m stockbrief fetch-model --transformer models/mobilebert
python -m stockbrief evaluate --transformer models/mobilebert
python -m stockbrief serve --transformer models/mobilebert
~~~

## Verify and reproduce

~~~bash
python -m unittest discover -v
python -m stockbrief evaluate --cpus 1  # Linux affinity; omit --cpus elsewhere
~~~

The benchmark writes JSON with model metrics, the confusion matrix, four source-backed checks, process RSS, and latency. See the [semantic model run](docs/semantic-evaluation.json) and the [earlier MobileBERT run](docs/transformer-evaluation.json).

| Model | Macro F1 | Coverage | Ambiguous cases sent to review |
| --- | ---: | ---: | ---: |
| Keyword baseline | 0.881 | 75.0% | 4 of 4 |
| TF-IDF + logistic regression | 0.933 | 93.8% | 2 of 4 |
| MobileBERT INT8 | 0.516 | 40.6% | 4 of 4 |
| BGE-small INT8 + supervised classifier | 1.000 | 87.5% | 4 of 4 |

BGE had the highest macro F1 on the 32 authored holdout cases and also classified eight additional pricing and review fixtures correctly. The original holdout contains 28 substantive and four review cases. The small authored suite measures project behavior, and a larger independent filing sample would be needed to estimate accuracy in wider use. The edge device and its source link are simulated.

In GitHub Actions, run **Checks** with the **transformer** input enabled to install the pinned weights, exercise the local second-opinion API, and save the evaluation JSON in the workflow artifact.

## Optional hosted mode

The hosted workspace serves the frontend and API from the same HTTPS origin. It opens with example disclosures and gives each browser its own research inbox and replay queue. Upload a text-based PDF or TXT, analyze it, inspect the extracted evidence, and run the edge simulation or evaluation from the same interface.

Hosted inference runs on the server. The link outage and 128 × 64 display simulate edge behavior. The default local mode runs inference on your own machine and continues working when the machine loses Internet access.

To host a separate instance, install the base requirements, download the BGE model with `python -m stockbrief fetch-semantic`, set `STOCKBRIEF_PUBLIC_ORIGIN` to the canonical HTTPS URL, and start `python -m stockbrief serve --hosted`. The server uses `PORT` when provided. On Render it can use the platform's `RENDER_EXTERNAL_URL`.

The [Render Blueprint](render.yaml) defines a Free web service with an HTTP health check at `/api/health`.

For persistent storage, mount a volume and pass `--db /your-volume/stockbrief.sqlite`, or install `requirements-hosted.txt` and set `STOCKBRIEF_DATABASE_URL` to a dedicated PostgreSQL connection string. Keep the connection string in the provider's environment settings. Hosted workspace access uses a browser cookie valid for seven days. The default SQLite database on a Free Render web service is temporary. Download important results before the service goes idle or redeploys.
