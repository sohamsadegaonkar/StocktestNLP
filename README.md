# StockBrief Edge

**A stock-market disclosure triage workbench with local NLP and a simulated edge device.**

NSE watchlist researchers often need to decide which announcements to open and verify. StockBrief classifies a disclosure, extracts the stated event stage and figures, and shows the exact passages used. It keeps working on local documents while its simulated inbound feed is disconnected.

[Product and system design](docs/DESIGN.md) · [Evaluation and jury demo](docs/DEMO.md) · [Editable Figma design](https://www.figma.com/design/bIwqXWeBhLKpGqO1IIj07s)

## Run locally

Requires Python 3.11 or 3.12. First-time dependency installation needs internet. After setup the default workflow runs without cloud credentials or external assets.

### Windows PowerShell

Open PowerShell in the repository folder:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m stockbrief serve
~~~

### macOS and Linux

~~~bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m stockbrief serve
~~~

Open [http://127.0.0.1:5050](http://127.0.0.1:5050). The app stores imports, analyses, and simulation state in `.runtime/stockbrief.sqlite` and reopens that file at the next launch. Pass `--db PATH` for an isolated database or `--port 5051` to use a different local port. It is a single-user, loopback application.

| View | What you can do |
| --- | --- |
| Inbox | View NSE price charts, filter disclosures, and inspect the event, figures, and supporting source. |
| Analyze | Paste English text or import a UTF-8 TXT or text-based PDF. Choose an issuer and optionally provide the disclosure date and source link. |
| Edge lab | Disconnect or reconnect the source link, advance replay packets, and see queue state on the virtual 128 × 64 display. |
| Evaluation | Compare the learned model with keyword matching. Inspect class-level results, matrix, abstention, timings, and source-backed examples. |

PDF import is limited to 5 MB and 30 pages. OCR is outside this release. Historical fixtures use short, attributed paraphrases with the original URLs. The display preserves the text of those fixtures or of your imported pages. Reading a historical fixture’s original URL requires a connection.

### NSE charts

The localhost Inbox fetches public Upstox candle data for Nifty 50, Tata Motors Passenger Vehicles, Infosys, Reliance, TCS, and HDFC Bank. The chart uses daily closes for **1M** and five-minute closes for **1D**. Outside the current trading session, **1D** can show the latest available session. The candle timestamp is displayed in IST. The chart refreshes once a minute while the dashboard is visible and labels cached data after a connection failure.

An internet connection is needed to fetch candles. The chart calls the provider with the selected instrument and period only. For an entirely offline demo, set `$env:STOCKBRIEF_DISABLE_MARKET = "1"` in PowerShell before startup. Disclosure analysis, the saved inbox, and replay run locally.

[Upstox historical candles](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/) · [Upstox intraday candles](https://upstox.com/developer/api-documentation/v3/get-intra-day-candle-data/)

## Models

The default pipeline fits word and character TF-IDF with logistic regression on 84 original disclosures. It evaluates document context and individual chunks, routes ambiguous outputs to review, and extracts page-level evidence. Event stage and figures are separate deterministic rules with provenance. The stage label reflects the text of that disclosure and is not updated from later events.

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

The benchmark writes JSON with model metrics, the confusion matrix, four source-backed checks, process RSS, and latency. See the [default test run](docs/evaluation.json) and the [one-CPU MobileBERT test run](docs/transformer-evaluation.json).

| Model | Macro F1 | Coverage | Ambiguous cases sent to review |
| --- | ---: | ---: | ---: |
| Keyword baseline | 0.881 | 75.0% | 4 of 4 |
| TF-IDF + logistic regression | 0.933 | 93.8% | 2 of 4 |
| MobileBERT INT8 | 0.516 | 40.6% | 4 of 4 |

This 32-case authored holdout favors the lighter model as the default for category coverage and macro F1. MobileBERT gives a second reading when a user wants to cross-check the category. These are small-sample host measurements. The edge device and its source link remain simulated.

In GitHub Actions, run **Checks** with the **transformer** input enabled to install the pinned weights, exercise the local second-opinion API, and save the evaluation JSON in the workflow artifact.

## Optional hosted mode

The hosted workspace serves the frontend and API from the same HTTPS origin. It opens with example disclosures and gives each browser its own research inbox and replay queue. Upload a text-based PDF or TXT, analyze it, inspect the extracted evidence, and run the edge simulation or evaluation from the same interface.

Hosted inference runs on the server. The link outage and 128 × 64 display simulate edge behavior. The default local mode runs inference on your own machine and continues working when the machine loses Internet access.

To host a separate instance, install the base requirements, set `STOCKBRIEF_PUBLIC_ORIGIN` to the canonical HTTPS URL, and start `python -m stockbrief serve --hosted`. The server uses `PORT` when provided. On Render it can use the platform's `RENDER_EXTERNAL_URL`.

The [Render Blueprint](render.yaml) defines a Free web service with an HTTP health check at `/api/health`.

For persistent storage, mount a volume and pass `--db /your-volume/stockbrief.sqlite`, or install `requirements-hosted.txt` and set `STOCKBRIEF_DATABASE_URL` to a dedicated PostgreSQL connection string. Keep the connection string in the provider's environment settings. Hosted workspace access uses a browser cookie valid for seven days. The default SQLite database on a Free Render web service is temporary. Download important results before the service goes idle or redeploys.
