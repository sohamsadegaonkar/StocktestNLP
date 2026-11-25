# StockBrief Edge

**A stock-market disclosure triage workbench with local NLP and a simulated edge device.**

NSE watchlist researchers often need to decide which announcements to open and verify. StockBrief classifies a disclosure, extracts the stated event stage and figures, and shows the exact passages used. It keeps working on local documents while its simulated inbound feed is disconnected.

[Product and system design](docs/DESIGN.md) · [Evaluation and jury demo](docs/DEMO.md) · [Editable Figma design](https://www.figma.com/design/bIwqXWeBhLKpGqO1IIj07s)

## Run locally

Requires Python 3.11 or 3.12. First-time dependency installation needs internet. After setup the default workflow runs without cloud credentials or external assets.

~~~bash
python -m venv .venv
source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m stockbrief serve
~~~

Open [http://127.0.0.1:5050](http://127.0.0.1:5050). The app stores imports, analyses, and simulation state in `.runtime/stockbrief.sqlite`. Pass `--db PATH` for an isolated database or `--port 5051` to use a different local port. It is a single-user, loopback application.

| View | What you can do |
| --- | --- |
| Inbox | Filter by ticker and category. Inspect the predicted event, stated stage, evidence offsets, extracted figures, and source. |
| Analyze | Paste English text or import a UTF-8 TXT or text-based PDF. Choose an issuer and optionally provide the disclosure date and source link. |
| Edge lab | Disconnect or reconnect the source link, advance replay packets, and see queue state on the virtual 128 × 64 display. |
| Evaluation | Compare the learned model with keyword matching. Inspect class-level results, matrix, abstention, timings, and source-backed examples. |

PDF import is limited to 5 MB and 30 pages. OCR is outside this release. Historical fixtures use short, attributed paraphrases with the original URLs. The display preserves the text of those fixtures or of your imported pages. Reading a historical fixture’s original URL requires a connection.

## Models

The default pipeline fits word and character TF-IDF with logistic regression on 84 original disclosures. It evaluates document context and individual chunks, routes ambiguous outputs to review, and extracts page-level evidence. Event stage and figures are separate deterministic rules with provenance. The stage label reflects the text of that disclosure and is not updated from later events.

The optional advanced path runs quantized MobileBERT MNLI through ONNX Runtime on CPU. It scores seven event hypotheses against up to the first six chunks as a second opinion. The comparator leaves the original brief and citation selection intact. The first download records the exact repository revision. The served comparator reads only local model files.

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

The benchmark writes JSON with model metrics, the confusion matrix, four source-backed checks, process RSS, and latency. Authored fixture results describe this small suite. The host numbers are neither device measurements nor production accuracy estimates. See [the released test run](docs/evaluation.json).
