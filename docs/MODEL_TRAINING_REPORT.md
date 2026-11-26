# StockBrief Domain Event Training & Evaluation Report

## 1. Executive Summary & Promotion Decision

We curated a **447-document** (465 curated, 18 exact/near-duplicates removed) dataset of real official Indian listed-company announcements from **NSE India Corporate Announcements** across **203 issuers**, preserving exact source metadata (`doc_id`, official PDF URL, issuer symbol, ISO date), event stages, character-level evidence offsets, amounts/dates/qualifiers/negations, headline-only/vague clarifications, rumor denials, scheduled vs. reported notices, mixed multi-event disclosures (`pricing + earnings`, `earnings + corporate_action`), and real failed PDF extractions.

On the **118-document frozen chronological + unseen-issuer holdout** (`48` issuers, including `19` completely unseen issuers accounting for `69` documents):
- **Fine-tuned `answerdotai/ModernBERT-base` (ONNX INT8)** improved held-out **Macro-F1 from `0.6189` to `0.6842`** (`+0.0653`; FP32 reached `0.7420`), **Review Recall from `0.4118` to `1.0000`** (`+0.5882`), **Review Precision from `0.3333` to `0.7727`** (`+0.4394`), and **Multilabel Micro-F1 from `0.5448` to `0.7055`** (`+0.1607`).
- **Frozen `ModernBERT-base` + LogisticRegression head (ONNX INT8)** reached `macro_f1 = 0.5613` (`0.5812` FP32), losing to the pinned BGE baseline on macro-F1 while achieving higher review recall (`0.8235`).
- **Promotion Gate Result (`promote_as_default = False`)**: **Keep the pinned BGE-small-en-v1.5 INT8 (`stockbrief-bge-events-v1`) as the production default.** Although the fine-tuned ModernBERT-base INT8 model outperforms BGE on real-world NSE macro-F1, review precision/recall, multilabel F1, and negation handling, it **exceeds the edge CPU budget** (`p95 = 544.1 ms` vs `<= 150 ms` gate; `143.54 MiB` vs `<= 80 MB` gate) and scores lower on out-of-distribution short synthetic repo fixtures (`repo_test_32` macro-F1 `0.5107` vs `1.0000`).

---

## 2. Reproducible PowerShell Commands

> Note on branch baseline: local `main` was at `304893b`, while `origin/main` was at `3868a68` (introducing `stockbrief/semantic_data.py`, `stockbrief/semantic.py`, and pinned BGE INT8). Branch `ml/domain-event-training` was created from `origin/main` (`3868a68`), leaving local `main`, `stockbrief/`, and `tests/` untouched.

```powershell
# 1. Verify existing baseline app + BGE INT8 (in .venv)
uv pip install --python .\.venv\Scripts\python.exe "onnxruntime>=1.19,<2" "tokenizers>=0.20,<1"
.\.venv\Scripts\python.exe -m stockbrief fetch-semantic
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m stockbrief evaluate

# 2. Create isolated training environment (in training/.venv)
$env:UV_CACHE_DIR = "D:\StocktestNLP\training\.cache\uv"
uv venv training\.venv --python 3.12
uv pip install --python training\.venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python training\.venv\Scripts\python.exe transformers accelerate onnx onnxruntime onnxscript scikit-learn httpx pypdf psutil tokenizers sentence-transformers

# 3. Collect, curate, deduplicate, and split NSE announcements
.\.venv\Scripts\python.exe -m training.fetch_nse_index
.\.venv\Scripts\python.exe -m training.download_candidates
.\.venv\Scripts\python.exe -m training.fetch_targeted
.\.venv\Scripts\python.exe -m training.fetch_scheduled_and_paired
.\.venv\Scripts\python.exe -m training.curate_dataset
.\.venv\Scripts\python.exe -m training.split_dataset

# 4. Score pinned BGE INT8 baseline on frozen splits + untouched repo benchmarks
.\.venv\Scripts\python.exe -m training.eval_baseline

# 5. Fetch ModernBERT-base, train frozen-head & fine-tuned models, export ONNX FP32 + INT8, evaluate
training\.venv\Scripts\python.exe -m training.fetch_modernbert
training\.venv\Scripts\python.exe -m training.train_models

# 6. Run isolated fresh-process CPU latency & peak RAM benchmarks (1 thread)
.\.venv\Scripts\python.exe -m training.bench_runtime

# 7. Run manual gated retrain CLI (never auto-trains, never schedules)
training\.venv\Scripts\python.exe -m training.retrain --arch modernbert-frozen --version retrain-gate-check-v1
# Optional with new human-reviewed JSONL:
# training\.venv\Scripts\python.exe -m training.retrain --reviewed-jsonl path\to\reviewed.jsonl --arch modernbert-finetuned --version candidate-v2

# 8. Run training pipeline tests
.\.venv\Scripts\python.exe -m unittest training\tests\test_training_pipeline.py -v
```

---

## 3. Dataset Manifest, Splits, Balance & License Notes

- **Dataset Version**: `nse-domain-events-v1`
- **Primary Source**: Official NSE India Corporate Announcements API & Archives (`https://www.nseindia.com/companies-listing/corporate-filings-announcements`, `https://nsearchives.nseindia.in/corporate/*.pdf`), fetched across `2025-07-01` to `2026-11-25`.
- **License & Provenance Notes**: Public exchange regulatory disclosures filed under SEBI (LODR) Regulations, 2015. Every record stores official `doc_id` (`nse:<seq_id>`), official filing `url`, `issuer_symbol`, `issuer_name`, ISO `date`, `nse_desc`, `event_family`, `source_mode`, `stage`, exact `evidence_offsets` (`text[start:end] == quote`), `insufficiency_type`, and `preserves` (`amounts`, `dates`, `qualifiers`, `negations`). **Zero user uploads** are included. Repo benchmark fixtures (`stockbrief/corpus.py`, `stockbrief/semantic_data.py`, `stockbrief/fixtures.py`, `tests/test_semantic.py` `TATA` text) remain completely untouched and excluded.
- **Deduplication**: `465` curated records -> `15` exact duplicates removed + `3` near-duplicates removed (4-gram word Jaccard `>= 0.85` within label) -> **`447` deduplicated documents** across **`203` issuers**.
- **Split Policy**: Grouped strictly by `event_family` (zero `event_family` or text-fingerprint overlap across splits). **19 issuers** (`ABB`, `ACLGATI`, `ADANIPORTS`, `ALKEM`, `BANARBEADS`, `BEL`, `DIXON`, `DMART`, `DRREDDY`, `EICHERMOT`, `INDHOTEL`, `IRCTC`, `M&M`, `NATCOPHARM`, `PHOENIXLTD`, `TECHM`, `TITAN`, `TVSMOTOR`, `VOLTAS` = **69 documents**) are held out completely from `train` and `calibration` into `test`. Remaining families are split chronologically within each stratum (`train` earliest -> `calibration` middle -> `test` latest).

### Split & File Manifest

| File | Records | Issuers | Date Window | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `training/data/nse_announcements.jsonl` | 465 | 203 | 2025-07-01 .. 2026-11-19 | `394ed1017c7328d0b72b3f34ad70db6cc107bc770bc23b16e9f12979911350cd` |
| `training/data/train.jsonl` | 244 | 160 | 2025-07-01 .. 2026-04-16 | `666dcd2f7775ad1aa60fb52e6d49174cc45c18e5ef3845c2eb530dffcc8a381e` |
| `training/data/calibration.jsonl` | 85 | 51 | 2025-07-29 .. 2026-09-11 | `44261df9543e5760dcbbd2721efa88dcbcab17fc3e060bf9b6c668ffec61f41a` |
| `training/data/test.jsonl` | 118 | 48 | 2025-07-01 .. 2026-11-19 | `3b4a68b23833837a7ce16a40d82d9d707f07f37ee0066ff2337ac356afdb3bad` |

### Label & Special-Slice Distribution

| Category / Slice | Train (244) | Calibration (85) | Test (118) | Total (447) |
| --- | ---: | ---: | ---: | ---: |
| `earnings` (Earnings & guidance) | 28 | 10 | 12 | 50 |
| `corporate_action` (Corporate action) | 30 | 10 | 10 | 50 |
| `deal` (M&A & investment) | 26 | 9 | 17 | 52 |
| `governance` (Leadership & governance) | 25 | 9 | 15 | 49 |
| `regulatory` (Regulatory & legal) | 22 | 8 | 19 | 49 |
| `operations` (Operations) | 31 | 10 | 7 | 48 |
| `routine` (Routine notice) | 28 | 10 | 9 | 47 |
| `pricing` (Pricing & commercial updates) | 31 | 11 | 12 | 54 |
| `review` (Needs review) | 23 | 8 | 17 | 48 |
| *Multilabel documents* | 7 | 2 | 5 | 14 |
| *Mixed `pricing + earnings`* | 4 | 1 | 4 | 9 |
| *Rumor denials (`rumor_denial`)* | 10 | 5 | 7 | 22 |
| *Scheduled notices (`stage="Scheduled"`)* | 17 | 5 | 3 | 25 |
| *Failed PDF extraction (`pdf_extraction_failed`)* | 8 | 3 | 5 | 16 |
| *Headline-only (`headline_only`)* | 8 | 3 | 9 | 20 |

Chunk/sentence preparation on `train.jsonl` produced **553 deduplicated chunk/single-event training samples** (`earnings: 82`, `operations: 79`, `corporate_action: 74`, `deal: 65`, `pricing: 61`, `governance: 58`, `regulatory: 55`, `routine: 53`, `review: 26`).

---

## 4. Model Card (`stockbrief-modernbert-ft-v1`)

- **Model Details**: Fine-tuned `answerdotai/ModernBERT-base` (149M parameters, 768-dim mean pooling + linear 9-class head, `attn_implementation="eager"`), trained 4 epochs on CPU (`batch_size=8` dynamic padding, `lr=3e-5`, cosine warmup 10%, `label_smoothing=0.05`, capped inverse-frequency class weights `[1.0, 2.5]`), exported to ONNX opset 17 FP32 and dynamically quantized to INT8 (`per_channel=True, reduce_range=True` after `quant_pre_process`).
- **Calibration Thresholds (tuned ONLY on `calibration.jsonl`)**: `floor = 0.40`, `margin = 0.0` (frozen-head used `C=10.0`, `class_weight="balanced"`, `floor=0.30`, `margin=0.0`).
- **Intended Use**: Offline CPU triage of English Indian listed-company corporate disclosures into the 9 `SEMANTIC_LABELS` categories, paired with exact source evidence offsets, stated event stages, and preserved figures/dates.
- **Hardware & Runtime**: Benchmarked on Windows AMD64, 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (12 GB RAM), single-threaded ONNX Runtime CPU (`intra_op_num_threads=1`, `inter_op_num_threads=1`).
- **Limitations**:
  - At `143.54 MiB` INT8 and `p95 = 544.1 ms` per document on 1 CPU thread, `ModernBERT-base` is ~4.4x larger and ~4.4x slower than `BGE-small-en-v1.5` (`32.44 MiB`, `p95 = 123.6 ms`).
  - Dynamic INT8 quantization introduces measurable drift vs. FP32 (`101/118 = 85.59%` prediction agreement; macro-F1 drops from `0.7420` FP32 to `0.6842` INT8).
  - Trained exclusively on real NSE excerpts without mixing `SEMANTIC_TRAIN` authored sentences, so short synthetic authored sentences (`repo_test_32`) over-abstain to `review` (`macro_f1 = 0.5107`, `review_recall = 1.0000`, `review_precision = 0.2500`).
  - Scanned/non-English/legacy-font PDFs cannot be read by `pypdf` and are routed conservatively to `review`.

---

## 5. Raw Evaluation Metrics (Frozen 118-Doc Holdout & Repo Benchmarks)

### Overall Comparison on Frozen `test.jsonl` (118 docs, 48 issuers, 19 unseen)

| Metric | Pinned BGE INT8 (Baseline) | ModernBERT Frozen FP32 | ModernBERT Frozen INT8 | ModernBERT Fine-Tuned FP32 | ModernBERT Fine-Tuned INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Macro-F1 (9 classes)** | 0.6189 | 0.5812 | 0.5613 | **0.7420** | **0.6842** |
| Weighted-F1 | 0.6021 | 0.6166 | 0.5835 | 0.7602 | 0.6963 |
| Coverage (non-review) | 0.8220 | 0.7034 | 0.7373 | 0.8559 | 0.8136 |
| Selective accuracy | 0.6392 | 0.6988 | 0.6437 | 0.7525 | 0.7083 |
| **Review precision** | 0.3333 | 0.4857 | 0.4516 | **0.8824** | **0.7727** |
| **Review recall** | 0.4118 | 1.0000 | 0.8235 | 0.8824 | **1.0000** |
| **Multilabel Micro-F1** | 0.5448 | 0.6192 | 0.5978 | 0.6957 | **0.7055** |
| Multilabel Macro-F1 | 0.5917 | 0.5933 | 0.5725 | 0.6820 | 0.6778 |
| Stage accuracy | 0.3983 | 0.5424 | 0.5085 | 0.6356 | 0.6017 |
| Evidence accuracy | 0.8834 | 0.9509 | 0.8957 | 0.9632 | 0.9018 |
| Offset exact valid rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Numeric error rate | 3/34 (8.8%) | 3/34 (8.8%) | 4/34 (11.8%) | **2/34 (5.9%)** | 3/34 (8.8%) |
| Negation / rumor errors | 5/7 (71.4%) | **1/7 (14.3%)** | 2/7 (28.6%) | 3/7 (42.9%) | **1/7 (14.3%)** |
| Scheduled vs reported errors | 3/3 | 3/3 | 3/3 | **2/3** | **2/3** |
| Failed PDF accuracy | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| Mixed `pricing + earnings` exact | 2/4 | 2/4 | 1/4 | 2/4 | 2/4 |

### Per-Class Support & F1 on Frozen `test.jsonl`

| Class | Support | BGE INT8 (P / R / F1) | ModernBERT FT FP32 (P / R / F1) | ModernBERT FT INT8 (P / R / F1) |
| --- | ---: | --- | --- | --- |
| `earnings` | 12 | 0.857 / 0.500 / 0.632 | 0.769 / 0.833 / 0.800 | 0.667 / 0.667 / 0.667 |
| `corporate_action` | 10 | 1.000 / 0.500 / 0.667 | 0.727 / 0.800 / 0.762 | 0.615 / 0.800 / 0.696 |
| `deal` | 17 | 1.000 / 0.294 / 0.455 | 0.667 / 0.353 / 0.462 | 0.800 / 0.235 / 0.364 |
| `governance` | 15 | 1.000 / 0.600 / 0.750 | 0.833 / 1.000 / 0.909 | 0.790 / 1.000 / 0.882 |
| `regulatory` | 19 | 0.515 / 0.895 / 0.654 | 0.818 / 0.947 / 0.878 | 0.750 / 0.790 / 0.769 |
| `operations` | 7 | 1.000 / 0.714 / 0.833 | 0.444 / 0.571 / 0.500 | 0.500 / 0.571 / 0.533 |
| `routine` | 9 | 0.280 / 0.778 / 0.412 | 0.667 / 0.667 / 0.667 | 0.714 / 0.556 / 0.625 |
| `pricing` | 12 | 1.000 / 0.667 / 0.800 | 0.900 / 0.750 / 0.818 | 0.750 / 0.750 / 0.750 |
| `review` | 17 | 0.333 / 0.412 / 0.368 | 0.882 / 0.882 / 0.882 | 0.773 / 1.000 / 0.872 |

### Untouched Repo Benchmark Results

| Benchmark | Pinned BGE INT8 | ModernBERT Frozen INT8 | ModernBERT FT INT8 |
| --- | ---: | ---: | ---: |
| `corpus.TEST` (32 cases) Macro-F1 / Review Recall | **1.0000 / 1.0000** | 0.6143 / 1.0000 | 0.5107 / 1.0000 |
| `SEMANTIC_ADDITIONAL_TEST` (8 cases) Macro-F1 | **1.0000** | 0.6667 | 0.8036 |
| `fixtures.CASE_STUDY` (4 cases) | **4/4** | 0/4 | 2/4 |
| `TATA` Mixed Pricing+Earnings Acceptance | **PASS** | **PASS** | **PASS** |

---

## 6. Confusion Matrices (Frozen `test.jsonl`, 118 Docs)

Class order: `[earnings, corporate_action, deal, governance, regulatory, operations, routine, pricing, review]`

### Pinned BGE-small-en-v1.5 INT8 (Baseline)
```text
[[ 6,  0,  0,  0,  2,  0,  4,  0,  0],
 [ 0,  5,  0,  0,  0,  0,  2,  0,  3],
 [ 0,  0,  5,  0,  6,  0,  1,  0,  5],
 [ 0,  0,  0,  9,  4,  0,  0,  0,  2],
 [ 0,  0,  0,  0, 17,  0,  1,  0,  1],
 [ 0,  0,  0,  0,  1,  5,  0,  0,  1],
 [ 0,  0,  0,  0,  2,  0,  7,  0,  0],
 [ 1,  0,  0,  0,  1,  0,  0,  8,  2],
 [ 0,  0,  0,  0,  0,  0, 10,  0,  7]]
```

### ModernBERT-base Fine-Tuned ONNX INT8
```text
[[ 8,  0,  0,  1,  1,  0,  2,  0,  0],
 [ 1,  8,  1,  0,  0,  0,  0,  0,  0],
 [ 0,  3,  4,  0,  3,  3,  0,  0,  4],
 [ 0,  0,  0, 15,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  1, 15,  1,  0,  2,  0],
 [ 0,  1,  0,  0,  1,  4,  0,  1,  0],
 [ 1,  0,  0,  2,  0,  0,  5,  0,  1],
 [ 2,  1,  0,  0,  0,  0,  0,  9,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0, 17]]
```

---

## 7. ONNX INT8 Export, Quantization Parity & Isolated CPU Benchmarks

All runtime measurements below were executed in isolated fresh subprocesses with `threadpool_limits(limits=1)` and single-threaded ONNX Runtime CPU (`CPUExecutionProvider`) across the 118-document `test.jsonl` holdout:

| Model Artifact | Bytes (MiB) | p50 (ms) | p95 (ms) | Peak RSS (MiB) | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `models/semantic/onnx/model_quantized.onnx` (BGE INT8) | 34,014,426 (32.44 MiB) | **60.891** | **123.571** | **217.1** | `6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4` |
| `models/modernbert-frozen-head/model_quantized.onnx` (INT8) | 150,515,512 (143.54 MiB) | 314.076 | 542.650 | 407.9 | `7bd0409f1e7db613253ad1ac5ae8659527184460ff281c3778270943ca40cca9` |
| `models/modernbert-finetuned/model.onnx` (FP32) | 596,682,941 (569.04 MiB) | 759.089 | 1291.110 | 772.6 | `02af92e38c41e0222c606d8873c76398330640fedb0629a548e13ab00264e9e8` |
| `models/modernbert-finetuned/model_quantized.onnx` (INT8) | 150,515,107 (143.54 MiB) | 315.931 | 544.107 | 408.1 | `d63d359c7c41e86ba25a0fc280a04a4e61b954af5a6be01c3a12af52c718eb9f` |

- **PyTorch vs. ONNX FP32 Parity**: `max |diff| = 3e-6` (frozen-head) and `5e-6` (fine-tuned).
- **Quantized vs. Unquantized Prediction Agreement on `test.jsonl`**:
  - `modernbert-frozen-head`: **89/118 (75.42%)** (`macro_f1`: `0.5812` FP32 -> `0.5613` INT8).
  - `modernbert-finetuned`: **101/118 (85.59%)** (`macro_f1`: `0.7420` FP32 -> `0.6842` INT8; `review_recall`: `0.8824` -> `1.0000`; 2.40x faster p50, 3.96x smaller artifact).

---

## 8. Adapter Contract for `SemanticEventModel.analyze(document)` & Manual Retrain Policy

- **Adapter Contract (`training/adapter.py`)**: `DomainEventAdapter(ONNXEndToEndScorer(...), floor, margin).analyze(document)` implements the exact dictionary contract returned by `stockbrief.semantic.SemanticEventModel.analyze(document)` (`category`, `category_name`, `candidate`, `candidate_name`, `score`, `scores`, `stage`, `stage_evidence`, `evidence`, `figures`, `dates`, `events`, `chunk_count`, `chunk_limit_reached`, `model`, `model_version`, `inference_ms`), while adding deterministic unreadable/garbled-PDF handling and `Scheduled` / `Denied` stage coverage.
- **Promotion Gate & Manual Retrain (`training/retrain.py`)**:
  - Models are stored outside Git (`models/`).
  - Retraining is **manual-only** (`python -m training.retrain ...`), consumes only human-reviewed annotations (`reviewed=True`), explicitly rejects user uploads (`source == "user_upload"`), never auto-trains on incoming headlines, and installs no recurring schedule.
  - `models/registry.json` promotes a new candidate ONLY if all 7 gates pass (`improves_test_macro_f1`, `preserves_review_recall`, `improves_or_keeps_multilabel_f1`, `repo_tata_passed`, `repo_test_32_review_recall_ok`, `cpu_p95 <= 150 ms`, `int8_size <= 80 MB`).

### Promotion Gate Summary

| Gate | ModernBERT Frozen INT8 | ModernBERT Fine-Tuned INT8 |
| --- | --- | --- |
| Improves `test.jsonl` Macro-F1 (`> 0.6189`) | FAIL (`0.5613`) | **PASS** (`0.6842`) |
| Preserves Review Recall (`>= 0.4118`) | **PASS** (`0.8235`) | **PASS** (`1.0000`) |
| Multilabel Micro-F1 (`>= 0.5448`) | **PASS** (`0.5978`) | **PASS** (`0.7055`) |
| `TATA` Mixed Acceptance Example | **PASS** | **PASS** |
| `repo_test_32` Review Recall (`>= 0.75`) | **PASS** (`1.0000`) | **PASS** (`1.0000`) |
| CPU 1-Thread p95 (`<= 150.0 ms`) | FAIL (`542.65 ms`) | FAIL (`544.11 ms`) |
| INT8 Artifact Size (`<= 80 MB`) | FAIL (`143.54 MiB`) | FAIL (`143.54 MiB`) |
| **Promote as Default** | **NO** | **NO (Keep BGE INT8 default)** |
