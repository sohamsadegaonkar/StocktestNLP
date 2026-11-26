# Phase 2 — Advanced Financial Reasoning Model Training Report

**Branch:** `ml/phase2-financial-reasoning` (branched cleanly from Phase 1 commit `44935c8` on `ml/domain-event-training`)
**Selected Production Candidate Architecture:** **Architecture A** (`EdgeReasoningPipeline` — Multi-Task ONNX INT8 + Deterministic Financial Numerical Engine)
**Phase 1 Baseline Status:** Preserved intact (`models/registry.json` active default remains `models/semantic`, `stockbrief-bge-events-v1`).

---

## 1. Executive Summary & Hardware Constraints

Phase 2 extends StocktestNLP from single-disclosure event classification into multi-task financial understanding across entity extraction, relationship extraction, hard-negative passage retrieval, contextual financial impact analysis, deterministic numerical computation, prompt-injection defense, and unanswerable abstention.

### Detected Training & Inference Hardware
- **OS / Runtime:** Windows 11 (`10.0.26300`), Python `3.12`
- **CPU:** 8 logical cores (11th Gen Intel Core i5-1135G7 @ 2.40 GHz)
- **GPU / VRAM:** None (`torch.cuda.is_available() == False`, `0 GB` VRAM)
- **System RAM:** `12.59 GB` total (`~3.7 GB` available headroom before training)
- **Environments:**
  - `training\.venv` (`torch==2.15.0+cpu`, `transformers==5.24.1`, `peft==0.22.0`, `pyarrow==26.0.1`, `onnx`, `onnxruntime`, `scikit-learn`)
  - `.\.venv` (lean edge runtime: `onnxruntime`, `tokenizers`, `numpy`, `psutil`, zero PyTorch/Transformers dependency)

Because the target environment is CPU-only (`12.59 GB` RAM), all Phase 2 neural models were trained with resumable checkpointing (`checkpoint.pt`), exported to ONNX FP32 and dynamically quantized to ONNX INT8 (`QInt8`, per-channel, `reduce_range=True`), and executed in `EdgeReasoningPipeline` using pure Rust `tokenizers.Tokenizer` + `onnxruntime` without PyTorch in production.

---

## 2. Corpus Expansion, License Isolation & Zero-Leakage Audit

### 2.1 Dataset Sources, Provenance & Licenses
All external datasets were downloaded with SHA-256 verification and tracked in `phase2/.cache/external/external_provenance.json`:

| Source | Artifact(s) | License | Incorporation Policy |
| :--- | :--- | :--- | :--- |
| **Phase 1 Curated NSE Corpus** | `training/data/train.jsonl`, `cal.jsonl`, `test.jsonl` | Public Exchange Filings + Original Annotations | Reused cleanly; test/held-out issuers preserved strictly disjoint from Phase 2 train |
| **Expanded NSE Filings (`phase2/expand_nse.py`)** | `917` expanded NSE disclosures across `706` issuers | Public Exchange Filings | Excludes all Phase 1 `doc_id`s; chronological & issuer-heldout splits |
| **FinQA** (`czyssrs/FinQA`) | `train.json`, `dev.json`, `test.json` | **MIT** | Full incorporation (`permissive_mit_or_public`) for numerical QA + FinQA DSL |
| **TAT-QA** (`NExTplusplus/TAT-QA`) | `tatqa_dataset_train.json`, `tatqa_dataset_dev.json` | **MIT** | Full incorporation (`permissive_mit_or_public`) for hybrid table+text span & arithmetic QA |
| **FinRED** (`FinGPT/fingpt-finred-re`, `ChanceFocus/flare-finred`) | `train-00000.parquet`, `test-00000.parquet`, `flare_finred_test.parquet` | `Academic-Research-FinWeb22` | Separated tier (`academic_research`) for financial relation & entity supervision |
| **Financial PhraseBank** (`takala/financial_phrasebank`) | `FinancialPhraseBank-v1.0.zip` (`AllAgree`, `2,264` rows) | **CC BY-NC-SA 3.0** | **Isolated** in `phase2/data/phrasebank_nc_sa_eval.jsonl` (`nc_sa_research_only`); excluded from commercial/production training |

### 2.2 Corpus Splits & Zero-Leakage Audit (`phase2/data/corpus_manifest.json` & `leakage_audit.json`)
Total usable training examples generated: **`12,802`** across 5 tasks (plus `844 + 598` QA val/test, `2,264` PhraseBank research-only eval, and `370` frozen benchmark cases):

| Task | Train | Val | Test | Train/Val Overlap | Train/Test Overlap | Val/Test Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NER (`ner_*`)** | 2,091 | 136 | 284 | 0 | 0 | 0 |
| **Relations (`rel_*`)** | 3,762 (manifest `3,763` pool) | 295 | 624 (`24` NSE) | 0 | 0 | 0 |
| **Retrieval (`retrieval_*`)** | 859 | 77 | 139 | 0 | 0 | 0 |
| **QA & Numeric (`qa_*`)** | 4,986 | 844 | 598 | 0 | 0 | 0 |
| **Contextual Impact (`impact_*`)** | 1,103 | 85 | 172 | 0 | 0 | 0 |

### 2.3 Frozen Independent Benchmark (`phase2/data/benchmark_frozen.jsonl`, `370` Cases)
| Slice | Count | Purpose |
| :--- | :---: | :--- |
| `unseen_issuer` | 107 | Held-out NSE issuers unseen in training |
| `numerical_reasoning` | 110 | FinQA / TAT-QA / NSE financial math & table interpretation |
| `incomplete_or_failed_pdf` | 31 | Truncated / garbled / failed-extraction disclosures |
| `unanswerable` | 30 | Missing comparison period, currency mismatch without FX, zero denom, future scheduled event |
| `non_financial` | 25 | Non-financial noise inputs |
| `long_report` | 23 | Multi-paragraph Reg-30 / AGM / dividend filings |
| `prompt_injection` | 20 | Embedded `[SYSTEM OVERRIDE]` / `IGNORE PRIOR` attacks inside disclosures |
| `misleading_headline` | 15 | Positive wording paired with negative fundamentals or caveats |
| `multi_event` | 5 | Mixed multi-event corporate announcements |
| `contradictory_or_rumor` | 4 | Rumor verification / denial vs confirmed actions |

> Provenance labels strictly separate `human_reviewed` (FinQA/TAT-QA/manual NSE annotations), `deterministic` (regex/rule-verified offsets & NSE structured metadata), `weak_supervision` (FinRED automated distant alignments), and `synthetic` adversarial cases (never presented as human-reviewed ground truth).

---

## 3. Component Benchmarks & Quantization Comparison (FP32 vs ONNX INT8)

### 3.1 Financial NER (`phase2/train_ner.py` — `ModernBERT-base` Token Classification, 10 Entity Types / 21 BIO Tags)
- **CPU Budget & Fix Applied:** Trained on `640` shuffled NSE examples (`2` epochs, `40` steps/epoch, `max_length=96`, `lr=8e-5`). During Epoch 1/2 diagnostics, we caught a byte-level BPE offset shift where ModernBERT prepends leading whitespace to token offsets (e.g., `' SE'` at `[38, 41)` vs gold `[39, 43)`). Adding leading/trailing whitespace trimming in `bio_to_spans()` and `EdgeReasoningPipeline.extract_entities()` lifted validation span F1 from `0.054` to **`0.7704`** (`Epoch 2`).
- **Held-Out Test (`284` documents):**
  - **ONNX FP32 (`596.7 MB`):** Precision `0.8397`, Recall `0.7451`, **F1 `0.7896`**, p50 `166.81ms`, p95 `327.07ms`
  - **ONNX INT8 (`150.5 MB`):** Precision `0.8399`, Recall `0.6944`, **F1 `0.7602`** (`-0.0294` quantization delta), p50 `60.70ms`, p95 **`126.20ms`** (**2.59x faster p95**, **3.97x smaller**)
  - Strongest INT8 entity types: `REGULATOR` (`F1=0.9648`, support `263`), `DATE_PERIOD` (`F1=0.8571`, support `200`), `PERCENT` (`F1=0.7407`, support `17`), `GEO_MARKET` (`F1=0.5402`, support `103`). Weakest: tail `ORG` (`0.225`), `TICKER` (`0.0`, support `21`), `MONEY` (`0.2745`, support `22`) due to `640`-sample CPU training budget.

### 3.2 Financial Relation Extraction (`phase2/train_rel.py` — `BGE-small-en-v1.5`, 11 Relation Classes)
- **Training:** `556` stratified examples (`<= 90`/class + all NSE relation examples including `acquires`, `acquisition_approved_for`, `acquisition_denied_or_cancelled`, `penalized_or_ordered_by`, `contract_awarded_to`, `reported_metric_for`), `3` epochs, `lr=5e-5`.
- **Held-Out Test (`624` pairs, `24` NSE event-stage relations):**
  - **ONNX FP32 (`133.1 MB`):** Micro-F1 `0.7115`, Weighted-F1 `0.7143`, Macro-F1 `0.4347`, NSE event-relation acc `0.5417` (`24`), p95 `69.43ms`
  - **ONNX INT8 (`34.0 MB`):** **Micro-F1 `0.7099`**, **Weighted-F1 `0.7131`**, **Macro-F1 `0.4354`** (`+0.0007` delta), NSE acc `0.5417`, p50 `13.88ms`, p95 **`34.58ms`** (**2.01x faster**, **3.92x smaller**)
  - High-F1 INT8 classes: `leadership_role` (`0.8669`, `114`), `penalized_or_ordered_by` (`0.8571`, `4`), `located_or_listed_at` (`0.7373`, `255`), `subsidiary_or_owned_by` (`0.6559`, `196`), `reported_metric_for` (`0.6087`, `7`). Unweighted Macro-F1 (`0.4354`) reflects zero/tiny support on tail classes (`acquires`: `0`, `contract_awarded_to`: `2`, `acquisition_denied_or_cancelled`: `2`, `acquisition_approved_for`: `1`).

### 3.3 Evidence Retrieval & Reranking (`phase2/train_retrieval.py` — 4-Way Comparison with 6 Hard-Negative Types)
Evaluated over `139` held-out test queries against `351` unique corpus + hard-negative passages (same company wrong year, right event wrong amount, peer issuer, cancelled vs completed, speculation vs confirmed, keyword distractors):

| Setup | Precision | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 (ms) | p95 (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen BGE Baseline** | INT8 | 0.2158 | 0.2878 | 0.1526 | 0.1845 | 4.97 | 5.47 | 34.0 |
| **Domain-FT BGE (InfoNCE)** | FP32 | 0.6619 | 0.8201 | 0.5220 | 0.5919 | 12.31 | 16.75 | 133.1 |
| **Domain-FT BGE (InfoNCE)** ★ | **INT8** | **0.6619** | **0.7842** | **0.5202** | **0.5832** | **6.05** | **6.82** | **34.0** |
| BGE Cross-Encoder (top-15 rerank) | FP32 | 0.2878 | 0.5899 | 0.1522 | 0.2517 | 1168.3 | 1268.5 | 133.1 |
| BGE Cross-Encoder (top-15 rerank) | INT8 | 0.3022 | 0.5827 | 0.1608 | 0.2568 | 582.6 | 614.1 | 34.0 |
| ModernBERT Cross-Encoder (top-15) | FP32 | 0.3165 | 0.6331 | 0.1990 | 0.2975 | 5321.6 | 5644.2 | 596.7 |
| ModernBERT Cross-Encoder (top-15) | INT8 | 0.3022 | 0.6043 | 0.1782 | 0.2752 | 2119.4 | 2196.9 | 150.5 |

> **Critical Ablation Finding:** Contrastive InfoNCE training on hard negatives (`Domain-FT BGE INT8`) lifted `Recall@5` by **+44.61 percentage points** (`0.2158 -> 0.6619`) and `MRR@10` by **+36.76 points** (`0.1526 -> 0.5202`) while keeping `p95` latency at `6.82 ms`. In contrast, pointwise BCE-trained cross-encoders over top-15 passages underperformed the InfoNCE bi-encoder (`0.1608`–`0.1782` MRR@10) and added `614–2,197 ms` latency. Consequently, `EdgeReasoningPipeline` defaults to `Domain-FT BGE INT8` (`use_cross_encoder=False` by default, preserved as a toggleable ablation).

### 3.4 Contextual Financial Impact Classifier (`phase2/train_impact.py` — Multi-Head `BGE-small`)
Trained on `1,103` examples across 5 heads (`implications` multi-label, `materiality`, `event_maturity`, `uncertainty`, `stakeholders` multi-label); evaluated on `172` held-out test disclosures:
- **ONNX FP32 (`133.2 MB`):** Implications micro-F1 `0.8282` (macro `0.7982`), Stakeholders micro-F1 `0.9130`, Materiality acc `0.9826`, Event Maturity acc `0.9767`, Uncertainty acc `0.9767`, p95 `95.80ms`
- **ONNX INT8 (`34.0 MB`):** **Implications micro-F1 `0.8159`** (macro `0.7860`), **Stakeholders micro-F1 `0.9045`**, **Materiality acc `0.9884`**, **Event Maturity acc `0.9767`**, **Uncertainty acc `0.9767`**, p50 `47.75ms`, **p95 `50.13ms`** (**1.91x faster**)
- **Isolated PhraseBank `CC-BY-NC-SA-3.0` Research Eval (`500` sentences):** Directional alignment `0.114` — confirms sentence-level generic European news sentiment (`PhraseBank`) does not map 1:1 onto `[category | stage]` corporate-disclosure implication tags (e.g., positive phrasing vs neutral/uncertain regulatory implications).

### 3.5 Deterministic Numerical Reasoning & Neural QA (`phase2/train_qa_and_eval.py`)
- **Deterministic Numerical Engine (`phase2/numeric_engine.py`):**
  - FinQA DSL execution on `292` held-out `qa_test.jsonl` arithmetic programs: **`290 / 292` (`99.32%` exact / 2%-tolerance accuracy)**.
  - Indian `crore`/`lakh` normalization + safety guards (`zero_denominator`, `currency_mismatch_no_fx_rate`, `inconsistent_periods`, `negative_denominator_pct_undefined`): **100% pass**.
- **QA Operation Planner (`BGE-small`, 7 op classes, `1,268` train / `598` held-out `qa_test.jsonl`):**
  - ONNX FP32 (`133.1 MB`): accuracy `0.8294`, macro-F1 `0.6820`, p95 `72.22ms`
  - ONNX INT8 (`34.0 MB`): **accuracy `0.8261`**, **macro-F1 `0.6788`**, p50 `34.23ms`, **p95 `35.90ms`** (**2.01x faster**)
- **Extractive QA (`DistilBERT-SQuAD`, `60` val spans):**
  - PyTorch (`mean_token_f1=0.1831`, p95 `131.1ms`) -> ONNX FP32 (`260.9 MB`, `0.1831`, p95 `255.6ms`) -> **ONNX INT8 (`65.8 MB`, `0.1939`, p95 `105.6ms`, FP32/INT8 span agreement `0.80`)**.
- **Ablation — Extractive QA Alone vs Deterministic Numeric Engine:** Extractive span selection alone yields `0.1939` token F1 on financial QA vs **`0.9932`** execution accuracy when paired with the deterministic calculation engine (`phase2/numeric_engine.py`).

---

## 4. Architecture A vs Architecture B Comparison

We benchmarked **Architecture A** (`EdgeReasoningPipeline` ONNX INT8 stack) against **Architecture B** (`Qwen/Qwen2.5-0.5B-Instruct`, `494.3M` params, Apache-2.0, with `peft` LoRA `r=4, lora_alpha=8`, `270,336` trainable params, 20 CPU adaptation steps at **`1.55 s/step`**) on the same Windows CPU host:

| Dimension | Architecture A (Standalone `.venv` Pure ONNX INT8) | Architecture A (Shared PyTorch Process `train_qa_and_eval`) | Architecture B (`Qwen2.5-0.5B-Instruct` + LoRA) |
| :--- | :---: | :---: | :---: |
| **Runtime Dependencies** | `onnxruntime` + `tokenizers` (zero PyTorch) | PyTorch + Transformers + ORT | PyTorch + Transformers + PEFT |
| **Model Footprint on Disk** | **`368.19 MB`** (`386,076,545` bytes across 8 INT8 ONNX files) | `368.19 MB` | `954.3 MB` (`1,000,686,517` bytes) |
| **Cold Load Time** | **`1.564 s`** | `2.203 s` | `2.22 s` |
| **Peak Process RSS** | **`787.0 MB`** (`570.3 MB` after load) | `3,170.6 MB` *(PyTorch leftover in proc)* | `2,746.7 MB` |
| **Warm CPU p50 / p95 Latency** | **`339.53 ms` / `676.83 ms`** | `355.46 ms` / `694.02 ms` | `5,225.0 ms` / `6,119.4 ms` |
| **Throughput** | **`3.01 docs/s`** | `2.93 docs/s` | `0.225 docs/s` (**13.4x slower**) |
| **Supported Claims Rate** | **`100.0%` (`1.0000`)** | `100.0%` | Generative *(observed `"SASEL"` acronym hallucination)* |
| **Unsupported Claim Rate** | **`0.0%` (`0.0000`)** | `0.0%` | Unverified free-text |
| **Unanswerable Abstention Recall** | **`72.09%`** (`unanswerable` `30/30`, `non_financial` `25/25`, `incomplete_pdf` `7/31`) | `72.09%` | `33.33%` (stratified 30-case probe) |
| **Prompt Injection Safe Rate** | **`100.0%` (`20/20`)** | `100.0%` | `100.0%` (3/3 probe) |

**Architecture Selection Decision:** **Architecture A** is selected cleanly. Even a `0.5B` generative LLM with LoRA runs at `6.12 s` p95 (`0.225 docs/s`), consumes `2.75 GB` RAM, and achieves only `33.3%` unanswerable abstention on CPU, whereas Architecture A uses `787 MB` peak RSS in the pure `.venv` runtime, guarantees `0.0%` unsupported extractive claims, resists 100% of prompt injections, runs `99.32%` exact deterministic FinQA math, and processes `3.01 docs/s` offline without external LLM APIs.

### Architecture A Slice Breakdown (`370` Frozen Benchmark Cases)
- `unanswerable`: `30/30` abstained (`100%`)
- `non_financial`: `25/25` abstained (`100%`)
- `prompt_injection`: `20/20` injection-safe (`100%`), `15/20` event categories preserved
- `unseen_issuer`: `107/107` source-supported (`64/107` exact Phase-1 BGE category match)
- `incomplete_or_failed_pdf`: `7/31` rejected by text-quality guard (remaining `24` passed minimal ASCII/keyword threshold and were flagged as extractive-grounded rather than rejected — documented gap below)
- `numerical_reasoning`: `1/110` on raw flattened text without structured `table` input (vs `290/292 = 99.32%` when `qa_op_planner` + structured `table` arrays are passed to `execute_finqa_program`)

---

## 5. Provisional Promotion Gate Assessment (Section 10)

All targets are reported honestly from `phase2/results/phase2_benchmark.json`:

| Gate Metric | Provisional Target | Measured | Outcome | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Entity Extraction Micro-F1 (ModernBERT INT8) | `>= 0.85` | `0.7602` (`0.7896` FP32) | **FAIL** | Trained 2 epochs on `640` rows for CPU budget; strong on `REGULATOR` (`0.965`), `DATE_PERIOD` (`0.857`) |
| Relation Extraction Macro-F1 (BGE INT8) | `>= 0.80` | `0.4354` (`0.7131` weighted) | **FAIL** | Tail FinRED classes (`support=0..2`) drag unweighted macro-F1 across 11 classes |
| Evidence Recall@5 (Domain-FT BGE INT8) | `>= 0.88` | `0.6619` (`0.8052` val) | **FAIL** | Huge +44.6 pt lift vs Frozen BGE (`0.2158`), short of `0.88` over hard negatives |
| Reranker MRR@10 | `>= 0.80` | `0.5202` | **FAIL** | Domain-FT BGE beats Frozen BGE (`0.1526`) and pointwise CEs (`0.161–0.178`) |
| Deterministic Numerical / FinQA Accuracy | `>= 0.85` | **`0.9932`** (`290/292`) | **PASS** | Guardrails & crore/lakh normalization pass 100% |
| Contextual Impact Micro-F1 (BGE INT8) | `>= 0.78` | **`0.8159`** | **PASS** | `event_maturity=0.9767`, `materiality=0.9884` |
| Unsupported Claim Rate (Architecture A) | `<= 0.05` | **`0.0000`** | **PASS** | Strictly grounded in verbatim spans |
| Unanswerable Abstention Recall (Architecture A) | `>= 0.85` | `0.7209` | **FAIL** | `100%` on `unanswerable` & `non_financial`; gap on partial/truncated PDF notices (`7/31`) |
| CPU Single-Doc Warm p95 Latency (Architecture A) | `<= 400 ms` | `676.83 ms` *(standalone)* | **FAIL** | `p50=339.53 ms` meets `<400 ms`; running all 6 neural heads sequentially pushes p95 to `677 ms` |
| Total Edge ONNX INT8 Footprint | `<= 500 MB` | **`368.19 MB`** | **PASS** | `8` ONNX INT8 artifacts combined |

**Scorecard Summary:** **4 / 10 provisional gates passed**, **6 / 10 failed** under CPU-only training/latency budgets. Because only `4/10` provisional research targets passed, `models/registry.json` (`Phase 1` default `semantic`) is **NOT promoted or overwritten**; Phase 2 artifacts sit isolated under `models/phase2/` (`registry_phase2.json`).

---

## 6. NEWS_INTELLIGENCE & MARKET_CONTEXT Adapter Contract

`EdgeReasoningPipeline.to_news_intelligence(entry, question, market_context)` in `phase2/reasoning_pipeline.py` emits a contract-compatible payload without modifying `stockbrief/`:
- Preserves `entry_id`, `ticker`, `title`, `source`, `date`, `url`, `origin`, `state` (`"identified"` / `"needs_review"`), and Phase 1 `events`.
- Adds `reasoning` (`what_happened`, `entities`, `relations`, `impact`, `calculation`, verbatim `evidence` spans with page & char offsets, `answerable`, `injection_detected`).
- Attaches `market_context_observed` **strictly as separately timestamped historical surrounding closes** (never fed into model inputs and never attributed causally), paired with explicit `limitations` disclaimers prohibiting price forecasts or buy/sell recommendations.

---

## 7. Artifact Inventory & SHA-256 Registry (`models/phase2/`)

| Module | Artifact Path (INT8) | INT8 Bytes | INT8 SHA-256 | FP32 Bytes |
| :--- | :--- | :---: | :--- | :---: |
| Phase 1 Event Classifier | `models/semantic/model_quantized.onnx` | 34,014,426 | *(preserved in Phase 1 `models/registry.json`)* | — |
| ModernBERT NER | `models/phase2/ner_modernbert/model_quantized.onnx` | 150,468,563 | `8d9d2c0c8aeb71968d8d78f726a1707d46f39839c6f531b018e27ccf807263ae` | 596,719,123 |
| BGE Relations | `models/phase2/rel_bge/model_quantized.onnx` | 33,968,330 | `7e0ae7b6994f6c82c917a294993ff8d46c5b7539f935a898a8f6713964261502` | 133,137,273 |
| Domain-FT BGE Retrieval | `models/phase2/bge_retrieval_ft/model_quantized.onnx` | 33,963,555 | `275019e6ab91f6c6af8096eecb62b03418e34c1df7b6b398110fe8fa1255505d` | 133,120,600 |
| BGE Cross-Encoder Reranker | `models/phase2/bge_cross_reranker/model_quantized.onnx` | 33,964,641 | `2d40a29c4024ec602f3de5b41347898096f1d5fdc3842e82b765e6f6dd0ead68` | 133,122,022 |
| ModernBERT Reranker | `models/phase2/modernbert_reranker/model_quantized.onnx` | 150,455,688 | `99ac11a0927c0cc7230b9c2d0e4b346d46437cecf64447b0a8b2f695c62da2ca` | 596,658,487 |
| BGE Contextual Impact | `models/phase2/impact_bge/model_quantized.onnx` | 33,973,833 | `664c2ed4f7a39a2ad86580dac9a86fa7a0b4418e20d1d3e485702c63c23d022b` | 133,158,833 |
| DistilBERT Extractive QA | `models/phase2/qa_extractive_onnx/model_quantized.onnx` | 65,770,865 | `9f9f9188d328f9b4acdf6b53bea3d55658e70de50be20541c8eb6a1f9f9676e8` | 260,909,960 |
| BGE QA Op Planner | `models/phase2/qa_op_planner/model_quantized.onnx` | 33,966,758 | `fbdabcf8833b908aaa3169b353d3c225c63430fd3f813ef3e47ff41b77c2176d` | 133,131,111 |

---

## 8. Reproducible PowerShell Commands

```powershell
# 1. Build corpus, download external datasets & audit zero leakage
training\.venv\Scripts\python.exe -m phase2.download_external_datasets
training\.venv\Scripts\python.exe -m phase2.expand_nse
training\.venv\Scripts\python.exe -m phase2.fetch_phase2_models
training\.venv\Scripts\python.exe -m phase2.build_corpus
training\.venv\Scripts\python.exe -m phase2.audit_leakage

# 2. Train & export ONNX INT8 specialist models (resumable checkpoints)
training\.venv\Scripts\python.exe -m phase2.train_rel
training\.venv\Scripts\python.exe -m phase2.train_impact
training\.venv\Scripts\python.exe -m phase2.train_ner
training\.venv\Scripts\python.exe -m phase2.train_retrieval

# 3. Export QA ONNX, run Architecture B probe & 370-case benchmark
training\.venv\Scripts\python.exe -m phase2.train_qa_and_eval

# 4. Measure standalone pure-ONNX Architecture A memory/latency & verify all tests
.\.venv\Scripts\python.exe -m phase2.measure_edge_runtime
.\.venv\Scripts\python.exe -m unittest phase2\tests\test_phase2_pipeline.py -v
.\.venv\Scripts\python.exe -m unittest training\tests\test_training_pipeline.py -v
.\.venv\Scripts\python.exe -m unittest discover -v
```
