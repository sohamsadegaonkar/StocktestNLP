# Phase 2 Model Card — StocktestNLP Edge Financial Reasoning System (`stockbrief-phase2-edge-v1`)

## Model Overview
- **System:** `EdgeReasoningPipeline` (`phase2/reasoning_pipeline.py`)
- **Architecture:** Modular multi-task CPU pipeline combining ONNX INT8 specialist encoders (`BGE-small-en-v1.5`, `ModernBERT-base`, `DistilBERT-SQuAD`) with a deterministic financial arithmetic engine (`phase2/numeric_engine.py`).
- **Base Pretrained Licenses:** `BAAI/bge-small-en-v1.5` (MIT), `answerdotai/ModernBERT-base` (Apache-2.0), `distilbert/distilbert-base-cased-distilled-squad` (Apache-2.0). Evaluated alternative: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0).
- **Bundle Footprint:** `368.19 MB` total ONNX INT8 artifacts; `787.0 MB` peak RAM in pure `.venv` runtime; `339.53 ms` warm p50 / `676.83 ms` warm p95 CPU latency.

## Intended Use & Scope
- Analyzing official Indian listed-company disclosures (NSE/SEBI filings), earnings releases, corporate actions, and numeric financial QA when source tables or passages are supplied.
- Designed strictly as an **evidence-grounded analytical assistant** for `NEWS_INTELLIGENCE` and `MARKET_CONTEXT` integration.

## Out-of-Scope & Prohibited Uses
- **Not investment advice:** Never generates price targets, return probabilities, or buy/sell recommendations.
- **Not causal price attribution:** Attached `MARKET_CONTEXT` market closes are preserved strictly as historical surrounding context (`market_context_observed`), never as input features or causal claims.
- **Commercial isolation:** Financial PhraseBank (`CC-BY-NC-SA-3.0`) is strictly excluded from model training and isolated for research-only validation (`phase2/data/phrasebank_nc_sa_eval.jsonl`).

## Known Limitations & Biases
1. **CPU Training Budgets:** Because training ran on an 8-thread CPU (`0 GB` VRAM), ModernBERT NER used a `640`-sample / 2-epoch budget (`0.7602` INT8 F1) and `ModernBERT-CrossEncoder` used 1 epoch (`360` queries). Full-corpus GPU training is needed to clear the `0.85` Entity F1, `0.80` Relation Macro-F1, and `0.88` Recall@5 provisional gates.
2. **Table-Free Numeric Benchmark Gap:** When structured `table` arrays are provided (`qa_test.jsonl`), FinQA DSL execution achieves `99.32%` (`290/292`). On raw flattened text without explicit table parsing (`benchmark_frozen.jsonl` `numerical_reasoning` slice), complex multi-step FinQA ratios are not reconstructed from prose alone (`1/110`).
3. **Partial PDF Abstention:** While `unanswerable` (`30/30`) and `non_financial` (`25/25`) abstain at `100%`, truncated disclosures that still retain valid financial keywords passed the quality gate (`7/31` abstained on `incomplete_or_failed_pdf`).
4. **Tail Latency:** Running all 6 ONNX INT8 models sequentially yields `339.53 ms` p50 and `676.83 ms` p95 on CPU (above the `400 ms` provisional p95 target; selective/lazy head execution is recommended in production).
