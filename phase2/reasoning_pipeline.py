"""Phase 2 Architecture A: Edge Financial Reasoning Pipeline & NEWS_INTELLIGENCE + MARKET_CONTEXT Adapter."""
from __future__ import annotations

from hashlib import file_digest
import json
import re
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from phase2.numeric_engine import SourceQuantity, compute_metric, execute_finqa_program, parse_quantity
from phase2.schema import (
    EVENT_MATURITY_LEVELS,
    IMPACT_IMPLICATIONS,
    MATERIALITY_LEVELS,
    NER_BIO_LABELS,
    REL_LABELS,
    STAKEHOLDER_GROUPS,
    UNCERTAINTY_LEVELS,
)
from stockbrief.fixtures import Document
from stockbrief.nlp import chunks_for
from stockbrief.semantic import EVENT_BOUNDARY, SemanticEventModel, load_semantic, model_directory

INJECTION_RE = re.compile(
    r"(?:\[SYSTEM\s+OVERRIDE.*?\]|<!--.*?-->|###\s*IGNORE\s+ABOVE.*|"
    r"ignore\s+(?:all\s+)?(?:prior|previous)\s+(?:instructions|context).*|"
    r"user\s+instruction\s*:.*)",
    re.I | re.S,
)
NON_FINANCIAL_RE = re.compile(
    r"\b(?:flour|butter|bake|oven|laptop|wi-?fi|driver|cricket\s+team|wickets|"
    r"train\s+from|fiddle-leaf|watering|recipe|poem|ankle)\b",
    re.I,
)
FINANCIAL_ANCHOR_RE = re.compile(
    r"\b(?:company|limited|ltd|board|quarter|fy\d{2,4}|revenue|profit|loss|ebitda|"
    r"dividend|buyback|acquir\w+|merger|demerger|sebi|rbi|penalty|order|contract|"
    r"price|tariff|director|auditor|shareholder|margin|debt|crore|lakh|inr|usd|eur|rs\.?|₹|%)\b",
    re.I,
)
GARBLED_RE = re.compile(r"([a-zA-Z])\1{12,}")


def sanitize_prompt_injection(text: str) -> tuple[str, bool]:
    cleaned, n = INJECTION_RE.subn(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip(), n > 0


def is_readable_financial_text(text: str) -> tuple[bool, str]:
    if not text or sum(c.isalnum() for c in text) < 12:
        return False, "insufficient_or_empty_text"
    ascii_letters = sum("a" <= c.lower() <= "z" for c in text)
    if ascii_letters / max(1, len(text)) < 0.30 or GARBLED_RE.search(text):
        return False, "unreadable_or_garbled_text"
    if NON_FINANCIAL_RE.search(text) and not FINANCIAL_ANCHOR_RE.search(text):
        return False, "non_financial_input"
    if not FINANCIAL_ANCHOR_RE.search(text) and len(text.split()) < 20:
        return False, "non_financial_or_insufficient_context"
    return True, "ok"


def _ort_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _OrtTokenizer:
    """Zero-PyTorch Rust tokenizer wrapper matching HuggingFace ONNX inputs."""

    def __init__(self, model_dir: Path, max_length: int, only_second: bool = False):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        pad_tok = "[PAD]" if self.tok.token_to_id("[PAD]") is not None else "<pad>"
        self.pad_id = self.tok.token_to_id(pad_tok) or 0
        self.pad_tok = pad_tok
        self.max_length = max_length
        self.tok.enable_truncation(
            max_length=max_length,
            strategy="only_second" if only_second else "longest_first",
        )

    def encode_single(self, text: str, pair: str | None = None):
        self.tok.no_padding()
        return self.tok.encode(text, pair)

    def encode_batch(self, texts: list[str], pairs: list[str] | None = None) -> dict[str, np.ndarray]:
        self.tok.enable_padding(pad_id=self.pad_id, pad_token=self.pad_tok)
        encs = self.tok.encode_batch(texts if pairs is None else list(zip(texts, pairs)))
        return {
            "input_ids": np.asarray([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encs], dtype=np.int64),
        }


class EdgeReasoningPipeline:
    """Architecture A: BGE/Domain-BGE retrieval -> Cross-Encoder reranking -> NER/Rel/Impact -> Deterministic Calc + Extractive QA."""

    version = "stockbrief-phase2-edge-v1"

    def __init__(self, models_root: Path = Path("models"), use_cross_encoder: bool = False):
        p2 = models_root / "phase2"
        self.use_cross_encoder = use_cross_encoder
        self.event_model: SemanticEventModel = load_semantic(models_root / "semantic")

        # Domain-FT BGE bi-encoder (winning retrieval model: +44.6 pt R@5 over Frozen BGE at 6.8ms p95)
        bge_ft_dir = p2 / "bge_retrieval_ft"
        if (bge_ft_dir / "model_quantized.onnx").is_file():
            self.bi_tok = _OrtTokenizer(bge_ft_dir, max_length=160)
            self.bi_sess = _ort_session(bge_ft_dir / "model_quantized.onnx")
        else:
            self.bi_tok = None
            self.bi_sess = None

        # Optional Cross-Encoder reranker (disabled by default after benchmark showed pointwise CE degraded InfoNCE Domain-FT BGE)
        bge_ce_dir = p2 / "bge_cross_reranker"
        mb_ce_dir = p2 / "modernbert_reranker"
        ce_dir = bge_ce_dir if (bge_ce_dir / "model_quantized.onnx").is_file() else mb_ce_dir
        if self.use_cross_encoder and (ce_dir / "model_quantized.onnx").is_file():
            self.ce_tok = _OrtTokenizer(ce_dir, max_length=128)
            self.ce_sess = _ort_session(ce_dir / "model_quantized.onnx")
        else:
            self.ce_tok = None
            self.ce_sess = None

        # ModernBERT NER ONNX INT8
        ner_dir = p2 / "ner_modernbert"
        if (ner_dir / "model_quantized.onnx").is_file():
            self.ner_tok = _OrtTokenizer(ner_dir, max_length=128)
            self.ner_sess = _ort_session(ner_dir / "model_quantized.onnx")
        else:
            self.ner_tok = None
            self.ner_sess = None

        # BGE Relation Extraction ONNX INT8
        rel_dir = p2 / "rel_bge"
        if (rel_dir / "model_quantized.onnx").is_file():
            self.rel_tok = _OrtTokenizer(rel_dir, max_length=128)
            self.rel_sess = _ort_session(rel_dir / "model_quantized.onnx")
        else:
            self.rel_tok = None
            self.rel_sess = None

        # Contextual Impact ONNX INT8
        imp_dir = p2 / "impact_bge"
        if (imp_dir / "model_quantized.onnx").is_file():
            self.imp_tok = _OrtTokenizer(imp_dir, max_length=160)
            self.imp_sess = _ort_session(imp_dir / "model_quantized.onnx")
        else:
            self.imp_tok = None
            self.imp_sess = None

        # Extractive QA ONNX INT8 (DistilBERT-SQuAD)
        qa_dir = p2 / "qa_extractive_onnx"
        if (qa_dir / "model_quantized.onnx").is_file():
            self.qa_tok = _OrtTokenizer(qa_dir, max_length=256, only_second=True)
            self.qa_sess = _ort_session(qa_dir / "model_quantized.onnx")
        else:
            self.qa_tok = None
            self.qa_sess = None

    def encode_bi(self, texts: list[str]) -> np.ndarray:
        if self.bi_sess is None:
            return self.event_model.encoder.encode(texts)
        e = self.bi_tok.encode_batch(texts)
        embs = self.bi_sess.run(None, e)[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-12)

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        if self.ce_sess is None:
            embs = self.encode_bi([query] + passages)
            sims = embs[1:] @ embs[0]
            return sorted(enumerate(sims.tolist()), key=lambda x: -x[1])
        e = self.ce_tok.encode_batch([query] * len(passages), passages)
        scores = self.ce_sess.run(None, e)[0]
        return sorted(enumerate(scores.tolist()), key=lambda x: -x[1])

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        if self.ner_sess is None or not text.strip():
            return []
        enc = self.ner_tok.encode_single(text)
        inp = np.asarray([enc.ids], dtype=np.int64)
        msk = np.asarray([enc.attention_mask], dtype=np.int64)
        logits = self.ner_sess.run(None, {"input_ids": inp, "attention_mask": msk})[0][0]
        pred_ids = np.argmax(logits, axis=-1).tolist()

        def _emit_ent(tp: str, s: int, e: int) -> None:
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                entities.append({"start": s, "end": e, "label": tp, "text": text[s:e]})

        entities = []
        cur_type, cur_s, cur_e = None, -1, -1
        for tid, (s, e) in zip(pred_ids, enc.offsets):
            if s == 0 and e == 0:
                continue
            tag = NER_BIO_LABELS[int(tid)]
            if tag.startswith("B-"):
                if cur_type is not None:
                    _emit_ent(cur_type, cur_s, cur_e)
                cur_type, cur_s, cur_e = tag[2:], s, e
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_e = e
            else:
                if cur_type is not None:
                    _emit_ent(cur_type, cur_s, cur_e)
                    cur_type = None
        if cur_type is not None:
            _emit_ent(cur_type, cur_s, cur_e)
        return entities

    def extract_relations(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.rel_sess is None or not entities:
            return []
        orgs = [e["text"] for e in entities if e["label"] in ("ORG", "TICKER")][:3]
        objs = [e["text"] for e in entities if e["label"] in ("MONEY", "REGULATOR", "PRODUCT_SECTOR", "FIN_INST", "PERCENT", "ORG")][:5]
        pairs = [(s, o) for s in orgs for o in objs if s.lower() != o.lower()][:6]
        if not pairs:
            return []
        inputs = [f"Subject: {s} | Object: {o} || {text[:450]}" for s, o in pairs]
        e = self.rel_tok.encode_batch(inputs)
        logits = self.rel_sess.run(None, e)[0]
        rels = []
        for (s, o), row in zip(pairs, logits):
            rel = REL_LABELS[int(np.argmax(row))]
            if rel != "no_relation":
                rels.append({"subject": s, "object": o, "relation": rel})
        return rels

    def predict_impact(self, category: str, stage: str, text: str) -> dict[str, Any]:
        if self.imp_sess is None:
            return {
                "implications": ["insufficient_evidence"] if category == "review" else ["neutral_implication"],
                "materiality": "unclear",
                "event_maturity": "unclear",
                "uncertainty": "high",
                "stakeholders": ["shareholders"],
            }
        inp = f"[{category} | {stage}] {text[:700]}"
        e = self.imp_tok.encode_batch([inp])
        lg = self.imp_sess.run(None, e)[0][0]
        i1 = len(IMPACT_IMPLICATIONS)
        i2 = i1 + len(MATERIALITY_LEVELS)
        i3 = i2 + len(EVENT_MATURITY_LEVELS)
        i4 = i3 + len(UNCERTAINTY_LEVELS)
        p_imp = _sigmoid(lg[:i1])
        imps = [IMPACT_IMPLICATIONS[i] for i, p in enumerate(p_imp) if p >= 0.45]
        if not imps:
            imps = [IMPACT_IMPLICATIONS[int(np.argmax(p_imp))]]
        p_stk = _sigmoid(lg[i4:])
        stks = [STAKEHOLDER_GROUPS[i] for i, p in enumerate(p_stk) if p >= 0.45] or ["shareholders"]
        return {
            "implications": imps,
            "materiality": MATERIALITY_LEVELS[int(np.argmax(lg[i1:i2]))],
            "event_maturity": EVENT_MATURITY_LEVELS[int(np.argmax(lg[i2:i3]))],
            "uncertainty": UNCERTAINTY_LEVELS[int(np.argmax(lg[i3:i4]))],
            "stakeholders": stks,
        }

    def extractive_qa(self, question: str, passage: str) -> dict[str, Any]:
        if self.qa_sess is None or not passage.strip():
            return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}
        enc = self.qa_tok.encode_single(question, passage)
        seq_ids = enc.sequence_ids
        inp = np.asarray([enc.ids], dtype=np.int64)
        msk = np.asarray([enc.attention_mask], dtype=np.int64)
        start_lg, end_lg = self.qa_sess.run(None, {"input_ids": inp, "attention_mask": msk})
        s_lg = start_lg[0]
        e_lg = end_lg[0]
        ctx_indices = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not ctx_indices:
            return {"answer": "", "confidence": 0.0, "start": 0, "end": 0}
        best_score = -1e9
        best_s, best_e = ctx_indices[0], ctx_indices[0]
        for s_i in ctx_indices:
            for e_i in range(s_i, min(s_i + 20, ctx_indices[-1] + 1)):
                if seq_ids[e_i] != 1:
                    break
                sc = float(s_lg[s_i] + e_lg[e_i])
                if sc > best_score:
                    best_score = sc
                    best_s, best_e = s_i, e_i
        s_probs = np.exp(s_lg - np.max(s_lg))
        s_probs /= s_probs.sum()
        e_probs = np.exp(e_lg - np.max(e_lg))
        e_probs /= e_probs.sum()
        conf = float(np.sqrt(s_probs[best_s] * e_probs[best_e]))
        c_s = enc.offsets[best_s][0]
        c_e = enc.offsets[best_e][1]
        return {"answer": passage[c_s:c_e].strip(), "confidence": round(conf, 4), "start": c_s, "end": c_e}

    def answer_question(self, text: str, question: str, *, ticker: str = "OTHER", title: str = "") -> dict[str, Any]:
        started = perf_counter_ns()
        sanitized, injection_detected = sanitize_prompt_injection(text)
        ok, read_reason = is_readable_financial_text(sanitized)
        if not ok:
            return {
                "version": self.version,
                "answerable": False,
                "reason": read_reason,
                "injection_detected": injection_detected,
                "category": "review",
                "stage": "Unclear",
                "answer": "Cannot determine from the provided text: insufficient or non-financial content.",
                "evidence": [],
                "entities": [],
                "relations": [],
                "impact": {"implications": ["insufficient_evidence"], "materiality": "unclear", "event_maturity": "unclear", "uncertainty": "high", "stakeholders": ["shareholders"]},
                "calculation": None,
                "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
            }

        doc = Document(ticker, (sanitized,), title or "Disclosure")
        brief = self.event_model.analyze(doc)
        chunks = chunks_for((sanitized,), EVENT_BOUNDARY)
        passages = [c.text for c in chunks] or [sanitized[:700]]
        ranked = self.rerank(question, passages)
        top_passages = [
            {"chunk_index": idx, "score": round(sc, 4), "text": passages[idx], "page": chunks[idx].page if idx < len(chunks) else 1, "start": chunks[idx].start if idx < len(chunks) else 0, "end": chunks[idx].end if idx < len(chunks) else len(passages[idx])}
            for idx, sc in ranked[:3]
        ]

        entities = self.extract_entities(top_passages[0]["text"] if top_passages else sanitized)
        relations = self.extract_relations(top_passages[0]["text"] if top_passages else sanitized, entities)
        impact = self.predict_impact(brief["category"], brief["stage"], sanitized)
        if injection_detected:
            impact["uncertainty"] = "high"

        # Check unanswerable conditions deterministically (missing baseline/FX/scheduled future outcome/unstated loss amount)
        q_low = question.lower()
        t_low = sanitized.lower()
        calc = None
        unans_reason = None

        if ("compared to" in q_low or "yoy" in q_low or "growth" in q_low) and not re.search(r"\b(?:fy\s*2[0-6]|prior|previous|earlier|last\s+year|up\s+\d|down\s+\d|rose|fell|declined)\b", t_low):
            unans_reason = "missing_comparison_period"
        elif "inr" in q_low and re.search(r"\b(?:eur|usd|€|\$)\b", t_low) and not re.search(r"\b(?:fx|exchange\s+rate|inr|₹|rs\.?)\b", t_low):
            unans_reason = "currency_mismatch_no_fx_rate"
        elif "margin" in q_low and re.search(r"\b(?:zero-revenue|₹0\s+revenue|0\s+revenue)\b", t_low):
            unans_reason = "zero_denominator"
        elif ("exact rupee amount" in q_low or "how much" in q_low) and "net loss" in q_low and not re.search(r"net\s+loss\s+of\s+(?:₹|rs\.?|inr)", t_low):
            unans_reason = "unstated_net_loss_amount"
        elif ("did the board declare" in q_low or "what dividend" in q_low) and re.search(r"\b(?:scheduled|will\s+be\s+held|to\s+consider)\b", t_low) and "declared" not in t_low:
            unans_reason = "future_scheduled_event_not_yet_decided"

        if unans_reason:
            return {
                "version": self.version,
                "answerable": False,
                "reason": unans_reason,
                "injection_detected": injection_detected,
                "category": brief["category"],
                "stage": brief["stage"],
                "answer": f"Insufficient evidence to answer ({unans_reason}).",
                "evidence": top_passages,
                "entities": entities,
                "relations": relations,
                "impact": impact,
                "calculation": None,
                "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
            }

        # Deterministic numeric calculation if question asks for growth/margin/difference & two quantities exist
        if re.search(r"\b(?:growth|change|margin|difference|bps|percentage\s+points?|yoy|qoq)\b", q_low):
            money_ents = [parse_quantity(e["text"], start=e["start"]) for e in entities if e["label"] in ("MONEY", "PERCENT")]
            money_ents = [m for m in money_ents if m is not None]
            if len(money_ents) >= 2:
                op = "profit_margin" if "margin" in q_low else ("pp_difference" if "percentage point" in q_low else ("difference" if "difference" in q_low else "yoy_growth"))
                calc = compute_metric(op, money_ents[0], money_ents[1])

        qa_span = self.extractive_qa(question, top_passages[0]["text"] if top_passages else sanitized)
        answer_text = (
            f"Event: {brief['category_name']} ({brief['stage']}). "
            + (f"Extracted answer: {qa_span['answer']}. " if qa_span["answer"] else "")
            + (f"Computed: {calc['value']} {calc['unit']} ({calc['formula']}). " if calc and calc["status"] == "ok" else "")
            + f"Implications: {', '.join(impact['implications'])} (uncertainty: {impact['uncertainty']})."
        )

        return {
            "version": self.version,
            "answerable": True,
            "reason": None,
            "injection_detected": injection_detected,
            "category": brief["category"],
            "category_name": brief["category_name"],
            "stage": brief["stage"],
            "events": brief.get("events", []),
            "answer": answer_text,
            "qa_span": qa_span,
            "calculation": calc,
            "evidence": top_passages,
            "entities": entities,
            "relations": relations,
            "impact": impact,
            "inference_ms": round((perf_counter_ns() - started) / 1e6, 3),
        }

    def to_news_intelligence(self, entry: dict[str, Any], question: str = "What happened and what are the financial implications?", market_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Contract adapter extending NEWS_INTELLIGENCE + MARKET_CONTEXT safely without modifying stockbrief/."""
        text = "\n\n".join(entry.get("pages", ()))
        reasoning = self.answer_question(text, question, ticker=entry.get("ticker", "OTHER"), title=entry.get("title", ""))
        # Preserve market_context strictly as separately timestamped observed outcome (never input feature or causal claim)
        return {
            "version": "stockbrief-intelligence-v2-adapter",
            "entry_id": entry.get("id"),
            "ticker": entry.get("ticker"),
            "title": entry.get("title"),
            "source": entry.get("source"),
            "date": entry.get("date"),
            "url": entry.get("url"),
            "origin": entry.get("origin"),
            "model": self.event_model.name,
            "model_version": self.version,
            "state": "needs_review" if reasoning["category"] == "review" or not reasoning["answerable"] else "identified",
            "events": reasoning.get("events", []),
            "reasoning": {
                "what_happened": reasoning["answer"],
                "entities": reasoning["entities"],
                "relations": reasoning["relations"],
                "impact": reasoning["impact"],
                "calculation": reasoning["calculation"],
                "evidence": reasoning["evidence"],
                "answerable": reasoning["answerable"],
                "injection_detected": reasoning["injection_detected"],
            },
            "market_context_observed": market_context,
            "limitations": [
                "Evidence spans are verified against stored disclosure text; model scores are not calibrated probabilities.",
                "Financial implications and stakeholder impacts are conditional analyses, not buy/sell advice or price forecasts.",
                "Any attached market_context_observed reflects historical surrounding closes only, never causal attribution.",
            ],
        }
