"""Train ModernBERT frozen-head and fine-tuned classifiers, tune thresholds on calibration, export ONNX INT8, and evaluate."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from threadpoolctl import threadpool_limits
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from stockbrief.fixtures import Document
from stockbrief.nlp import chunks_for
from stockbrief.semantic import EVENT_BOUNDARY
from training.adapter import DomainEventAdapter, ONNXEndToEndScorer
from training.dataset_utils import build_training_samples
from training.evaluator import evaluate_domain_split, evaluate_repo_benchmarks
from training.onnx_export import ModernBertClassifier, export_and_quantize
from training.schema import VALID_LABELS, load_jsonl

CLASSES = list(VALID_LABELS)
BASE_DIR = Path("models/modernbert-base-pretrained")
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rss_mib() -> float:
    return round(psutil.Process().memory_info().rss / (1024**2), 1)


def encode_texts(
    encoder: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 160,
    batch_size: int = 16,
) -> np.ndarray:
    encoder.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            hs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp(min=1e-9)
            vecs.append((pooled / norm).numpy())
    return np.concatenate(vecs, axis=0)


class ArrayScorer:
    """In-memory chunk probability scorer using a precomputed predict_proba callable."""

    def __init__(self, name: str, version: str, classes: list[str], fn):
        self.name = name
        self.version = version
        self.classes = classes
        self.fn = fn

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.fn(texts)


def precompute_chunk_probs(rows: list[dict[str, Any]], score_fn) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    cached_probs: list[np.ndarray] = []
    for row in rows:
        chunks = chunks_for(tuple(row["pages"]), EVENT_BOUNDARY)
        if not chunks:
            cached_probs.append(np.zeros((0, len(CLASSES)), dtype=np.float32))
        else:
            cached_probs.append(score_fn([c.text for c in chunks]))
    return rows, cached_probs


def predict_from_chunk_probs(
    rows: list[dict[str, Any]],
    chunk_probs: list[np.ndarray],
    floor: float,
    margin: float,
) -> tuple[float, float]:
    expected = [r["label"] for r in rows]
    predicted = []
    for row, probs in zip(rows, chunk_probs):
        if row["source_mode"] == "pdf_extraction_failed" or probs.shape[0] == 0:
            predicted.append("review")
            continue
        events: dict[str, list[float]] = {}
        for p_row in probs:
            order = np.argsort(p_row)[::-1]
            top_i, sec_i = int(order[0]), int(order[1])
            lbl = CLASSES[top_i]
            if lbl != "review" and p_row[top_i] >= floor and (p_row[top_i] - p_row[sec_i]) >= margin:
                events.setdefault(lbl, []).append(float(p_row[top_i]))
        if not events:
            predicted.append("review")
        else:
            best_lbl = max(events, key=lambda k: max(events[k]))
            predicted.append(best_lbl)
    macro = float(f1_score(expected, predicted, labels=CLASSES, average="macro", zero_division=0))
    rev_rec = float(recall_score(expected, predicted, labels=["review"], average="macro", zero_division=0))
    return macro, rev_rec


def tune_thresholds_on_calibration(
    cal_rows: list[dict[str, Any]],
    score_fn,
) -> tuple[float, float, dict[str, float]]:
    _, cal_probs = precompute_chunk_probs(cal_rows, score_fn)
    best_key = (-1.0, -1.0, -1.0)
    best_pair = (0.40, 0.0)
    best_stats = {"cal_macro_f1": 0.0, "cal_review_recall": 0.0}
    for floor in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        for margin in (0.0, 0.04, 0.08, 0.12):
            macro, rev_rec = predict_from_chunk_probs(cal_rows, cal_probs, floor, margin)
            # Prioritize macro_f1 while preserving strong review_recall
            key = (round(macro, 4), round(rev_rec, 4), floor + margin)
            if key > best_key:
                best_key = key
                best_pair = (floor, margin)
                best_stats = {"cal_macro_f1": round(macro, 4), "cal_review_recall": round(rev_rec, 4)}
    return best_pair[0], best_pair[1], best_stats


def train_frozen_head(
    train_texts: list[str],
    train_labels: list[str],
    cal_rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 160,
) -> tuple[ModernBertClassifier, float, float, dict[str, Any]]:
    print("\n=== Training Candidate 1: ModernBERT-base Frozen Encoder + Supervised Head ===", flush=True)
    t0 = time.perf_counter()
    encoder = AutoModel.from_pretrained(str(BASE_DIR), attn_implementation="eager")
    train_embs = encode_texts(encoder, tokenizer, train_texts, max_length=max_length, batch_size=16)

    # Pre-encode calibration chunks once so hyperparameter & threshold search is instant
    cal_chunk_lists = [chunks_for(tuple(r["pages"]), EVENT_BOUNDARY) for r in cal_rows]
    flat_cal_texts = [c.text for chs in cal_chunk_lists for c in chs]
    flat_cal_embs = encode_texts(encoder, tokenizer, flat_cal_texts, max_length=max_length, batch_size=16)

    cal_slices: list[np.ndarray] = []
    pos = 0
    for chs in cal_chunk_lists:
        cal_slices.append(flat_cal_embs[pos:pos + len(chs)])
        pos += len(chs)

    best_cfg = None
    best_head = None
    best_score = (-1.0, -1.0)
    for C in (2.0, 5.0, 10.0, 20.0, 40.0):
        for cw in (None, "balanced"):
            clf = LogisticRegression(C=C, class_weight=cw, max_iter=1000, random_state=19)
            clf.fit(train_embs, train_labels)
            # Align columns to CLASSES order
            col_map = [list(clf.classes_).index(c) for c in CLASSES]
            cal_probs = [
                clf.predict_proba(sl)[:, col_map] if sl.shape[0] > 0 else np.zeros((0, len(CLASSES)))
                for sl in cal_slices
            ]
            for floor in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
                for margin in (0.0, 0.04, 0.08, 0.12):
                    macro, rev_rec = predict_from_chunk_probs(cal_rows, cal_probs, floor, margin)
                    key = (round(macro, 4), round(rev_rec, 4))
                    if key > best_score:
                        best_score = key
                        best_cfg = {"C": C, "class_weight": cw, "floor": floor, "margin": margin, "cal_macro_f1": round(macro, 4), "cal_review_recall": round(rev_rec, 4)}
                        best_head = clf

    assert best_head is not None and best_cfg is not None
    col_map = [list(best_head.classes_).index(c) for c in CLASSES]
    wrapper = ModernBertClassifier(encoder, len(CLASSES))
    with torch.no_grad():
        wrapper.head.weight.copy_(torch.tensor(best_head.coef_[col_map], dtype=torch.float32))
        wrapper.head.bias.copy_(torch.tensor(best_head.intercept_[col_map], dtype=torch.float32))
    wrapper.eval()

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"Frozen-head trained in {elapsed}s; best cal config={best_cfg}", flush=True)
    return wrapper, best_cfg["floor"], best_cfg["margin"], {**best_cfg, "train_seconds": elapsed, "peak_rss_mib": rss_mib()}


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], y_ids: list[int]):
        self.texts = texts
        self.y_ids = y_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.y_ids[idx]


def train_finetuned(
    train_texts: list[str],
    train_labels: list[str],
    cal_rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 160,
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 3e-5,
) -> tuple[ModernBertClassifier, float, float, dict[str, Any]]:
    print("\n=== Training Candidate 2: ModernBERT-base Fine-Tuned Classifier ===", flush=True)
    t0 = time.perf_counter()
    set_seed(19)
    encoder = AutoModel.from_pretrained(str(BASE_DIR), attn_implementation="eager")
    model = ModernBertClassifier(encoder, len(CLASSES), normalize=False, dropout=0.1)

    y_ids = [CLASSES.index(lbl) for lbl in train_labels]
    ds = TextLabelDataset(train_texts, y_ids)

    def collate_fn(batch: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_texts = [item[0] for item in batch]
        b_y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        enc = tokenizer(b_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], b_y

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(19),
    )

    counts = np.bincount(np.asarray(y_ids), minlength=len(CLASSES))
    raw_weights = counts.max() / np.maximum(counts, 1)
    weights = torch.tensor(np.clip(raw_weights, 1.0, 2.5), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)

    best_state = None
    best_cal_key = (-1.0, -1.0)
    history = []
    peak_rss = rss_mib()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for b_ids, b_mask, b_y in loader:
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            peak_rss = max(peak_rss, rss_mib())

        # Score on calibration split
        model.eval()
        def pt_score_fn(txts: list[str]) -> np.ndarray:
            out = []
            with torch.no_grad():
                for i in range(0, len(txts), 16):
                    e = tokenizer(txts[i:i + 16], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                    lg = model(e["input_ids"], e["attention_mask"])
                    out.append(torch.softmax(lg, dim=-1).numpy())
            return np.concatenate(out, axis=0)

        floor, margin, cal_stats = tune_thresholds_on_calibration(cal_rows, pt_score_fn)
        ep_sec = round(time.perf_counter() - ep_t0, 1)
        avg_loss = round(total_loss / max(1, len(loader)), 4)
        rec = {
            "epoch": epoch,
            "loss": avg_loss,
            "cal_macro_f1": cal_stats["cal_macro_f1"],
            "cal_review_recall": cal_stats["cal_review_recall"],
            "floor": floor,
            "margin": margin,
            "epoch_seconds": ep_sec,
            "rss_mib": rss_mib(),
        }
        history.append(rec)
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss}, cal_macro_f1={rec['cal_macro_f1']}, cal_rev_rec={rec['cal_review_recall']}, floor={floor}, margin={margin}, time={ep_sec}s, RSS={rec['rss_mib']} MiB", flush=True)

        key = (rec["cal_macro_f1"], rec["cal_review_recall"])
        if key > best_cal_key:
            best_cal_key = key
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    def final_score_fn(txts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(txts), 16):
                e = tokenizer(txts[i:i + 16], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                lg = model(e["input_ids"], e["attention_mask"])
                out.append(torch.softmax(lg, dim=-1).numpy())
        return np.concatenate(out, axis=0)

    floor, margin, cal_stats = tune_thresholds_on_calibration(cal_rows, final_score_fn)
    total_sec = round(time.perf_counter() - t0, 2)
    meta = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "floor": floor,
        "margin": margin,
        **cal_stats,
        "history": history,
        "train_seconds": total_sec,
        "peak_rss_mib": peak_rss,
    }
    return model, floor, margin, meta


def compare_quantized_vs_unquantized(
    model_dir: Path,
    floor: float,
    margin: float,
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        scorer_fp32 = ONNXEndToEndScorer(model_dir, "model.onnx")
        scorer_int8 = ONNXEndToEndScorer(model_dir, "model_quantized.onnx")
        adapter_fp32 = DomainEventAdapter(scorer_fp32, floor=floor, margin=margin)
        adapter_int8 = DomainEventAdapter(scorer_int8, floor=floor, margin=margin)

        res_fp32 = evaluate_domain_split(f"{scorer_fp32.name} (ONNX FP32)", adapter_fp32.analyze, test_rows)
        res_int8 = evaluate_domain_split(f"{scorer_int8.name} (ONNX INT8)", adapter_int8.analyze, test_rows)

        preds_fp32 = []
        preds_int8 = []
        for row in test_rows:
            doc = Document(row["issuer_symbol"], tuple(row["pages"]), row["doc_id"])
            if row["source_mode"] == "pdf_extraction_failed":
                preds_fp32.append("review")
                preds_int8.append("review")
            else:
                preds_fp32.append(adapter_fp32.analyze(doc)["category"])
                preds_int8.append(adapter_int8.analyze(doc)["category"])
        agree = sum(a == b for a, b in zip(preds_fp32, preds_int8))

    return {
        "fp32_test": res_fp32,
        "int8_test": res_int8,
        "prediction_agreement_rate": round(agree / max(1, len(test_rows)), 4),
        "prediction_agreement_count": f"{agree}/{len(test_rows)}",
    }


def main() -> None:
    torch.set_num_threads(4)
    set_seed(19)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    cal_rows = load_jsonl(DATA_DIR / "calibration.jsonl")
    test_rows = load_jsonl(DATA_DIR / "test.jsonl")

    train_texts, train_labels, sample_counts = build_training_samples(train_rows)
    print(f"Prepared {len(train_texts)} training samples from {len(train_rows)} docs: {sample_counts}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

    # Candidate 1: ModernBERT frozen + supervised LogisticRegression head
    frozen_model, f_floor, f_margin, f_meta = train_frozen_head(train_texts, train_labels, cal_rows, tokenizer)
    frozen_dir = Path("models/modernbert-frozen-head")
    frozen_export = export_and_quantize(
        frozen_model,
        BASE_DIR,
        frozen_dir,
        {
            "name": "ModernBERT-base Frozen / Supervised Head",
            "version": "stockbrief-modernbert-frozen-v1",
            "classes": CLASSES,
            "floor": f_floor,
            "margin": f_margin,
        },
        train_texts[:6],
    )
    frozen_cmp = compare_quantized_vs_unquantized(frozen_dir, f_floor, f_margin, test_rows)
    with threadpool_limits(limits=1):
        f_int8_adapter = DomainEventAdapter(ONNXEndToEndScorer(frozen_dir, "model_quantized.onnx"), floor=f_floor, margin=f_margin)
        frozen_cal = evaluate_domain_split("ModernBERT-base Frozen (ONNX INT8)", f_int8_adapter.analyze, cal_rows)
        frozen_repo = evaluate_repo_benchmarks(f_int8_adapter.analyze)

    # Free memory before fine-tuning
    del frozen_model

    # Candidate 2: Fine-tuned ModernBERT-base
    ft_model, ft_floor, ft_margin, ft_meta = train_finetuned(train_texts, train_labels, cal_rows, tokenizer)
    ft_dir = Path("models/modernbert-finetuned")
    ft_export = export_and_quantize(
        ft_model,
        BASE_DIR,
        ft_dir,
        {
            "name": "ModernBERT-base Fine-Tuned",
            "version": "stockbrief-modernbert-ft-v1",
            "classes": CLASSES,
            "floor": ft_floor,
            "margin": ft_margin,
        },
        train_texts[:6],
    )
    ft_cmp = compare_quantized_vs_unquantized(ft_dir, ft_floor, ft_margin, test_rows)
    with threadpool_limits(limits=1):
        ft_int8_adapter = DomainEventAdapter(ONNXEndToEndScorer(ft_dir, "model_quantized.onnx"), floor=ft_floor, margin=ft_margin)
        ft_cal = evaluate_domain_split("ModernBERT-base Fine-Tuned (ONNX INT8)", ft_int8_adapter.analyze, cal_rows)
        ft_repo = evaluate_repo_benchmarks(ft_int8_adapter.analyze)

    # Compare against pinned BGE baseline + check promotion gates
    bge = json.loads((RESULTS_DIR / "bge_baseline.json").read_text(encoding="utf-8"))
    bge_test = bge["test"]

    def check_gates(candidate_int8: dict[str, Any], repo_res: dict[str, Any], export_info: dict[str, Any]) -> dict[str, Any]:
        gates = {
            "improves_test_macro_f1": candidate_int8["macro_f1"] > bge_test["macro_f1"],
            "preserves_review_recall": candidate_int8["review_recall"] >= bge_test["review_recall"],
            "improves_or_keeps_multilabel_f1": candidate_int8["multilabel_micro_f1"] >= bge_test["multilabel_micro_f1"],
            "repo_tata_passed": bool(repo_res["tata_acceptance_passed"]),
            "repo_test_32_review_recall_ok": repo_res["repo_test_32"]["review_recall"] >= 0.75,
            "cpu_p95_within_150ms_budget": (candidate_int8["latency_ms"]["p95"] or 9999) <= 150.0,
            "int8_size_within_80mb_edge_budget": export_info["int8_bytes"] <= 80 * 1024 * 1024,
        }
        gates["promote_as_default"] = all(gates.values())
        return gates

    report = {
        "training_samples": len(train_texts),
        "training_sample_counts": sample_counts,
        "bge_baseline": bge,
        "modernbert_frozen_head": {
            "train_meta": f_meta,
            "export": frozen_export,
            "calibration_int8": frozen_cal,
            "quantized_vs_unquantized": frozen_cmp,
            "repo_benchmarks": frozen_repo,
            "gates": check_gates(frozen_cmp["int8_test"], frozen_repo, frozen_export),
        },
        "modernbert_finetuned": {
            "train_meta": ft_meta,
            "export": ft_export,
            "calibration_int8": ft_cal,
            "quantized_vs_unquantized": ft_cmp,
            "repo_benchmarks": ft_repo,
            "gates": check_gates(ft_cmp["int8_test"], ft_repo, ft_export),
        },
    }
    out_file = RESULTS_DIR / "training_benchmark.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved full benchmark results to {out_file}", flush=True)
    for key in ("modernbert_frozen_head", "modernbert_finetuned"):
        t = report[key]["quantized_vs_unquantized"]["int8_test"]
        print(
            f"{key}: test macro_f1={t['macro_f1']} (BGE={bge_test['macro_f1']}), "
            f"rev_r={t['review_recall']} (BGE={bge_test['review_recall']}), "
            f"rev_p={t['review_precision']}, multilabel_micro_f1={t['multilabel_micro_f1']}, "
            f"p50={t['latency_ms']['p50']}ms, p95={t['latency_ms']['p95']}ms, "
            f"int8_MB={round(report[key]['export']['int8_bytes']/1e6, 1)}, "
            f"agree={report[key]['quantized_vs_unquantized']['prediction_agreement_count']}, "
            f"promote={report[key]['gates']['promote_as_default']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
