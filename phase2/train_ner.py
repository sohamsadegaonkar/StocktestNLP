"""Train ModernBERT-base BIO Financial NER with checkpointing, ONNX INT8 export, and entity-level F1 eval."""
from __future__ import annotations

from collections import Counter
import copy
from hashlib import file_digest
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
import psutil
from threadpoolctl import threadpool_limits
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from phase2.schema import NER_BIO_LABELS, NER_ENTITY_TYPES, load_jsonl

BASE_MODEL = Path("models/modernbert-base-pretrained")
DATA_DIR = Path("phase2/data")
OUT_DIR = Path("models/phase2/ner_modernbert")
RESULTS_DIR = Path("phase2/results")

TAG2ID = {t: i for i, t in enumerate(NER_BIO_LABELS)}
ID2TAG = {i: t for i, t in enumerate(NER_BIO_LABELS)}


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def align_bio_labels(text: str, entities: list[dict[str, Any]], tokenizer: Any, max_length: int = 128) -> dict[str, Any]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    labels = []
    for s, e in offsets:
        if s == 0 and e == 0:
            labels.append(-100)
            continue
        assigned = "O"
        for ent in entities:
            es, ee, elbl = int(ent["start"]), int(ent["end"]), ent["label"]
            if e <= es or s >= ee:
                continue
            # Check if this token starts at or before the entity start (first overlapping subword)
            if s <= es or (s > es and (not labels or labels[-1] in (-100, 0) or not ID2TAG[labels[-1]].endswith(elbl))):
                assigned = f"B-{elbl}"
            else:
                assigned = f"I-{elbl}"
            break
        labels.append(TAG2ID[assigned])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "offsets": offsets,
    }


class NerDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int = 128):
        self.items = [align_bio_labels(r["text"], r["entities"], tokenizer, max_length) for r in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def collate_ner(batch: list[dict[str, Any]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, mask, lbls = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        mask.append(x["attention_mask"] + [0] * pad)
        lbls.append(x["labels"] + [-100] * pad)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(lbls, dtype=torch.long),
    )


class ModernBertNer(nn.Module):
    def __init__(self, encoder: nn.Module, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.drop(hs))


def bio_to_spans(tag_ids: list[int], offsets: list[tuple[int, int]], text: str = "") -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    cur_type = None
    cur_start = -1
    cur_end = -1
    for tid, (s, e) in zip(tag_ids, offsets):
        if s == 0 and e == 0:
            continue
        tag = ID2TAG.get(int(tid), "O")
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.add((cur_start, cur_end, cur_type))
            cur_type = tag[2:]
            cur_start, cur_end = s, e
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_end = e
        else:
            if cur_type is not None:
                spans.add((cur_start, cur_end, cur_type))
                cur_type = None
    if cur_type is not None:
        spans.add((cur_start, cur_end, cur_type))
    if not text:
        return spans
    trimmed: set[tuple[int, int, str]] = set()
    for s, e, t in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            trimmed.add((s, e, t))
    return trimmed


def evaluate_ner_predictions(
    rows: list[dict[str, Any]],
    logits_fn,
    tokenizer: Any,
    max_length: int = 128,
) -> dict[str, Any]:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    latencies = []

    for r in rows:
        enc = tokenizer(r["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        inp = np.asarray([enc["input_ids"]], dtype=np.int64)
        msk = np.asarray([enc["attention_mask"]], dtype=np.int64)
        t0 = time.perf_counter_ns()
        logits = logits_fn(inp, msk)[0]
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
        pred_ids = np.argmax(logits, axis=-1).tolist()

        pred_spans = bio_to_spans(pred_ids, enc["offset_mapping"], r["text"])
        gold_spans = {
            (int(e["start"]), int(e["end"]), e["label"])
            for e in r["entities"]
            if int(e["end"]) <= (enc["offset_mapping"][-2][1] if len(enc["offset_mapping"]) >= 2 else len(r["text"]))
        }
        for sp in pred_spans & gold_spans:
            tp[sp[2]] += 1
        for sp in pred_spans - gold_spans:
            fp[sp[2]] += 1
        for sp in gold_spans - pred_spans:
            fn[sp[2]] += 1

    tot_tp = sum(tp.values())
    tot_fp = sum(fp.values())
    tot_fn = sum(fn.values())
    prec = tot_tp / max(1, tot_tp + tot_fp)
    rec = tot_tp / max(1, tot_tp + tot_fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    per_type = {}
    for et in NER_ENTITY_TYPES:
        t, p_err, r_err = tp[et], fp[et], fn[et]
        ep = t / max(1, t + p_err)
        er = t / max(1, t + r_err)
        ef1 = (2 * ep * er / (ep + er)) if (ep + er) > 0 else 0.0
        per_type[et] = {"precision": round(ep, 4), "recall": round(er, 4), "f1": round(ef1, 4), "support": t + r_err}

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "per_entity_type": per_type,
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    set_seed(19)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows_all = load_jsonl(DATA_DIR / "ner_train.jsonl")
    # Stratified/shuffled CPU budget: 640 NSE NER rows (40 steps/epoch x 2 epochs = 80 steps, ~9 min CPU)
    rng = random.Random(19)
    shuffled = list(train_rows_all)
    rng.shuffle(shuffled)
    train_rows = shuffled[:640]

    val_rows = load_jsonl(DATA_DIR / "ner_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "ner_test.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ds_train = NerDataset(train_rows, tokenizer, max_length=96)
    loader = DataLoader(
        ds_train,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_ner(b, pad_id),
        generator=torch.Generator().manual_seed(19),
    )

    encoder = AutoModel.from_pretrained(str(BASE_MODEL), attn_implementation="eager")
    model = ModernBertNer(encoder, len(NER_BIO_LABELS))
    ckpt_path = OUT_DIR / "checkpoint.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_state = None
    history = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    epochs = 2
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * len(loader) * epochs)), num_training_steps=len(loader) * epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_state = ckpt.get("best_state", copy.deepcopy(model.state_dict()))
        best_f1 = ckpt.get("best_f1", -1.0)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed NER checkpoint from epoch {ckpt['epoch']} (best_val_f1={best_f1})", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for step, (b_ids, b_mask, b_lbls) in enumerate(loader, start=1):
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = criterion(logits.view(-1, len(NER_BIO_LABELS)), b_lbls.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += float(loss.item())
            if step % 10 == 0 or step == len(loader):
                print(f"  NER Ep {epoch} step {step}/{len(loader)} loss={round(tot_loss / step, 4)}", flush=True)

        model.eval()
        def pt_fn(inp, msk):
            with torch.no_grad():
                return model(torch.from_numpy(inp), torch.from_numpy(msk)).numpy()
        val_res = evaluate_ner_predictions(val_rows, pt_fn, tokenizer)
        ep_sec = round(time.perf_counter() - t0, 1)
        rec = {
            "epoch": epoch,
            "loss": round(tot_loss / max(1, len(loader)), 4),
            "val_f1": val_res["f1"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "seconds": ep_sec,
            "rss_mib": round(psutil.Process().memory_info().rss / (1024**2), 1),
        }
        history.append(rec)
        print(f"NER Epoch {epoch}/{epochs}: loss={rec['loss']}, val_f1={rec['val_f1']}, time={ep_sec}s, RSS={rec['rss_mib']} MiB", flush=True)
        if val_res["f1"] > best_f1:
            best_f1 = val_res["f1"]
            best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_state": best_state, "best_f1": best_f1, "history": history}, ckpt_path)

    assert best_state is not None
    if start_epoch > epochs and ckpt_path.is_file():
        # Re-evaluate final checkpoint model vs saved best_state after any span decoder fix
        def _pt_eval_fn(inp, msk):
            with torch.no_grad():
                return model(torch.from_numpy(inp), torch.from_numpy(msk)).numpy()
        last_val = evaluate_ner_predictions(val_rows, _pt_eval_fn, tokenizer)
        last_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        best_val = evaluate_ner_predictions(val_rows, _pt_eval_fn, tokenizer)
        print(f"Resumed NER comparison -> Epoch1 val_f1={best_val['f1']}, Epoch2 val_f1={last_val['f1']}", flush=True)
        if last_val["f1"] >= best_val["f1"]:
            best_state = last_state
    model.load_state_dict(best_state)
    model.eval()

    # Export to ONNX FP32 + INT8
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        src = BASE_MODEL / fname
        if src.is_file():
            shutil.copy2(src, OUT_DIR / fname)

    sample = tokenizer([train_rows[0]["text"], train_rows[1]["text"]], padding=True, truncation=True, max_length=128, return_tensors="pt")
    fp32_path = OUT_DIR / "model.onnx"
    int8_path = OUT_DIR / "model_quantized.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        prep = Path(tmpdir) / "prep.onnx"
        src_for_q = fp32_path
        try:
            quant_pre_process(str(fp32_path), str(prep), skip_symbolic_shape=True)
            src_for_q = prep
        except Exception:
            src_for_q = fp32_path
        quantize_dynamic(str(src_for_q), str(int8_path), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True)

    with threadpool_limits(limits=1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        s_fp32 = ort.InferenceSession(str(fp32_path), sess_options=opts, providers=["CPUExecutionProvider"])
        s_int8 = ort.InferenceSession(str(int8_path), sess_options=opts, providers=["CPUExecutionProvider"])
        test_fp32 = evaluate_ner_predictions(test_rows, lambda i, m: s_fp32.run(None, {"input_ids": i, "attention_mask": m})[0], tokenizer)
        test_int8 = evaluate_ner_predictions(test_rows, lambda i, m: s_int8.run(None, {"input_ids": i, "attention_mask": m})[0], tokenizer)

    meta = {
        "labels": NER_BIO_LABELS,
        "fp32_bytes": fp32_path.stat().st_size,
        "fp32_sha256": sha256_file(fp32_path),
        "int8_bytes": int8_path.stat().st_size,
        "int8_sha256": sha256_file(int8_path),
    }
    (OUT_DIR / "ner_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    report = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "history": history,
        "artifact": meta,
        "test_fp32": test_fp32,
        "test_int8": test_int8,
    }
    (RESULTS_DIR / "ner_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"NER Test FP32 F1={test_fp32['f1']} (p95={test_fp32['p95_ms']}ms) | INT8 F1={test_int8['f1']} (p95={test_int8['p95_ms']}ms, {round(meta['int8_bytes']/1e6, 1)}MB)", flush=True)


if __name__ == "__main__":
    main()
