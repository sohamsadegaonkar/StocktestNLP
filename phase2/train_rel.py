"""Train BGE-small-en-v1.5 Financial Relation Extraction classifier with checkpointing and ONNX INT8 export."""
from __future__ import annotations

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
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from phase2.schema import REL_LABELS, load_jsonl

BASE_MODEL = Path("models/phase2/pretrained/bge-small-en-v1.5")
DATA_DIR = Path("phase2/data")
OUT_DIR = Path("models/phase2/rel_bge")
RESULTS_DIR = Path("phase2/results")


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_rel_input(row: dict[str, Any]) -> str:
    txt = row["text"]
    s = row["subject"]
    o = row["object"]
    return f"Subject: {s} | Object: {o} || {txt}"


class RelDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.texts = [format_rel_input(r) for r in rows]
        self.labels = [REL_LABELS.index(r["relation"]) for r in rows]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


class BgeRelClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head(self.drop(cls))


def evaluate_rel(rows: list[dict[str, Any]], score_fn) -> dict[str, Any]:
    texts = [format_rel_input(r) for r in rows]
    expected = [r["relation"] for r in rows]
    latencies = []
    preds = []
    for t in texts:
        t0 = time.perf_counter_ns()
        logits = score_fn([t])[0]
        latencies.append((time.perf_counter_ns() - t0) / 1e6)
        preds.append(REL_LABELS[int(np.argmax(logits))])

    p, r, f1, sup = precision_recall_fscore_support(expected, preds, labels=list(REL_LABELS), zero_division=0)
    nse_indices = [i for i, row in enumerate(rows) if row["id"].startswith("nserel:")]
    nse_acc = sum(preds[i] == expected[i] for i in nse_indices) / max(1, len(nse_indices))

    return {
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float(f1_score(expected, preds, average="weighted", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(expected, preds, average="micro", zero_division=0)), 4),
        "nse_event_relation_acc": round(float(nse_acc), 4),
        "nse_event_relation_count": len(nse_indices),
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "per_relation": [
            {"relation": rel, "precision": round(float(pv), 4), "recall": round(float(rv), 4), "f1": round(float(fv), 4), "support": int(sv)}
            for rel, pv, rv, fv, sv in zip(REL_LABELS, p, r, f1, sup)
        ],
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    set_seed(19)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows_all = load_jsonl(DATA_DIR / "rel_train.jsonl")
    # Stratified CPU budget: keep all NSE relation examples + balanced FinRED sample up to 1200 total
    rng = random.Random(19)
    nse_train = [r for r in train_rows_all if r["id"].startswith("nserel:")]
    by_rel: dict[str, list[dict[str, Any]]] = {rel: [] for rel in REL_LABELS}
    for r in train_rows_all:
        if not r["id"].startswith("nserel:"):
            by_rel[r["relation"]].append(r)
    train_rows = list(nse_train)
    for rel, grp in by_rel.items():
        rng.shuffle(grp)
        train_rows.extend(grp[:90])
    rng.shuffle(train_rows)

    val_rows = load_jsonl(DATA_DIR / "rel_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "rel_test.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    ds = RelDataset(train_rows)

    def collate(batch):
        b_txt = [b[0] for b in batch]
        b_y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(b_txt, padding=True, truncation=True, max_length=96, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], b_y

    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(19))

    encoder = AutoModel.from_pretrained(str(BASE_MODEL))
    model = BgeRelClassifier(encoder, len(REL_LABELS))

    counts = np.bincount(np.asarray(ds.labels), minlength=len(REL_LABELS))
    raw_w = np.sqrt(counts.max() / np.maximum(counts, 1))
    weights = torch.tensor(np.clip(raw_w, 1.0, 6.0), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * len(loader) * epochs)), num_training_steps=len(loader) * epochs)

    ckpt_path = OUT_DIR / "checkpoint.pt"
    start_epoch = 1
    best_f1 = -1.0
    best_state = None
    history = []

    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_state = ckpt.get("best_state")
        best_f1 = ckpt.get("best_f1", -1.0)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for step, (b_ids, b_mask, b_y) in enumerate(loader, start=1):
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += float(loss.item())
            if step % 25 == 0 or step == len(loader):
                print(f"  REL Ep {epoch} step {step}/{len(loader)} loss={round(tot_loss / step, 4)}", flush=True)

        model.eval()
        def pt_fn(txts: list[str]) -> np.ndarray:
            with torch.no_grad():
                e = tokenizer(txts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                return model(e["input_ids"], e["attention_mask"]).numpy()
        val_res = evaluate_rel(val_rows, pt_fn)
        ep_sec = round(time.perf_counter() - t0, 1)
        rec = {
            "epoch": epoch,
            "loss": round(tot_loss / max(1, len(loader)), 4),
            "val_macro_f1": val_res["macro_f1"],
            "val_micro_f1": val_res["micro_f1"],
            "val_nse_acc": val_res["nse_event_relation_acc"],
            "seconds": ep_sec,
            "rss_mib": round(psutil.Process().memory_info().rss / (1024**2), 1),
        }
        history.append(rec)
        print(f"REL Epoch {epoch}/{epochs}: loss={rec['loss']}, val_macro_f1={rec['val_macro_f1']}, val_micro_f1={rec['val_micro_f1']}, nse_acc={rec['val_nse_acc']}, time={ep_sec}s", flush=True)
        score_key = val_res["micro_f1"] + 0.3 * val_res["macro_f1"]
        if score_key > best_f1:
            best_f1 = score_key
            best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_state": best_state, "best_f1": best_f1, "history": history}, ckpt_path)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "config.json"):
        src = BASE_MODEL / fname
        if src.is_file():
            shutil.copy2(src, OUT_DIR / fname)

    sample = tokenizer([format_rel_input(train_rows[0]), format_rel_input(train_rows[1])], padding=True, truncation=True, max_length=128, return_tensors="pt")
    fp32_path = OUT_DIR / "model.onnx"
    int8_path = OUT_DIR / "model_quantized.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
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

        def run_sess(sess, txts):
            e = tokenizer(txts, padding=True, truncation=True, max_length=128, return_tensors="np")
            return sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]

        test_fp32 = evaluate_rel(test_rows, lambda txts: run_sess(s_fp32, txts))
        test_int8 = evaluate_rel(test_rows, lambda txts: run_sess(s_int8, txts))

    meta = {
        "labels": list(REL_LABELS),
        "fp32_bytes": fp32_path.stat().st_size,
        "fp32_sha256": sha256_file(fp32_path),
        "int8_bytes": int8_path.stat().st_size,
        "int8_sha256": sha256_file(int8_path),
    }
    (OUT_DIR / "rel_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    report = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "history": history,
        "artifact": meta,
        "test_fp32": test_fp32,
        "test_int8": test_int8,
    }
    (RESULTS_DIR / "rel_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REL Test FP32 micro_f1={test_fp32['micro_f1']} macro_f1={test_fp32['macro_f1']} | INT8 micro_f1={test_int8['micro_f1']} macro_f1={test_int8['macro_f1']} (p95={test_int8['p95_ms']}ms, {round(meta['int8_bytes']/1e6, 1)}MB)", flush=True)


if __name__ == "__main__":
    main()
