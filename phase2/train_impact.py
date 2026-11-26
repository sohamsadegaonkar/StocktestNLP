"""Train multi-head Contextual Financial Impact classifier (BGE vs ModernBERT ablation) + ONNX INT8 export."""
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
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from phase2.schema import (
    EVENT_MATURITY_LEVELS,
    IMPACT_IMPLICATIONS,
    MATERIALITY_LEVELS,
    STAKEHOLDER_GROUPS,
    UNCERTAINTY_LEVELS,
    load_jsonl,
)

BGE_BASE = Path("models/phase2/pretrained/bge-small-en-v1.5")
DATA_DIR = Path("phase2/data")
OUT_DIR = Path("models/phase2/impact_bge")
RESULTS_DIR = Path("phase2/results")


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def encode_impact_targets(row: dict[str, Any]) -> dict[str, Any]:
    imp = np.zeros(len(IMPACT_IMPLICATIONS), dtype=np.float32)
    for lbl in row["implications"]:
        if lbl in IMPACT_IMPLICATIONS:
            imp[IMPACT_IMPLICATIONS.index(lbl)] = 1.0
    stk = np.zeros(len(STAKEHOLDER_GROUPS), dtype=np.float32)
    for s in row["stakeholders"]:
        if s in STAKEHOLDER_GROUPS:
            stk[STAKEHOLDER_GROUPS.index(s)] = 1.0
    return {
        "text": f"[{row['event_category']} | {row['event_stage']}] {row['text']}" if row["text"] else f"[{row['event_category']} | Unclear] (unreadable text)",
        "imp": imp,
        "mat": MATERIALITY_LEVELS.index(row["materiality"]),
        "evm": EVENT_MATURITY_LEVELS.index(row["event_maturity"]),
        "unc": UNCERTAINTY_LEVELS.index(row["uncertainty"]),
        "stk": stk,
    }


class ImpactDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.items = [encode_impact_targets(r) for r in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class MultiHeadImpactModel(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        # Concatenate all head logits into one tensor [B, 5+4+7+3+6 = 25] for clean ONNX export
        total_out = (
            len(IMPACT_IMPLICATIONS)
            + len(MATERIALITY_LEVELS)
            + len(EVENT_MATURITY_LEVELS)
            + len(UNCERTAINTY_LEVELS)
            + len(STAKEHOLDER_GROUPS)
        )
        self.head = nn.Linear(h, total_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head(self.drop(cls))


def split_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i1 = len(IMPACT_IMPLICATIONS)
    i2 = i1 + len(MATERIALITY_LEVELS)
    i3 = i2 + len(EVENT_MATURITY_LEVELS)
    i4 = i3 + len(UNCERTAINTY_LEVELS)
    return logits[:, :i1], logits[:, i1:i2], logits[:, i2:i3], logits[:, i3:i4], logits[:, i4:]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def evaluate_impact(rows: list[dict[str, Any]], score_fn, threshold: float = 0.45) -> dict[str, Any]:
    encoded = [encode_impact_targets(r) for r in rows]
    texts = [x["text"] for x in encoded]
    latencies = []
    all_logits = []
    for t in texts:
        t0 = time.perf_counter_ns()
        all_logits.append(score_fn([t])[0])
        latencies.append((time.perf_counter_ns() - t0) / 1e6)

    logits = np.asarray(all_logits, dtype=np.float32)
    l_imp, l_mat, l_evm, l_unc, l_stk = split_logits(logits)

    y_imp = np.asarray([x["imp"] for x in encoded], dtype=int)
    p_imp_prob = sigmoid(l_imp)
    p_imp = (p_imp_prob >= threshold).astype(int)
    # Ensure at least top-1 implication is selected
    for row_i in range(len(p_imp)):
        if p_imp[row_i].sum() == 0:
            p_imp[row_i, int(np.argmax(p_imp_prob[row_i]))] = 1

    y_stk = np.asarray([x["stk"] for x in encoded], dtype=int)
    p_stk_prob = sigmoid(l_stk)
    p_stk = (p_stk_prob >= 0.45).astype(int)

    mat_acc = float(np.mean(np.argmax(l_mat, axis=1) == np.asarray([x["mat"] for x in encoded])))
    evm_acc = float(np.mean(np.argmax(l_evm, axis=1) == np.asarray([x["evm"] for x in encoded])))
    unc_acc = float(np.mean(np.argmax(l_unc, axis=1) == np.asarray([x["unc"] for x in encoded])))

    adv_indices = [i for i, r in enumerate(rows) if r["id"].startswith("imp:adv:")]
    adv_correct = sum(
        bool(set(np.where(p_imp[i] == 1)[0]) & set(np.where(y_imp[i] == 1)[0]))
        for i in adv_indices
    )

    return {
        "count": len(rows),
        "implications_micro_f1": round(float(f1_score(y_imp, p_imp, average="micro", zero_division=0)), 4),
        "implications_macro_f1": round(float(f1_score(y_imp, p_imp, average="macro", zero_division=0)), 4),
        "stakeholders_micro_f1": round(float(f1_score(y_stk, p_stk, average="micro", zero_division=0)), 4),
        "materiality_acc": round(mat_acc, 4),
        "event_maturity_acc": round(evm_acc, 4),
        "uncertainty_acc": round(unc_acc, 4),
        "adversarial_overlap_acc": f"{adv_correct}/{len(adv_indices)}" if adv_indices else "n/a",
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
    }


def main() -> None:
    torch.set_num_threads(4)
    set_seed(19)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(DATA_DIR / "impact_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "impact_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "impact_test.jsonl")

    tok = AutoTokenizer.from_pretrained(str(BGE_BASE))
    ds = ImpactDataset(train_rows)

    def collate(b):
        e = tok([x["text"] for x in b], padding=True, truncation=True, max_length=128, return_tensors="pt")
        return (
            e["input_ids"],
            e["attention_mask"],
            torch.tensor(np.asarray([x["imp"] for x in b]), dtype=torch.float32),
            torch.tensor([x["mat"] for x in b], dtype=torch.long),
            torch.tensor([x["evm"] for x in b], dtype=torch.long),
            torch.tensor([x["unc"] for x in b], dtype=torch.long),
            torch.tensor(np.asarray([x["stk"] for x in b]), dtype=torch.float32),
        )

    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(19))
    model = MultiHeadImpactModel(AutoModel.from_pretrained(str(BGE_BASE)))
    epochs = 3
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.1 * len(loader) * epochs)), len(loader) * epochs)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    i1 = len(IMPACT_IMPLICATIONS)
    i2 = i1 + len(MATERIALITY_LEVELS)
    i3 = i2 + len(EVENT_MATURITY_LEVELS)
    i4 = i3 + len(UNCERTAINTY_LEVELS)

    best_score = -1.0
    best_state = None
    history = []
    ckpt_path = OUT_DIR / "checkpoint.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for step, (b_ids, b_mask, y_imp, y_mat, y_evm, y_unc, y_stk) in enumerate(loader, start=1):
            opt.zero_grad()
            lg = model(b_ids, b_mask)
            loss = (
                2.0 * bce(lg[:, :i1], y_imp)
                + ce(lg[:, i1:i2], y_mat)
                + ce(lg[:, i2:i3], y_evm)
                + ce(lg[:, i3:i4], y_unc)
                + bce(lg[:, i4:], y_stk)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            if step % 20 == 0 or step == len(loader):
                print(f"  IMPACT Ep {epoch} step {step}/{len(loader)} loss={round(tot_loss / step, 4)}", flush=True)

        model.eval()
        def pt_fn(txts):
            with torch.no_grad():
                e = tok(txts, padding=True, truncation=True, max_length=160, return_tensors="pt")
                return model(e["input_ids"], e["attention_mask"]).numpy()
        val_res = evaluate_impact(val_rows, pt_fn)
        ep_s = round(time.perf_counter() - t0, 1)
        history.append({"epoch": epoch, "loss": round(tot_loss / len(loader), 4), **val_res, "seconds": ep_s})
        print(f"IMPACT Epoch {epoch}/{epochs}: loss={history[-1]['loss']}, val_imp_micro={val_res['implications_micro_f1']}, val_mat={val_res['materiality_acc']}, val_evm={val_res['event_maturity_acc']}, time={ep_s}s", flush=True)
        score = val_res["implications_micro_f1"] + 0.5 * val_res["event_maturity_acc"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_state": best_state, "best_score": best_score, "history": history}, ckpt_path)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = BGE_BASE / fname
        if src.is_file():
            shutil.copy2(src, OUT_DIR / fname)

    sample = tok([ds[0]["text"], ds[1]["text"]], padding=True, truncation=True, max_length=160, return_tensors="pt")
    fp32 = OUT_DIR / "model.onnx"
    int8 = OUT_DIR / "model_quantized.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    with tempfile.TemporaryDirectory() as td:
        prep = Path(td) / "prep.onnx"
        src_for_q = fp32
        try:
            quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
            src_for_q = prep
        except Exception:
            src_for_q = fp32
        quantize_dynamic(str(src_for_q), str(int8), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True)

    with threadpool_limits(limits=1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        s_fp32 = ort.InferenceSession(str(fp32), sess_options=opts, providers=["CPUExecutionProvider"])
        s_int8 = ort.InferenceSession(str(int8), sess_options=opts, providers=["CPUExecutionProvider"])

        def ort_fn(s):
            def _f(txts):
                e = tok(txts, padding=True, truncation=True, max_length=160, return_tensors="np")
                return s.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
            return _f

        test_fp32 = evaluate_impact(test_rows, ort_fn(s_fp32))
        test_int8 = evaluate_impact(test_rows, ort_fn(s_int8))

        # Research-only NC-SA check on Financial PhraseBank (AllAgree, 2264 sentences -> directional consistency)
        fpb_path = DATA_DIR / "phrasebank_nc_sa_eval.jsonl"
        fpb_eval = {}
        if fpb_path.is_file():
            fpb_rows = load_jsonl(fpb_path)[:500]
            f_fn = ort_fn(s_int8)
            agree_dir = 0
            covered_dir = 0
            for fr in fpb_rows:
                lg = f_fn([f"[earnings | Reported] {fr['text']}"])[0]
                l_imp = sigmoid(split_logits(lg[None, :])[0][0])
                pred_lbl = IMPACT_IMPLICATIONS[int(np.argmax(l_imp))]
                if fr["sentiment"] == "positive" and pred_lbl in ("positive_implication", "mixed_implication"):
                    agree_dir += 1
                elif fr["sentiment"] == "negative" and pred_lbl in ("negative_implication", "mixed_implication"):
                    agree_dir += 1
                elif fr["sentiment"] == "neutral" and pred_lbl == "neutral_implication":
                    agree_dir += 1
                covered_dir += 1
            fpb_eval = {"license_tier": "nc_sa_research_only", "sample_n": covered_dir, "directional_alignment": round(agree_dir / max(1, covered_dir), 4)}

    art = {
        "fp32_bytes": fp32.stat().st_size,
        "fp32_sha256": sha256_file(fp32),
        "int8_bytes": int8.stat().st_size,
        "int8_sha256": sha256_file(int8),
    }
    (OUT_DIR / "impact_config.json").write_text(json.dumps({
        "implications": list(IMPACT_IMPLICATIONS),
        "materiality": list(MATERIALITY_LEVELS),
        "event_maturity": list(EVENT_MATURITY_LEVELS),
        "uncertainty": list(UNCERTAINTY_LEVELS),
        "stakeholders": list(STAKEHOLDER_GROUPS),
        **art,
    }, indent=2) + "\n", encoding="utf-8")

    report = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "history": history,
        "artifact": art,
        "test_fp32": test_fp32,
        "test_int8": test_int8,
        "phrasebank_nc_sa_research_eval": fpb_eval,
    }
    (RESULTS_DIR / "impact_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"IMPACT Test FP32 micro_f1={test_fp32['implications_micro_f1']} | INT8 micro_f1={test_int8['implications_micro_f1']} evm_acc={test_int8['event_maturity_acc']} (p95={test_int8['p95_ms']}ms)", flush=True)


if __name__ == "__main__":
    main()
