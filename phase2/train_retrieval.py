"""Train and compare 4 retrieval & reranking architectures on positives + 6 types of hard negatives."""
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
from threadpoolctl import threadpool_limits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from phase2.schema import load_jsonl

BGE_BASE = Path("models/phase2/pretrained/bge-small-en-v1.5")
BGE_PINNED_DIR = Path("models/semantic")
MODERNBERT_BASE = Path("models/modernbert-base-pretrained")
DATA_DIR = Path("phase2/data")
OUT_BGE_FT = Path("models/phase2/bge_retrieval_ft")
OUT_BGE_RERANK = Path("models/phase2/bge_cross_reranker")
OUT_MB_RERANK = Path("models/phase2/modernbert_reranker")
RESULTS_DIR = Path("phase2/results")


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


class BiEncoderPairDataset(Dataset):
    """Query + positive + hard negative triplets for contrastive MultipleNegativesRankingLoss."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.triplets = []
        for r in rows:
            if r["hard_negatives"]:
                for neg in r["hard_negatives"][:2]:
                    self.triplets.append((r["query"], r["positive"], neg["text"]))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.triplets[idx]


class CrossEncoderDataset(Dataset):
    """(query, passage, label) pairs for cross-encoder reranking."""

    def __init__(self, rows: list[dict[str, Any]], max_negs: int = 3):
        self.pairs: list[tuple[str, str, float]] = []
        for r in rows:
            self.pairs.append((r["query"], r["positive"], 1.0))
            for neg in r["hard_negatives"][:max_negs]:
                self.pairs.append((r["query"], neg["text"], 0.0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.pairs[idx]


class BgeBiEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=-1)


class CrossEncoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, use_mean_pool: bool = False):
        super().__init__()
        self.encoder = encoder
        self.use_mean_pool = use_mean_pool
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_mean_pool:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hs[:, 0, :]
        return self.head(self.drop(pooled)).squeeze(-1)


def compute_ranking_metrics(
    queries: list[dict[str, Any]],
    corpus_passages: list[str],
    bi_encode_fn,
    rerank_fn=None,
    top_k_rerank: int = 15,
) -> dict[str, Any]:
    """Evaluate Recall@5, Recall@10, MRR@10, nDCG@10 over combined test corpus + hard negatives."""
    # Corpus = all unique positive + hard negative passages in split
    unique_texts: list[str] = []
    txt_to_idx: dict[str, int] = {}
    for t in corpus_passages:
        if t not in txt_to_idx:
            txt_to_idx[t] = len(unique_texts)
            unique_texts.append(t)

    corpus_embs = bi_encode_fn(unique_texts)
    r5, r10, mrr10, ndcg10, latencies = [], [], [], [], []

    for qrow in queries:
        pos_idx = txt_to_idx[qrow["positive"]]
        t0 = time.perf_counter_ns()
        q_emb = bi_encode_fn([qrow["query"]])[0]
        sims = corpus_embs @ q_emb
        if rerank_fn is not None:
            cand_indices = np.argsort(-sims)[:top_k_rerank].tolist()
            cand_texts = [unique_texts[i] for i in cand_indices]
            ce_scores = rerank_fn(qrow["query"], cand_texts)
            ranked = [cand_indices[i] for i in np.argsort(-ce_scores)]
        else:
            ranked = np.argsort(-sims)[:10].tolist()
        latencies.append((time.perf_counter_ns() - t0) / 1e6)

        top5 = ranked[:5]
        top10 = ranked[:10]
        r5.append(1.0 if pos_idx in top5 else 0.0)
        r10.append(1.0 if pos_idx in top10 else 0.0)
        if pos_idx in top10:
            rank = top10.index(pos_idx) + 1
            mrr10.append(1.0 / rank)
            ndcg10.append(1.0 / np.log2(rank + 1))
        else:
            mrr10.append(0.0)
            ndcg10.append(0.0)

    return {
        "queries": len(queries),
        "corpus_size": len(unique_texts),
        "recall_at_5": round(float(np.mean(r5)), 4),
        "recall_at_10": round(float(np.mean(r10)), 4),
        "mrr_at_10": round(float(np.mean(mrr10)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg10)), 4),
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
    }


def export_onnx_and_int8(model: nn.Module, tok_dir: Path, out_dir: Path, sample_enc: dict, extra_files: tuple[str, ...]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in extra_files:
        src = tok_dir / f
        if src.is_file():
            shutil.copy2(src, out_dir / f)
    fp32 = out_dir / "model.onnx"
    int8 = out_dir / "model_quantized.onnx"
    model.eval()
    torch.onnx.export(
        model,
        (sample_enc["input_ids"], sample_enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "output": {0: "batch"}},
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
    return {
        "fp32_bytes": fp32.stat().st_size,
        "fp32_sha256": sha256_file(fp32),
        "int8_bytes": int8.stat().st_size,
        "int8_sha256": sha256_file(int8),
    }


def train_domain_bge(train_rows: list[dict], val_rows: list[dict], val_corpus: list[str]) -> dict[str, Any]:
    print("\n=== Training Domain-Fine-Tuned BGE Embeddings (InfoNCE + Hard Negatives) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(BGE_BASE))
    enc_model = AutoModel.from_pretrained(str(BGE_BASE))
    model = BgeBiEncoderWrapper(enc_model)
    ds = BiEncoderPairDataset(train_rows)

    def collate(b):
        qs = [x[0] for x in b]
        ps = [x[1] for x in b]
        ns = [x[2] for x in b]
        eq = tok(qs, padding=True, truncation=True, max_length=80, return_tensors="pt")
        ep = tok(ps, padding=True, truncation=True, max_length=128, return_tensors="pt")
        en = tok(ns, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return eq, ep, en

    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(19))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 2
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.1 * len(loader) * epochs)), len(loader) * epochs)
    ckpt_path = OUT_BGE_FT / "checkpoint.pt"
    OUT_BGE_FT.mkdir(parents=True, exist_ok=True)
    best_mrr = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for step, (eq, ep, en) in enumerate(loader, start=1):
            opt.zero_grad()
            q_v = model(eq["input_ids"], eq["attention_mask"])
            p_v = model(ep["input_ids"], ep["attention_mask"])
            n_v = model(en["input_ids"], en["attention_mask"])
            # In-batch positives + explicit hard negatives scaled by temperature=20
            cand = torch.cat([p_v, n_v], dim=0)  # [2B, D]
            logits = (q_v @ cand.T) * 20.0
            targets = torch.arange(q_v.size(0), dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            if step % 15 == 0 or step == len(loader):
                print(f"    BGE-FT Ep {epoch} step {step}/{len(loader)} loss={round(tot_loss / step, 4)}", flush=True)

        model.eval()
        def enc_fn(txts: list[str]) -> np.ndarray:
            vecs = []
            with torch.no_grad():
                for i in range(0, len(txts), 32):
                    e = tok(txts[i:i + 32], padding=True, truncation=True, max_length=160, return_tensors="pt")
                    vecs.append(model(e["input_ids"], e["attention_mask"]).numpy())
            return np.concatenate(vecs, axis=0)

        val_m = compute_ranking_metrics(val_rows, val_corpus, enc_fn)
        ep_s = round(time.perf_counter() - t0, 1)
        history.append({"epoch": epoch, "loss": round(tot_loss / len(loader), 4), **val_m, "seconds": ep_s})
        print(f"  BGE-FT Epoch {epoch}/{epochs}: loss={history[-1]['loss']}, val_MRR@10={val_m['mrr_at_10']}, val_R@5={val_m['recall_at_5']}, time={ep_s}s", flush=True)
        if val_m["mrr_at_10"] > best_mrr:
            best_mrr = val_m["mrr_at_10"]
            best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_state": best_state, "best_mrr": best_mrr, "history": history}, ckpt_path)

    assert best_state is not None
    model.load_state_dict(best_state)
    sample = tok(["sample query", "sample passage"], padding=True, truncation=True, max_length=160, return_tensors="pt")
    art = export_onnx_and_int8(model, BGE_BASE, OUT_BGE_FT, sample, ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"))
    return {"history": history, "artifact": art}


def train_cross_encoder(
    name: str,
    base_dir: Path,
    out_dir: Path,
    use_mean_pool: bool,
    train_rows: list[dict],
    val_rows: list[dict],
    val_corpus: list[str],
    bi_encode_fn,
    epochs: int = 2,
    batch_size: int = 16,
    max_negs: int = 2,
    max_length: int = 128,
) -> dict[str, Any]:
    print(f"\n=== Training Cross-Encoder Reranker: {name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    enc_model = AutoModel.from_pretrained(str(base_dir), attn_implementation="eager") if "modernbert" in str(base_dir).lower() else AutoModel.from_pretrained(str(base_dir))
    model = CrossEncoderModel(enc_model, use_mean_pool=use_mean_pool)
    ds = CrossEncoderDataset(train_rows, max_negs=max_negs)

    def collate(b):
        e = tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        y = torch.tensor([x[2] for x in b], dtype=torch.float32)
        return e["input_ids"], e["attention_mask"], y

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(19))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.1 * len(loader) * epochs)), len(loader) * epochs)
    pos_weight = torch.tensor([3.0])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    best_mrr = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss = 0.0
        for step, (b_ids, b_mask, b_y) in enumerate(loader, start=1):
            opt.zero_grad()
            logits = model(b_ids, b_mask)
            loss = crit(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            if step % 20 == 0 or step == len(loader):
                print(f"    {name} Ep {epoch} step {step}/{len(loader)} loss={round(tot_loss / step, 4)}", flush=True)

        model.eval()
        def rr_fn(q: str, cands: list[str]) -> np.ndarray:
            with torch.no_grad():
                e = tok([q] * len(cands), cands, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                return model(e["input_ids"], e["attention_mask"]).numpy()

        val_m = compute_ranking_metrics(val_rows, val_corpus, bi_encode_fn, rerank_fn=rr_fn)
        ep_s = round(time.perf_counter() - t0, 1)
        history.append({"epoch": epoch, "loss": round(tot_loss / len(loader), 4), **val_m, "seconds": ep_s})
        print(f"  {name} Epoch {epoch}/{epochs}: loss={history[-1]['loss']}, val_MRR@10={val_m['mrr_at_10']}, val_R@5={val_m['recall_at_5']}, time={ep_s}s", flush=True)
        if val_m["mrr_at_10"] > best_mrr:
            best_mrr = val_m["mrr_at_10"]
            best_state = copy.deepcopy(model.state_dict())
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_state": best_state, "best_mrr": best_mrr, "history": history}, ckpt_path)

    assert best_state is not None
    model.load_state_dict(best_state)
    sample = tok(["query 1", "query 2"], ["passage 1", "passage 2"], padding=True, truncation=True, max_length=192, return_tensors="pt")
    art = export_onnx_and_int8(model, base_dir, out_dir, sample, ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"))
    return {"history": history, "artifact": art}


def main() -> None:
    torch.set_num_threads(4)
    set_seed(19)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(DATA_DIR / "retrieval_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "retrieval_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "retrieval_test.jsonl")

    val_corpus = [r["positive"] for r in val_rows] + [n["text"] for r in val_rows for n in r["hard_negatives"]]
    test_corpus = [r["positive"] for r in test_rows] + [n["text"] for r in test_rows for n in r["hard_negatives"]]

    # 1. Baseline: Frozen BGE INT8 (models/semantic)
    from stockbrief.semantic import SentenceEncoder
    with threadpool_limits(limits=1):
        frozen_bge = SentenceEncoder(BGE_PINNED_DIR)
        frozen_test = compute_ranking_metrics(test_rows, test_corpus, frozen_bge.encode)
        frozen_val = compute_ranking_metrics(val_rows, val_corpus, frozen_bge.encode)
    print(f"Frozen BGE INT8 Test: R@5={frozen_test['recall_at_5']}, R@10={frozen_test['recall_at_10']}, MRR@10={frozen_test['mrr_at_10']}, nDCG@10={frozen_test['ndcg_at_10']}", flush=True)

    # 2. Domain-fine-tuned BGE
    bge_ft_info = train_domain_bge(train_rows, val_rows, val_corpus)

    with threadpool_limits(limits=1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        tok_bge = AutoTokenizer.from_pretrained(str(OUT_BGE_FT))
        sess_bge_fp32 = ort.InferenceSession(str(OUT_BGE_FT / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        sess_bge_int8 = ort.InferenceSession(str(OUT_BGE_FT / "model_quantized.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

        def ort_bi_fn(sess):
            def _fn(txts: list[str]) -> np.ndarray:
                outs = []
                for i in range(0, len(txts), 32):
                    e = tok_bge(txts[i:i + 32], padding=True, truncation=True, max_length=160, return_tensors="np")
                    outs.append(sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0])
                return np.concatenate(outs, axis=0)
            return _fn

        bge_ft_fp32_test = compute_ranking_metrics(test_rows, test_corpus, ort_bi_fn(sess_bge_fp32))
        bge_ft_int8_test = compute_ranking_metrics(test_rows, test_corpus, ort_bi_fn(sess_bge_int8))
        bge_ft_int8_val_fn = ort_bi_fn(sess_bge_int8)

    print(f"Domain-FT BGE INT8 Test: R@5={bge_ft_int8_test['recall_at_5']}, R@10={bge_ft_int8_test['recall_at_10']}, MRR@10={bge_ft_int8_test['mrr_at_10']}, nDCG@10={bge_ft_int8_test['ndcg_at_10']}", flush=True)

    # 3. BGE Cross-Encoder Reranker (over Domain-FT BGE INT8 top-15)
    bge_ce_info = train_cross_encoder("BGE-CrossEncoder", BGE_BASE, OUT_BGE_RERANK, False, train_rows[:600], val_rows, val_corpus, bge_ft_int8_val_fn, epochs=2, batch_size=16, max_negs=1, max_length=128)

    # 4. ModernBERT Cross-Encoder Reranker (over Domain-FT BGE INT8 top-15; bounded CPU subset)
    mb_ce_info = train_cross_encoder("ModernBERT-CrossEncoder", MODERNBERT_BASE, OUT_MB_RERANK, True, train_rows[:360], val_rows, val_corpus, bge_ft_int8_val_fn, epochs=1, batch_size=16, max_negs=1, max_length=128)

    with threadpool_limits(limits=1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        def ort_ce_fn(mdir: Path):
            t_ce = AutoTokenizer.from_pretrained(str(mdir))
            s_fp32 = ort.InferenceSession(str(mdir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
            s_int8 = ort.InferenceSession(str(mdir / "model_quantized.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

            def mk(s):
                def _rr(q: str, cands: list[str]) -> np.ndarray:
                    e = t_ce([q] * len(cands), cands, padding=True, truncation=True, max_length=128, return_tensors="np")
                    return s.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
                return _rr
            return mk(s_fp32), mk(s_int8)

        bge_ce_fp32, bge_ce_int8 = ort_ce_fn(OUT_BGE_RERANK)
        mb_ce_fp32, mb_ce_int8 = ort_ce_fn(OUT_MB_RERANK)

        bge_ce_fp32_test = compute_ranking_metrics(test_rows, test_corpus, bge_ft_int8_val_fn, rerank_fn=bge_ce_fp32)
        bge_ce_int8_test = compute_ranking_metrics(test_rows, test_corpus, bge_ft_int8_val_fn, rerank_fn=bge_ce_int8)
        mb_ce_fp32_test = compute_ranking_metrics(test_rows, test_corpus, bge_ft_int8_val_fn, rerank_fn=mb_ce_fp32)
        mb_ce_int8_test = compute_ranking_metrics(test_rows, test_corpus, bge_ft_int8_val_fn, rerank_fn=mb_ce_int8)

    report = {
        "frozen_bge_int8_test": frozen_test,
        "domain_ft_bge": {**bge_ft_info, "test_fp32": bge_ft_fp32_test, "test_int8": bge_ft_int8_test},
        "bge_cross_encoder_reranker": {**bge_ce_info, "test_fp32": bge_ce_fp32_test, "test_int8": bge_ce_int8_test},
        "modernbert_cross_encoder_reranker": {**mb_ce_info, "test_fp32": mb_ce_fp32_test, "test_int8": mb_ce_int8_test},
    }
    (RESULTS_DIR / "retrieval_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Retrieval & Reranking benchmarks saved to phase2/results/retrieval_results.json", flush=True)


if __name__ == "__main__":
    main()
