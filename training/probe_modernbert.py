"""Check ModernBERT-base download, CPU embedding speed, fine-tuning feasibility, and memory usage."""
from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil
import torch
from transformers import AutoModel, AutoTokenizer


def rss_mib() -> float:
    return round(psutil.Process().memory_info().rss / (1024**2), 1)


def main() -> None:
    torch.set_num_threads(4)
    model_id = "answerdotai/ModernBERT-base"
    print(f"Loading {model_id} (initial RSS={rss_mib()} MiB)...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s (RSS={rss_mib()} MiB, params={sum(p.numel() for p in model.parameters()):,})", flush=True)

    texts = [
        "Hyundai Motor India Limited will undertake a weighted average price increase of around 0.6% across its model range effective January 1, 2026.",
        "The Board of Directors approved the unaudited standalone and consolidated financial results for the quarter ended September 30, 2025.",
    ] * 4  # batch of 8
    enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")

    model.eval()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**enc).last_hidden_state[:, 0, :]
    print(f"Eval forward (batch=8, len={enc['input_ids'].shape[1]}): {(time.perf_counter() - t1)*1000:.1f} ms (RSS={rss_mib()} MiB)", flush=True)

    # Test 1 backward step to verify fine-tuning feasibility on 12GB RAM CPU
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    t2 = time.perf_counter()
    out = model(**enc).last_hidden_state[:, 0, :]
    loss = out.pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Train fwd+bwd+step (batch=8): {(time.perf_counter() - t2)*1000:.1f} ms (RSS={rss_mib()} MiB)", flush=True)


if __name__ == "__main__":
    main()
