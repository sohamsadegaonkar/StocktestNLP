"""Export PyTorch ModernBERT classifiers to ONNX FP32 and ONNX INT8 with parity verification."""
from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ModernBertClassifier(nn.Module):
    """Mean-pooled ModernBERT encoder with optional L2-normalization and linear classification head."""

    def __init__(self, encoder: nn.Module, num_classes: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.normalize:
            norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp(min=1e-9)
            pooled = pooled / norm
        return self.head(self.drop(pooled))


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def export_and_quantize(
    model: ModernBertClassifier,
    tokenizer_dir: Path,
    out_dir: Path,
    adapter_meta: dict[str, Any],
    sample_texts: list[str],
    max_length: int = 160,
) -> dict[str, Any]:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        src = tokenizer_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model.eval()
    enc = tok(sample_texts[:2], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model_quantized.onnx"

    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        prep_path = Path(tmpdir) / "model_prep.onnx"
        quant_pre_process(str(fp32_path), str(prep_path))
        quantize_dynamic(
            str(prep_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )

    probe_enc = tok(sample_texts[:4], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        pt_logits = model(probe_enc["input_ids"], probe_enc["attention_mask"]).numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_fp32 = ort.InferenceSession(str(fp32_path), sess_options=opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), sess_options=opts, providers=["CPUExecutionProvider"])

    feed = {
        "input_ids": probe_enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": probe_enc["attention_mask"].numpy().astype(np.int64),
    }
    onnx_fp32_logits = sess_fp32.run(None, feed)[0]
    onnx_int8_logits = sess_int8.run(None, feed)[0]

    pt_vs_fp32_diff = float(np.max(np.abs(pt_logits - onnx_fp32_logits)))
    fp32_vs_int8_diff = float(np.max(np.abs(onnx_fp32_logits - onnx_int8_logits)))

    adapter_config = {
        **adapter_meta,
        "max_length": max_length,
        "pad_token": tok.pad_token or "[PAD]",
        "fp32_sha256": sha256_file(fp32_path),
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_sha256": sha256_file(int8_path),
        "int8_bytes": int8_path.stat().st_size,
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "fp32_bytes": adapter_config["fp32_bytes"],
        "fp32_sha256": adapter_config["fp32_sha256"],
        "int8_bytes": adapter_config["int8_bytes"],
        "int8_sha256": adapter_config["int8_sha256"],
        "pt_vs_fp32_max_abs_diff": round(pt_vs_fp32_diff, 6),
        "fp32_vs_int8_probe_max_abs_diff": round(fp32_vs_int8_diff, 6),
    }
