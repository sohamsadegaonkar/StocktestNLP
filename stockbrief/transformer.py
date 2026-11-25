"""Optional ONNX MobileBERT MNLI comparator. Inference uses local files only."""
import json
from pathlib import Path

import numpy as np

from .corpus import LABELS
from .nlp import chunks_for

REPOSITORY = "Xenova/mobilebert-uncased-mnli"
FILES = ("config.json", "tokenizer.json", "onnx/model_quantized.onnx")
HYPOTHESES = {
    "earnings": "financial results, earnings, or earnings guidance",
    "corporate_action": "a dividend, stock split, repurchase, rights issue, or bonus shares",
    "deal": "an acquisition, merger, demerger, stake sale, or joint venture",
    "governance": "appointment, departure, or retirement of company leadership or auditors",
    "regulatory": "regulatory action, a fine, a court order, or an enforcement proceeding",
    "operations": "a supply contract, plant operations, or a production disruption",
    "routine": "a routine meeting notice, voting instructions, or filing logistics",
}


def fetch_model(folder: Path):
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(REPOSITORY).sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable model revision.")
    snapshot_download(repo_id=REPOSITORY, revision=revision, local_dir=folder, allow_patterns=list(FILES))
    (folder / "stockbrief-revision.txt").write_text(revision + "\n", encoding="utf-8")
    return revision


class Transformer:
    name = "MobileBERT MNLI / ONNX INT8"

    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        missing = [name for name in FILES if not (folder / name).is_file()]
        if missing:
            raise ValueError(f"Model files are missing: {', '.join(missing)}")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        labels = {str(label).lower(): int(index) for index, label in config["id2label"].items()}
        self.entailment = labels["entailment"]
        self.contradiction = labels["contradiction"]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=384, strategy="only_first")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / "onnx/model_quantized.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.inputs = [field.name for field in self.session.get_inputs()]

    def classify(self, document):
        chunks = chunks_for(document.pages)[:6]
        if not chunks:
            return "review"
        candidates = list(LABELS)
        pairs = [
            (chunk.text, f"This disclosure is about {HYPOTHESES[label]}.")
            for chunk in chunks
            for label in candidates
        ]
        encoded = self.tokenizer.encode_batch(pairs)
        fields = {
            "input_ids": [item.ids for item in encoded],
            "attention_mask": [item.attention_mask for item in encoded],
            "token_type_ids": [item.type_ids for item in encoded],
        }
        logits = self.session.run(None, {name: np.asarray(fields[name], dtype=np.int64) for name in self.inputs})[0]
        # MNLI entailment vs. contradiction for every candidate. Neutral is deliberately excluded.
        difference = np.clip(logits[:, self.entailment] - logits[:, self.contradiction], -50, 50)
        scores = (1 / (1 + np.exp(-difference))).reshape(len(chunks), len(candidates)).max(axis=0)
        high, second = np.argsort(scores)[::-1][:2]
        if scores[high] < 0.6 or scores[high] - scores[second] < 0.08:
            return "review"
        return candidates[high]
