"""Optional ONNX MobileBERT MNLI comparator. Inference uses local files only."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from .corpus import LABELS
from .nlp import chunks_for

REPOSITORY = "Xenova/mobilebert-uncased-mnli"
REVISION = "8b0ea66ab7b190bba77418ba03b67d69cfc9a1ee"
WEIGHTS = "onnx/model_quantized.onnx"
WEIGHTS_SHA256 = "1dc36bc1f41b1d1da9e8f28f69f8dc6fa316410b2cde057756ddfb7083a48021"
FILES = ("config.json", "tokenizer.json", WEIGHTS)
HYPOTHESES = {
    "earnings": "financial results, earnings, or earnings guidance",
    "corporate_action": "a dividend, stock split, repurchase, rights issue, or bonus shares",
    "deal": "an acquisition, merger, demerger, stake sale, or joint venture",
    "governance": "appointment, departure, or retirement of company leadership or auditors",
    "regulatory": "regulatory action, a fine, a court order, or an enforcement proceeding",
    "operations": "a supply contract, plant operations, or a production disruption",
    "routine": "a routine meeting notice, voting instructions, or filing logistics",
}


def _verified_weights(folder: Path) -> Path:
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Model files are missing: {', '.join(missing)}")
    weights = folder / WEIGHTS
    with weights.open("rb") as handle:
        actual = file_digest(handle, "sha256").hexdigest()
    if actual != WEIGHTS_SHA256:
        raise ValueError("MobileBERT weights failed SHA-256 verification. Fetch the pinned model into a fresh folder.")
    return weights


def fetch_model(folder: Path) -> str:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPOSITORY, revision=REVISION, local_dir=folder, allow_patterns=list(FILES))
    _verified_weights(folder)
    (folder / "stockbrief-revision.txt").write_text(REVISION + "\n", encoding="utf-8")
    return REVISION


class Transformer:
    name = "MobileBERT MNLI / ONNX INT8"

    def __init__(self, folder: Path):
        weights = _verified_weights(folder)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        try:
            labels = {str(label).lower(): int(index) for index, label in config["id2label"].items()}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("MobileBERT config must define MNLI label indices.") from exc
        if set(labels) != {"entailment", "neutral", "contradiction"} or set(labels.values()) != {0, 1, 2}:
            raise ValueError("MobileBERT config must define distinct entailment, neutral, and contradiction indices.")
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
            str(weights), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.inputs = [field.name for field in self.session.get_inputs()]
        self.artifact = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "weights_bytes": weights.stat().st_size,
            "onnxruntime_version": ort.__version__,
            "execution_providers": self.session.get_providers(),
        }

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
