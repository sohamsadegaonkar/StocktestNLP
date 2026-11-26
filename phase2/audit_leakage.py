"""Verify strict zero-leakage and inspect slice/class counts across all Phase 2 datasets."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from phase2.schema import fingerprint, load_jsonl

DATA_DIR = Path("phase2/data")


def check_split_leakage(name: str, tr: list[dict], va: list[dict], te: list[dict], key_fn) -> dict:
    tr_k = {key_fn(r) for r in tr}
    va_k = {key_fn(r) for r in va}
    te_k = {key_fn(r) for r in te}
    return {
        "task": name,
        "train": len(tr),
        "val": len(va),
        "test": len(te),
        "train_val_overlap": len(tr_k & va_k),
        "train_test_overlap": len(tr_k & te_k),
        "val_test_overlap": len(va_k & te_k),
    }


def main() -> None:
    report = {"leakage_checks": []}
    for task, fn in (
        ("ner", lambda r: fingerprint(r["text"])),
        ("rel", lambda r: fingerprint(r["text"])),
        ("retrieval", lambda r: fingerprint(r["positive"])),
        ("qa", lambda r: f"{r['source']}:{fingerprint(r['question'] + '|' + r['context'][:120])}"),
        ("impact", lambda r: fingerprint(r["text"]) if r.get("text") else r["id"]),
    ):
        tr = load_jsonl(DATA_DIR / f"{task}_train.jsonl")
        va = load_jsonl(DATA_DIR / f"{task}_val.jsonl")
        te = load_jsonl(DATA_DIR / f"{task}_test.jsonl")
        chk = check_split_leakage(task, tr, va, te, fn)
        report["leakage_checks"].append(chk)
        assert chk["train_val_overlap"] == 0, f"{task} train/val leak!"
        assert chk["train_test_overlap"] == 0, f"{task} train/test leak!"
        assert chk["val_test_overlap"] == 0, f"{task} val/test leak!"

    bench = load_jsonl(DATA_DIR / "benchmark_frozen.jsonl")
    report["benchmark_total"] = len(bench)
    report["benchmark_slices"] = dict(Counter(r["slice"] for r in bench))

    ner_tr = load_jsonl(DATA_DIR / "ner_train.jsonl")
    ent_counts = Counter(e["label"] for r in ner_tr for e in r["entities"])
    report["ner_train_entity_counts"] = dict(ent_counts)

    rel_tr = load_jsonl(DATA_DIR / "rel_train.jsonl")
    rel_te = load_jsonl(DATA_DIR / "rel_test.jsonl")
    report["rel_train_counts"] = dict(Counter(r["relation"] for r in rel_tr))
    report["rel_test_counts"] = dict(Counter(r["relation"] for r in rel_te))

    (DATA_DIR / "leakage_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
