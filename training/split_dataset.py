"""Deduplicate by text/similarity, group by event_family, hold out unseen issuers, and split chronologically."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from stockbrief.corpus import CALIBRATION as LEGACY_CAL, TEST as LEGACY_TEST, TRAIN as LEGACY_TRAIN
from stockbrief.fixtures import CASE_STUDY
from stockbrief.semantic_data import SEMANTIC_ADDITIONAL_TEST, SEMANTIC_CALIBRATION, SEMANTIC_TRAIN
from training.schema import VALID_LABELS, load_jsonl, normalize_text, save_jsonl, text_fingerprint

DATA_DIR = Path(__file__).resolve().parent / "data"
CURATED_PATH = DATA_DIR / "nse_announcements.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
CAL_PATH = DATA_DIR / "calibration.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"


def word_shingles(text: str, k: int = 4) -> set[str]:
    tokens = re.findall(r"[a-z0-9₹%]+", normalize_text(text))
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(rows: list[dict[str, Any]], sim_threshold: float = 0.85) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    kept_shingles: list[tuple[str, set[str]]] = []
    exact_dropped = 0
    near_dropped = 0

    # Ensure deterministic order: sort by date ascending, then doc_id
    ordered = sorted(rows, key=lambda r: (r["date"], r["doc_id"]))
    for row in ordered:
        text = row["text"]
        if row["source_mode"] == "pdf_extraction_failed":
            # Deduplicate empty/garbled extractions by issuer+date+doc_id rather than empty text
            key = f"failed:{row['issuer_symbol']}:{row['date']}"
            if key in seen_exact:
                exact_dropped += 1
                continue
            seen_exact.add(key)
            kept.append(row)
            continue

        fp = text_fingerprint(text)
        if fp in seen_exact:
            exact_dropped += 1
            continue

        sh = word_shingles(text)
        is_near_dup = False
        if len(sh) >= 8:
            for prev_lbl, prev_sh in kept_shingles:
                if prev_lbl == row["label"] and jaccard(sh, prev_sh) >= sim_threshold:
                    is_near_dup = True
                    break
        if is_near_dup:
            near_dropped += 1
            continue

        seen_exact.add(fp)
        kept_shingles.append((row["label"], sh))
        kept.append(row)

    stats = {
        "input_count": len(rows),
        "deduped_count": len(kept),
        "exact_duplicates_removed": exact_dropped,
        "near_duplicates_removed": near_dropped,
    }
    return kept, stats


def repo_benchmark_fingerprints() -> set[str]:
    fps: set[str] = set()
    for split_dict in (LEGACY_TRAIN, LEGACY_CAL, LEGACY_TEST, SEMANTIC_TRAIN, SEMANTIC_CALIBRATION, SEMANTIC_ADDITIONAL_TEST):
        for texts in split_dict.values():
            for t in texts:
                fps.add(text_fingerprint(t))
    for doc, _, _ in CASE_STUDY:
        fps.add(text_fingerprint(" ".join(doc.pages)))
    return fps


def build_splits(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    # Group by event_family
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        families[r["event_family"]].append(r)

    # Select held-out issuers (completely unseen in train and calibration, spanning all labels + mixed + special slices)
    issuer_families: dict[str, list[str]] = defaultdict(list)
    for fam, docs in families.items():
        issuer_families[docs[0]["issuer_symbol"]].append(fam)

    # Pick issuers with 1-3 families so holding them out gives diverse test coverage across all labels
    heldout_issuers: set[str] = set()
    # Explicitly hold out 2 mixed pricing+earnings issuers and key single-event issuers across classes
    preferred_heldout = ["ACLGATI", "ALKEM", "M&M", "INDHOTEL", "PHOENIXLTD", "DMART", "DRREDDY", "TITAN", "ABB", "BEL", "TECHM", "EICHERMOT", "TVSMOTOR", "IRCTC", "VOLTAS", "DIXON"]
    for sym in preferred_heldout:
        if sym in issuer_families:
            heldout_issuers.add(sym)

    # Ensure every label gets some held-out issuer presence if needed, capping held-out share around ~12-15% of corpus
    heldout_count = sum(len(families[f]) for sym in heldout_issuers for f in issuer_families[sym])
    for sym, fams in sorted(issuer_families.items()):
        if heldout_count >= 55:
            break
        if sym in heldout_issuers or len(fams) > 2:
            continue
        # Only add if it doesn't exhaust a rare slice
        heldout_issuers.add(sym)
        heldout_count += len(fams)

    heldout_test: list[dict[str, Any]] = []
    chrono_families: list[tuple[str, str, list[dict[str, Any]]]] = []
    for fam, docs in families.items():
        sym = docs[0]["issuer_symbol"]
        fam_date = max(d["date"] for d in docs)
        if sym in heldout_issuers:
            heldout_test.extend(docs)
        else:
            chrono_families.append((fam_date, fam, docs))

    # Chronological split within each primary label across non-heldout families:
    # earliest ~64% -> train, middle ~18% -> calibration, latest ~18% -> test (combined with heldout_test)
    train_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = list(heldout_test)

    by_stratum: dict[tuple[str, bool, str], list[tuple[str, str, list[dict[str, Any]]]]] = defaultdict(list)
    for fam_date, fam, docs in chrono_families:
        d0 = docs[0]
        is_multi = len(d0["labels"]) > 1
        sub = d0["source_mode"] if d0["label"] == "review" else ("mixed" if is_multi else "single")
        by_stratum[(d0["label"], is_multi, sub)].append((fam_date, fam, docs))

    for (_, _, _), items in sorted(by_stratum.items()):
        items.sort(key=lambda x: (x[0], x[1]))
        n = len(items)
        if n <= 2:
            for _, _, docs in items:
                train_rows.extend(docs)
            continue
        if n <= 5:
            n_cal = 1
            n_chrono_test = 1
        else:
            n_cal = max(1, round(n * 0.22))
            n_chrono_test = max(1, round(n * 0.14))
        n_train = n - n_cal - n_chrono_test
        for _, _, docs in items[:n_train]:
            train_rows.extend(docs)
        for _, _, docs in items[n_train:n_train + n_cal]:
            cal_rows.extend(docs)
        for _, _, docs in items[n_train + n_cal:]:
            test_rows.extend(docs)

    # Assign split tag & sort deterministically
    for r in train_rows:
        r["split"] = "train"
    for r in cal_rows:
        r["split"] = "calibration"
    for r in test_rows:
        r["split"] = "test"

    train_rows.sort(key=lambda r: (r["date"], r["doc_id"]))
    cal_rows.sort(key=lambda r: (r["date"], r["doc_id"]))
    test_rows.sort(key=lambda r: (r["date"], r["doc_id"]))

    # Verify strict leakage invariants
    repo_fps = repo_benchmark_fingerprints()
    for split_name, split_rows in (("train", train_rows), ("calibration", cal_rows), ("test", test_rows)):
        for r in split_rows:
            if r["source_mode"] != "pdf_extraction_failed":
                assert text_fingerprint(r["text"]) not in repo_fps, f"Overlap with repo benchmark in {split_name}: {r['doc_id']}"

    train_fams = {r["event_family"] for r in train_rows}
    cal_fams = {r["event_family"] for r in cal_rows}
    test_fams = {r["event_family"] for r in test_rows}
    assert not (train_fams & cal_fams), "event_family leaked between train and calibration"
    assert not (train_fams & test_fams), "event_family leaked between train and test"
    assert not (cal_fams & test_fams), "event_family leaked between calibration and test"

    train_issuers = {r["issuer_symbol"] for r in train_rows}
    cal_issuers = {r["issuer_symbol"] for r in cal_rows}
    test_issuers = {r["issuer_symbol"] for r in test_rows}
    unseen_test_issuers = sorted(test_issuers - train_issuers - cal_issuers)
    assert len(unseen_test_issuers) >= 10, f"Expected >=10 held-out unseen issuers, got {len(unseen_test_issuers)}"

    def summarize(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(split_rows),
            "issuers": len({r["issuer_symbol"] for r in split_rows}),
            "date_min": min(r["date"] for r in split_rows),
            "date_max": max(r["date"] for r in split_rows),
            "labels": dict(Counter(r["label"] for r in split_rows)),
            "multilabel": sum(1 for r in split_rows if len(r["labels"]) > 1),
            "mixed_pricing_earnings": sum(1 for r in split_rows if set(r["labels"]) >= {"pricing", "earnings"}),
            "rumor_denials": sum(1 for r in split_rows if r["insufficiency_type"] == "rumor_denial"),
            "scheduled_stage": sum(1 for r in split_rows if r["stage"] == "Scheduled"),
            "failed_pdf": sum(1 for r in split_rows if r["source_mode"] == "pdf_extraction_failed"),
            "headline_only": sum(1 for r in split_rows if r["source_mode"] == "headline_only"),
        }

    meta = {
        "unseen_heldout_issuers": unseen_test_issuers,
        "unseen_heldout_docs": sum(1 for r in test_rows if r["issuer_symbol"] in set(unseen_test_issuers)),
        "splits": {
            "train": summarize(train_rows),
            "calibration": summarize(cal_rows),
            "test": summarize(test_rows),
        },
    }
    return train_rows, cal_rows, test_rows, meta


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = load_jsonl(CURATED_PATH)
    deduped, dedup_stats = deduplicate(raw)
    train_rows, cal_rows, test_rows, split_meta = build_splits(deduped)

    save_jsonl(TRAIN_PATH, train_rows)
    save_jsonl(CAL_PATH, cal_rows)
    save_jsonl(TEST_PATH, test_rows)

    manifest = {
        "dataset_version": "nse-domain-events-v1",
        "source": "Official NSE India Corporate Announcements (https://www.nseindia.com/companies-listing/corporate-filings-announcements)",
        "license_notes": (
            "Public exchange regulatory disclosures filed under SEBI (LODR) Regulations, 2015. "
            "Dataset stores official metadata (doc_id, URL, issuer, ISO date) with permitted short factual excerpts, "
            "headline-only notices, failed PDF extraction samples, and clearly labeled original annotation paraphrases. "
            "No user uploads are included. Repo benchmark fixtures (corpus.py, semantic_data.py, fixtures.py, test_semantic.py) remain untouched and excluded."
        ),
        "deduplication": dedup_stats,
        **split_meta,
        "files": {
            "curated": {"path": str(CURATED_PATH.relative_to(DATA_DIR.parent.parent)), "sha256": sha256_file(CURATED_PATH)},
            "train": {"path": str(TRAIN_PATH.relative_to(DATA_DIR.parent.parent)), "sha256": sha256_file(TRAIN_PATH)},
            "calibration": {"path": str(CAL_PATH.relative_to(DATA_DIR.parent.parent)), "sha256": sha256_file(CAL_PATH)},
            "test": {"path": str(TEST_PATH.relative_to(DATA_DIR.parent.parent)), "sha256": sha256_file(TEST_PATH)},
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
