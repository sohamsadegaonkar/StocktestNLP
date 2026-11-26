"""Download and verify checksums/licenses of FinQA, TAT-QA, FinRED, and Financial PhraseBank."""
from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path
import zipfile
import httpx

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "external"

SOURCES = [
    ("finqa/LICENSE", "https://raw.githubusercontent.com/czyssrs/FinQA/main/LICENSE", "MIT"),
    ("finqa/train.json", "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json", "MIT"),
    ("finqa/dev.json", "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json", "MIT"),
    ("finqa/test.json", "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json", "MIT"),
    ("tatqa/LICENSE", "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/LICENSE", "MIT"),
    ("tatqa/tatqa_dataset_train.json", "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json", "MIT"),
    ("tatqa/tatqa_dataset_dev.json", "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json", "MIT"),
    ("phrasebank/FinancialPhraseBank-v1.0.zip", "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip", "CC-BY-NC-SA-3.0"),
    ("finred/flare_finred_test.parquet", "https://huggingface.co/datasets/ChanceFocus/flare-finred/resolve/main/data/test-00000-of-00001-136bfc312f5271da.parquet", "Academic-Research-FinWeb22"),
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    provenance = []

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        # Also query FinGPT/fingpt-finred-re parquet siblings dynamically
        r_fg = client.get("https://huggingface.co/api/datasets/FinGPT/fingpt-finred-re")
        extra_sources = list(SOURCES)
        if r_fg.status_code == 200:
            for sib in r_fg.json().get("siblings", []):
                rf = sib.get("rfilename", "")
                if rf.endswith(".parquet"):
                    extra_sources.append(
                        (f"finred/{Path(rf).name}", f"https://huggingface.co/datasets/FinGPT/fingpt-finred-re/resolve/main/{rf}", "Academic-Research-FinWeb22")
                    )

        for rel, url, lic in extra_sources:
            dest = CACHE_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file() or dest.stat().st_size == 0:
                print(f"Downloading {rel} ({lic})...", flush=True)
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_bytes(65536):
                            f.write(chunk)
                    tmp.replace(dest)
            provenance.append({
                "file": rel,
                "url": url,
                "license": lic,
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            })

    # Unzip FinancialPhraseBank-v1.0.zip
    zip_path = CACHE_DIR / "phrasebank" / "FinancialPhraseBank-v1.0.zip"
    extract_dir = CACHE_DIR / "phrasebank" / "extracted"
    if zip_path.is_file() and not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    (CACHE_DIR / "external_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded/verified {len(provenance)} external dataset artifacts.")


if __name__ == "__main__":
    main()
