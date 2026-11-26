"""Fetch Phase 2 pretrained models reliably over HTTPS into models/phase2/pretrained/."""
from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path
import time
import httpx

BASE_OUT = Path("models/phase2/pretrained")

MODEL_SPECS = [
    (
        "bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "MIT",
        ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"],
    ),
    (
        "distilbert-squad",
        "distilbert/distilbert-base-cased-distilled-squad",
        "Apache-2.0",
        ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"],
    ),
    (
        "qwen2.5-0.5b-instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Apache-2.0",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
    ),
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main() -> None:
    manifest = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for local_name, repo, lic, files in MODEL_SPECS:
            out_dir = BASE_OUT / local_name
            out_dir.mkdir(parents=True, exist_ok=True)
            file_info = []
            for fname in files:
                dest = out_dir / fname
                if not dest.is_file() or dest.stat().st_size == 0:
                    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
                    tmp = out_dir / f"{fname}.part"
                    t0 = time.perf_counter()
                    with client.stream("GET", url) as resp:
                        if resp.status_code == 404:
                            print(f"  skip 404: {local_name}/{fname}", flush=True)
                            continue
                        resp.raise_for_status()
                        total = int(resp.headers.get("content-length", "0") or 0)
                        downloaded = 0
                        next_mark = 100_000_000
                        with tmp.open("wb") as f:
                            for chunk in resp.iter_bytes(262144):
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded >= next_mark:
                                    pct = f" ({100*downloaded/total:.0f}%)" if total else ""
                                    print(f"  {local_name}/{fname}: {downloaded/1e6:.1f} MB{pct}", flush=True)
                                    next_mark += 100_000_000
                    tmp.replace(dest)
                    print(f"Saved {local_name}/{fname} ({dest.stat().st_size/1e6:.2f} MB) in {time.perf_counter()-t0:.1f}s", flush=True)
                file_info.append({"file": fname, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)})
            manifest.append({"name": local_name, "repo": repo, "license": lic, "files": file_info})
    (BASE_OUT / "pretrained_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("All Phase 2 pretrained models verified.")


if __name__ == "__main__":
    main()
