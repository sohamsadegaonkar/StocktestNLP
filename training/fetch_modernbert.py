"""Download answerdotai/ModernBERT-base files reliably over HTTPS with progress logging."""
from __future__ import annotations

from pathlib import Path
import time
import httpx

REPO = "answerdotai/ModernBERT-base"
OUT_DIR = Path("models/modernbert-base-pretrained")
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for fname in FILES:
            dest = OUT_DIR / fname
            if dest.is_file() and dest.stat().st_size > 0:
                print(f"Cached {fname} ({dest.stat().st_size / 1e6:.2f} MB)", flush=True)
                continue
            url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
            tmp = OUT_DIR / f"{fname}.part"
            t0 = time.perf_counter()
            downloaded = 0
            next_mark = 50_000_000
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", "0") or 0)
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes(262144):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_mark:
                            pct = f" ({100*downloaded/total:.0f}%)" if total else ""
                            print(f"  {fname}: {downloaded/1e6:.1f} MB{pct} in {time.perf_counter()-t0:.1f}s", flush=True)
                            next_mark += 50_000_000
            tmp.replace(dest)
            print(f"Saved {fname} ({dest.stat().st_size / 1e6:.2f} MB) in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
