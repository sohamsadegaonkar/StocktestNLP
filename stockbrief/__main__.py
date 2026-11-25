import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="StockBrief Edge disclosure workspace")
    parser.add_argument("command", choices=("serve", "evaluate", "fetch-model"), nargs="?", default="serve")
    parser.add_argument("--port", type=int, help="Server port (default 5050 locally, PORT in hosted mode)")
    parser.add_argument("--db", type=Path, help="SQLite database path")
    parser.add_argument("--hosted", action="store_true", help="Serve private browser workspaces on a configured HTTPS origin")
    parser.add_argument("--cpus", type=int, help="Limit Linux CPU affinity to this many available CPUs")
    parser.add_argument("--transformer", type=Path, help="Path to local transformer files for evaluation or download")
    args = parser.parse_args()
    if args.cpus:
        if args.cpus < 1 or not hasattr(os, "sched_getaffinity"):
            parser.error("--cpus requires a positive count and Linux affinity support")
        available = sorted(os.sched_getaffinity(0))
        if args.cpus > len(available):
            parser.error(f"only {len(available)} CPUs are available to this process")
        os.sched_setaffinity(0, set(available[:args.cpus]))
    elif args.cpus == 0:
        parser.error("--cpus must be positive")

    try:
        if args.command == "fetch-model":
            from .transformer import fetch_model

            folder = args.transformer or Path("models/mobilebert")
            print(f"Saved MobileBERT revision {fetch_model(folder)} in {folder}")
            return

        from threadpoolctl import threadpool_limits

        from .transformer import Transformer

        with threadpool_limits(limits=1):
            comparator = Transformer(args.transformer) if args.transformer else None
            if args.command == "serve":
                import uvicorn

                from .app import create_app

                origin = None
                database_url = None
                port = args.port
                if args.hosted:
                    origin = os.environ.get("STOCKBRIEF_PUBLIC_ORIGIN") or os.environ.get("RENDER_EXTERNAL_URL")
                    if not origin:
                        raise ValueError("Set STOCKBRIEF_PUBLIC_ORIGIN to the deployed HTTPS website URL.")
                    database_url = os.environ.get("STOCKBRIEF_DATABASE_URL")
                    if port is None:
                        port = int(os.environ.get("PORT", "10000"))
                uvicorn.run(
                    create_app(args.db, transformer=comparator, public_origin=origin, database_url=database_url),
                    host="0.0.0.0" if args.hosted else "127.0.0.1",
                    port=port if port is not None else 5050,
                )
            else:
                from .evaluation import evaluate
                from .nlp import EventModel

                print(json.dumps(evaluate(EventModel(), comparator), ensure_ascii=False, indent=2))
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        print(f"StockBrief Edge: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
