from __future__ import annotations

import argparse
import json
import sys

from .catalog_sources import fetch_openrouter_catalog, save_openrouter_catalog, simplify_openrouter_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="api-model-forensics-catalog",
        description="Fetch and save external model catalogs for local research.",
    )
    parser.add_argument(
        "--source",
        choices=("openrouter",),
        default="openrouter",
        help="Catalog source to fetch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many rows to print to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.source != "openrouter":
        parser.error("Only openrouter is supported right now.")

    payload = fetch_openrouter_catalog()
    rows = simplify_openrouter_catalog(payload)
    paths = save_openrouter_catalog(payload, rows)

    output = {
        "source": args.source,
        "model_count": payload["model_count"],
        "artifacts": paths,
        "sample_rows": rows[: max(0, args.limit)],
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
