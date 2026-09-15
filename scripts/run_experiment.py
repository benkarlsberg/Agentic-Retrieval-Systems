#!/usr/bin/env python3
"""CLI to run ARS experiments from YAML configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ars.experiment import run_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic Retrieval Systems experiment")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory under results/runs/",
    )
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    meta = run_from_config(config_path, output_dir=args.output_dir)
    print(json.dumps(meta, indent=2))
    print(f"\n[offline-fixture] Results written to: {meta['output_dir']}")


if __name__ == "__main__":
    main()
