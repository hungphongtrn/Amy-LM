#!/usr/bin/env python3
"""One-time setup: download all HF assets for Amy-LM training.

Usage:
    python scripts/setup.py                # download everything
    python scripts/setup.py --check-only   # report what's missing

Downloads:
    hungphongtrn/facodec-checkpoints  → checkpoints/facodec/
    hungphongtrn/mustard-facodec      → data/processed/mustard-processed/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import get_token

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS = {
    "facodec": {
        "repo": "hungphongtrn/facodec-checkpoints",
        "local": PROJECT_ROOT / "checkpoints" / "facodec",
        "type": "model",
        "marker": "ns3_facodec_decoder.bin",
        "desc": "FACodec encoder + decoder (398MB)",
    },
    "mustard": {
        "repo": "hungphongtrn/mustard-facodec",
        "local": PROJECT_ROOT / "data" / "processed" / "mustard-processed",
        "type": "dataset",
        "marker": "train.parquet",
        "desc": "MUStARD preprocessed (304MB)",
    },
}


def check_assets() -> dict[str, bool]:
    return {name: (a["local"] / a["marker"]).exists() for name, a in ASSETS.items()}


def download_assets() -> None:
    token = get_token()
    if not token:
        # Try env as fallback
        env_token = os.getenv("HF_TOKEN", "")
        if env_token:
            login(token=env_token)
            token = get_token()

    if token:
        print(f"HF authenticated")
    else:
        print("WARNING: No HF token found. Run 'hf auth login' or set HF_TOKEN.")

    for name, asset in ASSETS.items():
        local = asset["local"]
        marker = local / asset["marker"]
        if marker.exists():
            print(f"  [{name}] already present → {local}")
            continue
        print(f"  [{name}] downloading {asset['desc']} ...")
        local.mkdir(parents=True, exist_ok=True)
        snapshot_download(asset["repo"], local_dir=str(local), repo_type=asset["type"])
        print(f"  [{name}] done → {local}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HF assets for Amy-LM training")
    p.add_argument("--check-only", action="store_true", help="Only report what's missing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    status = check_assets()
    all_ok = True
    for name, asset in ASSETS.items():
        ok = status[name]
        if not ok:
            all_ok = False
        print(f"  [{name}] {'OK' if ok else 'MISSING'}  {asset['local']}")

    if args.check_only or all_ok:
        return 0 if all_ok else 1

    print()
    download_assets()
    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
