"""FinAcumen data download helper.

Placeholder — update DATASET_URLS after review process is complete.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Placeholder URLs — replace with actual download links after review
# ---------------------------------------------------------------------------
DATASETS = ["finmme", "finmmr_easy", "finmmr_hard", "finmmr_medium", "fintmm", "bizbench"]

DATASET_URLS: dict[str, str] = {
    # "finmme":        "PLACEHOLDER_URL",
    # "finmmr_easy":   "PLACEHOLDER_URL",
    # "finmmr_hard":   "PLACEHOLDER_URL",
    # "finmmr_medium": "PLACEHOLDER_URL",
    # "fintmm":        "PLACEHOLDER_URL",
    # "bizbench":      "PLACEHOLDER_URL",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    zip_path.unlink()


def create_expected_structure() -> None:
    """Print the expected directory structure after download."""
    print("\nExpected directory structure after download:\n")
    for ds in DATASETS:
        print(f"  data/{ds}/")
        print(f"    train.json")
        print(f"    test.json")
    print()


def download_all() -> None:
    if not DATASET_URLS:
        print("No download URLs configured yet.")
        print("After the review process, uncomment and update DATASET_URLS in this script.")
        create_expected_structure()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, ds in enumerate(DATASETS, 1):
        url = DATASET_URLS.get(ds)
        if not url or url == "PLACEHOLDER_URL":
            print(f"[{i}/{len(DATASETS)}] {ds}: SKIP (no URL)")
            continue

        zip_path = DATA_DIR / f"{ds}.zip"
        ds_dir = DATA_DIR / ds

        if ds_dir.exists() and (ds_dir / "train.json").exists():
            print(f"[{i}/{len(DATASETS)}] {ds}: SKIP (already exists)")
            continue

        print(f"[{i}/{len(DATASETS)}] {ds}:")
        download_file(url, zip_path)
        extract_zip(zip_path, DATA_DIR)

    print("\nDone. Verify with:")
    print("  ls data/*/train.json data/*/test.json")


if __name__ == "__main__":
    download_all()
