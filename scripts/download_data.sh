#!/usr/bin/env bash
# FinAcumen data download helper
# Placeholder — update URLs after review process is complete.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

echo "================================="
echo "FinAcumen Dataset Download"
echo "================================="
echo ""
echo "Data will be downloaded to: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

# ---------------------------------------------------------------------------
# Placeholder URLs — replace with actual download links after review
# ---------------------------------------------------------------------------
DATASETS=(
  "finmme"
  "finmmr_easy"
  "finmmr_hard"
  "finmmr_medium"
  "fintmm"
  "bizbench"
)

# Uncomment and update URLs when available:
#
# echo "[1/6] Downloading finmme..."
# curl -L -o "$DATA_DIR/finmme.zip" "PLACEHOLDER_URL"
# echo "[2/6] Downloading finmmr_easy..."
# curl -L -o "$DATA_DIR/finmmr_easy.zip" "PLACEHOLDER_URL"
# ...
# echo "Extracting..."
# for f in "$DATA_DIR"/*.zip; do
#   unzip -o "$f" -d "$DATA_DIR"
#   rm "$f"
# done

echo ""
echo "Dataset URLs are not yet available."
echo "After the review process, update this script with the download links."
echo ""
echo "Expected directory structure after download:"
echo ""
for ds in "${DATASETS[@]}"; do
  echo "  data/$ds/"
  echo "    train.json"
  echo "    test.json"
done
echo ""
echo "Run: python scripts/download_data.py  (Python alternative)"
