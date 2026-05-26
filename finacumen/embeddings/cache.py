"""Query embedding cache — avoid repeated API calls for known targets.

Cache files are per-dataset JSON files stored in datasets/{split}/{dataset}/.
Maps target_id → 1024-d float32 list.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from finacumen.ft.paths import REPO_ROOT

logger = logging.getLogger(__name__)

DATASETS_ROOT = REPO_ROOT / "datasets"
CACHE_VERSION = 1


def _cache_path(dataset: str, split: str) -> Path:
    cache_dir = DATASETS_ROOT / split / dataset / "query_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "embeddings.json"


def load_query_cache(dataset: str, split: str) -> dict[str, np.ndarray]:
    path = _cache_path(dataset, split)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != CACHE_VERSION:
        logger.warning("Cache version mismatch for %s/%s, ignoring", dataset, split)
        return {}
    result: dict[str, np.ndarray] = {}
    for tid, emb_list in data.get("embeddings", {}).items():
        result[tid] = np.array(emb_list, dtype=np.float32)
    return result


def save_query_cache(
    dataset: str,
    split: str,
    embeddings: dict[str, np.ndarray],
    provider: str,
    model: str,
    dim: int,
) -> None:
    from datetime import datetime, timezone

    emb_dict = {tid: vec.tolist() for tid, vec in embeddings.items()}
    data = {
        "version": CACHE_VERSION,
        "dataset": dataset,
        "split": split,
        "provider": provider,
        "model": model,
        "dim": dim,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "embeddings": emb_dict,
        "total": len(emb_dict),
    }
    path = _cache_path(dataset, split)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_cached_query_embedding(
    target_id: str, dataset: str, split: str
) -> np.ndarray | None:
    cache = load_query_cache(dataset, split)
    return cache.get(target_id)
