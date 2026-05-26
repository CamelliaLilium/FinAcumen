"""Dataset Embedding Manager — load pre-computed embeddings from datasets/ dir.

Singleton pattern. Resolves bank entries to numpy vectors via target_id lookup.
Train split is checked first (bank entries are always train).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_emb_manager: DatasetEmbeddingManager | None = None


class DatasetEmbeddingManager:
    """Loads all pre-computed *_emb.npy + *_ids.json pairs into memory.

    Maintains separate train/test dicts because target_ids can overlap
    between splits (e.g., bizbench-0 appears in both).
    """

    def __init__(self, datasets_root: Path) -> None:
        self._train: dict[str, np.ndarray] = {}
        self._test: dict[str, np.ndarray] = {}
        for split_name in ("train", "test"):
            split_dir = datasets_root / split_name
            if not split_dir.is_dir():
                continue
            target = self._train if split_name == "train" else self._test
            for emb_path in sorted(split_dir.glob("*_emb.npy")):
                dataset = emb_path.stem.replace("_emb", "")
                ids_path = split_dir / f"{dataset}_ids.json"
                if not ids_path.exists():
                    continue
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                matrix = np.load(emb_path)
                if matrix.ndim != 2:
                    continue
                for i, tid in enumerate(ids):
                    if i < matrix.shape[0]:
                        target[tid] = matrix[i].astype(np.float32, copy=False)

    def resolve(self, target_id: str) -> np.ndarray | None:
        """Look up embedding by target_id. Train is checked first."""
        v = self._train.get(target_id)
        if v is not None:
            return v
        return self._test.get(target_id)

    def resolve_entry(self, entry: dict) -> np.ndarray | None:
        """Resolve embedding from a bank entry dict."""
        src = entry.get("source", {})
        return self.resolve(src.get("target_id", ""))

    @property
    def train_count(self) -> int:
        return len(self._train)

    @property
    def test_count(self) -> int:
        return len(self._test)


def get_emb_manager(datasets_root: Path | None = None) -> DatasetEmbeddingManager:
    """Return the process-level singleton DatasetEmbeddingManager."""
    global _emb_manager
    if _emb_manager is None:
        if datasets_root is None:
            from finacumen.ft.paths import REPO_ROOT
            datasets_root = REPO_ROOT / "datasets"
        _emb_manager = DatasetEmbeddingManager(datasets_root)
    return _emb_manager


def reset_emb_manager() -> None:
    """Clear the singleton (for tests)."""
    global _emb_manager
    _emb_manager = None
