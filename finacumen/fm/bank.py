"""Memory bank — meta.json only (embeddings resolved from pre-computed .npy).

Atomic write via tmp → rename. No fallback, no dead code.
load_emb / save_emb are legacy stubs for backward compatibility.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

BANK_VERSION = 1
MIN_COSINE = 0.65
CONTEXT_WINDOW = 600
K_MAX_DEFAULT = 3
RETRIEVAL_DEDUP_COSINE = 0.92
COLD_QUERY_THRESHOLD = 500

# ── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_meta() -> dict:
    return {"version": BANK_VERSION, "entries": []}


# ── Meta I/O ─────────────────────────────────────────────────────────────────

def load_meta(path: Path) -> list[dict]:
    """Load meta.json, return entries list. Returns [] if file missing."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def save_meta(path: Path, entries: list[dict]) -> None:
    """Atomically write meta.json."""
    path = path.resolve()
    if not entries:
        print(f"WARNING: save_meta called with empty entries at {path}, skipping write")
        return
    # Guard against overwriting a valid file with corrupted (shrunk) data
    if path.exists():
        try:
            old_entries = load_meta(path)
            if len(old_entries) > len(entries):
                print(f"WARNING: save_meta refusing to shrink {path} from {len(old_entries)} to {len(entries)} entries")
                return
        except Exception:
            pass  # old file corrupted, proceed with overwrite
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": BANK_VERSION, "entries": entries}
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ── Emb I/O ──────────────────────────────────────────────────────────────────

def load_emb(path: Path) -> np.ndarray:
    """[legacy] Load emb.npy. Use DatasetEmbeddingManager for new code."""
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    return np.load(path)


def save_emb(path: Path, matrix: np.ndarray) -> None:
    """[legacy] Save float32 matrix to emb.npy. Use prebuild script for new code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix.astype(np.float32))


# ── Entry operations ─────────────────────────────────────────────────────────

def append_entry(meta_path: Path, entry: dict) -> None:
    """Atomically append one entry to meta.json (meta-only, no emb.npy)."""
    entries = load_meta(meta_path)
    entries.append(entry)
    save_meta(meta_path, entries)


def bump_stats(
    meta_path: Path, dataset: str, target_id: str,
    use_delta: int = 0, hit_delta: int = 0,
) -> None:
    """Increment use/hit counters. Meta-only operation (no emb change)."""
    entries = load_meta(meta_path)
    if not entries:
        return
    for entry in entries:
        s = entry.get("source", {})
        if s.get("dataset") == dataset and s.get("target_id") == target_id:
            st = entry.setdefault("stats", {})
            st["use_count"] = st.get("use_count", 0) + use_delta
            st["hit_count"] = st.get("hit_count", 0) + hit_delta
            break
    save_meta(meta_path, entries)


# ── Pruning ──────────────────────────────────────────────────────────────────

def prune(meta_path: Path, total_queries_seen: int) -> int:
    """Remove cold entries (meta-only). Returns number removed."""
    entries = load_meta(meta_path)

    kept_entries = []
    for entry in entries:
        st = entry.get("stats", {})
        if st.get("use_count", 0) == 0 and total_queries_seen >= COLD_QUERY_THRESHOLD:
            continue
        kept_entries.append(entry)

    removed = len(entries) - len(kept_entries)
    if removed > 0:
        save_meta(meta_path, kept_entries)
    return removed
