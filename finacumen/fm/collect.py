"""Collect — cross-verify multi-path reasoning, lookup pre-computed embedding, dedup, write to bank.

Uses cross_verify module for parallel reasoning + synthesis + experience distillation.
Embedding resolved from pre-computed datasets/*_emb.npy via DatasetEmbeddingManager.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from finacumen.fm.bank import (
    append_entry,
    load_meta,
)
from finacumen.fm.cross_verify import collect_verified
from finacumen.fm.emb_manager import get_emb_manager
from finacumen.fm.pruning_rules import is_anti_give_up
from finacumen.fm.schema import Trace
from finacumen.ft.logger import logger

# ── Constants ────────────────────────────────────────────────────────────────

COLLECT_ERRORS_FILENAME = "collect_errors.jsonl"


# ── Public API ───────────────────────────────────────────────────────────────


async def collect(
    target: dict,
    trace: Trace,
    final_answer: str,
    is_correct: bool,
    bank_dir: Path,
) -> Optional[dict]:
    """Run cross-verify collection, embed question+context, dedup, write to bank.

    Returns the new entry dict if written, None if dedup'd or blocked.
    """
    meta_path = bank_dir / "meta.json"

    # 1. Cross-verify collection (multi-path reasoning + synthesis + distillation)
    result = await collect_verified(target)
    if result is None:
        _log_collect_error(bank_dir, target, "collect_verified_none")
        return None

    experience_dict = result["experience"]
    analysis_text = result.get("analysis", "")

    findings = experience_dict.get("findings", [])
    cautions = experience_dict.get("cautions", [])

    # 2. Per-item anti-give-up filter: remove individual give-up strings.
    findings = [f for f in findings if not is_anti_give_up(f)]
    cautions = [c for c in cautions if not is_anti_give_up(c)]
    experience_dict["findings"] = findings
    experience_dict["cautions"] = cautions

    if not findings and not cautions:
        _log_collect_error(bank_dir, target, "anti_give_up_emptied")
        logger.warning(f"collect: empty findings and cautions after anti-give-up filter for {target.get('id')}")
        return None

    # 3. Dedup: exact target_id match only (one question → one memory entry).
    mgr = get_emb_manager()
    target_id = target.get("id", "")
    target_emb = mgr.resolve(target_id)
    if target_emb is None:
        _log_collect_error(bank_dir, target, "no_embedding")
        logger.warning(f"collect: no pre-computed embedding for {target_id}")
        return None

    entries = load_meta(meta_path)
    for existing in entries:
        existing_src = existing.get("source", {})
        if existing_src.get("target_id") == target_id:
            _log_collect_error(bank_dir, target, "exact_dedup")
            return None

    # 4. Build entry
    q_text = target.get("question", "")
    ctx = target.get("context") or ""
    question_full = q_text
    if ctx:
        question_full += "\n\nContext: " + str(ctx)
    img_paths = target.get("image_paths") or []
    if img_paths:
        question_full += "\n[Chart image was provided.]"

    entry = {
        "source": {
            "dataset": target.get("dataset", ""),
            "target_id": target_id,
        },
        "experience": experience_dict,
        "analysis": analysis_text,
        "question": question_full,
        "gold_answer": str(target.get("gold_answer", "")),
        "image_paths": img_paths,
        "stats": {"use_count": 0, "hit_count": 0},
        "created_at": _utc_now(),
        "source_variant": trace.source_variant,
    }

    # 5. Write to bank (meta-only)
    append_entry(meta_path, entry)
    return entry


# ── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_collect_error(bank_dir: Path, target: dict, error: str, detail: str = "") -> None:
    errors_path = bank_dir / COLLECT_ERRORS_FILENAME
    record = {
        "timestamp": _utc_now(),
        "target_id": target.get("id", ""),
        "error": error,
        "detail": detail,
    }
    try:
        with open(errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
