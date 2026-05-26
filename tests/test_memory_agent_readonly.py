"""memory-agent: train split paths and test-mode read-only bank semantics (V3).

V3 FinAcumenVariant.solve() calls the real LLM via agent.run().
To test counter semantics without real API calls, we mock both
ToolCallAgent.run (no-op) and _extract_final_answer (controlled return).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from finacumen.fm import bank
from finacumen.ft.dataset.adapters import _TRAIN_PATHS
from finacumen.ft.variant.finacumen import FinAcumenVariant

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_entry(
    dataset: str, target_id: str,
    use_count: int = 3, hit_count: int = 1,
) -> dict:
    return {
        "source": {"dataset": dataset, "target_id": target_id},
        "experience": {"findings": [], "cautions": []},
        "analysis": "test entry for counter semantics",
        "question": "test question",
        "gold_answer": "42",
        "stats": {"use_count": use_count, "hit_count": hit_count},
        "created_at": _utc_now(),
        "source_variant": "finacumen",
    }


def _find_entry_by_source(entries: list[dict], dataset: str, target_id: str) -> dict | None:
    for e in entries:
        src = e.get("source", {})
        if src.get("dataset") == dataset and src.get("target_id") == target_id:
            return e
    return None


def test_train_split_paths_point_under_datasets_train() -> None:
    for name, p in _TRAIN_PATHS.items():
        assert "train" in p.parts, f"{name}: expected under datasets/train/, got {p}"


def test_load_train_split_when_datasets_present() -> None:
    sample = REPO_ROOT / "datasets" / "train" / "bizbench" / "data" / "train.json"
    if not sample.exists():
        pytest.skip("datasets/train not populated")
    from finacumen.ft.dataset.adapters import load_dataset

    for name in _TRAIN_PATHS:
        rows = load_dataset(name, split="train")
        assert len(rows) > 0, name


def test_memory_agent_test_mode_does_not_bump_counters(tmp_path: Path) -> None:
    async def _run() -> None:
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir(parents=True, exist_ok=True)
        meta_path = bank_dir / "meta.json"
        entry = _make_entry("finmmr_easy", "x", use_count=3, hit_count=1)
        bank.append_entry(meta_path, entry)

        retrieval = {
            "mode": "with-memory",
            "experiences": [
                {
                    "source": {"dataset": "finmmr_easy", "target_id": "x"},
                    "experience": {"findings": [], "cautions": []},
                    "question": "test question",
                    "gold_answer": "42",
                }
            ],
            "scores": [0.85],
        }

        args = argparse.Namespace(
            memory_dir=str(bank_dir),
            memory_k_max=3,
            collect_concurrency=1,
            memory_mode="test",
        )
        variant = FinAcumenVariant(args)

        target = {
            "id": "probe-target",
            "dataset": "finmmr_easy",
            "gold_answer": "42",
            "answer_type": "numerical",
            "question": "q",
            "context": "",
            "difficulty": "easy",
        }

        with patch(
            "finacumen.ft.variant.finacumen.retrieve",
            new=AsyncMock(return_value=retrieval),
        ), patch(
            "finacumen.ft.agent.toolcall.ToolCallAgent.run",
            new=AsyncMock(),
        ), patch(
            "finacumen.ft.variant.finacumen._extract_final_answer",
            return_value="99",  # wrong answer — should not affect test-mode counters
        ):
            await variant.solve(target)

        after_entry = _find_entry_by_source(
            bank.load_meta(meta_path), "finmmr_easy", "x"
        )
        assert after_entry is not None
        # test mode: counters unchanged regardless of answer correctness
        assert after_entry["stats"]["use_count"] == 3
        assert after_entry["stats"]["hit_count"] == 1

    asyncio.run(_run())


def test_memory_agent_train_mode_bumps_counters(tmp_path: Path) -> None:
    async def _run() -> None:
        bank_dir = tmp_path / "bank2"
        bank_dir.mkdir(parents=True, exist_ok=True)
        meta_path = bank_dir / "meta.json"
        entry = _make_entry("finmmr_easy", "y", use_count=3, hit_count=1)
        bank.append_entry(meta_path, entry)

        retrieval = {
            "mode": "with-memory",
            "experiences": [
                {
                    "source": {"dataset": "finmmr_easy", "target_id": "y"},
                    "experience": {"findings": [], "cautions": []},
                    "question": "test question",
                    "gold_answer": "42",
                }
            ],
            "scores": [0.85],
        }

        args = argparse.Namespace(
            memory_dir=str(bank_dir),
            memory_k_max=3,
            collect_concurrency=1,
            memory_mode="train",
        )
        variant = FinAcumenVariant(args)

        target = {
            "id": "t2",
            "dataset": "finmmr_easy",
            "gold_answer": "42",
            "answer_type": "numerical",
            "question": "q",
            "context": "",
            "difficulty": "easy",
        }

        with patch(
            "finacumen.ft.variant.finacumen.retrieve",
            new=AsyncMock(return_value=retrieval),
        ), patch(
            "finacumen.ft.agent.toolcall.ToolCallAgent.run",
            new=AsyncMock(),
        ), patch(
            "finacumen.ft.variant.finacumen._extract_final_answer",
            return_value="42",  # correct answer triggers hit bump
        ), patch(
            "finacumen.ft.variant.finacumen.collect_experience",
            new=AsyncMock(return_value=None),
        ), patch(
            "finacumen.ft.variant.finacumen.trace_adapter.build_trace",
            return_value=None,
        ):
            await variant.solve(target)
            await variant.finalize()

        after_entry = _find_entry_by_source(
            bank.load_meta(meta_path), "finmmr_easy", "y"
        )
        assert after_entry is not None
        # V3: use_count bumped per retrieved entry in train mode
        assert after_entry["stats"]["use_count"] == 4
        # V3: hit_count bumped only when answer is correct
        assert after_entry["stats"]["hit_count"] == 2

    asyncio.run(_run())
