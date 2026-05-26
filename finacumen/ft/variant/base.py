"""
DSERVariant — unified protocol for all FinAcumen variant strategies.

Each variant (baseline-raw, ft-only, finacumen) implements a single
`solve(target) -> result` coroutine. The benchmark harness iterates over
targets, calls `variant.solve`, and records the result dict for statistical comparison.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from finacumen.ft.eval.bizbench_eval import bizbench_is_correct
from finacumen.ft.eval.finmme_eval import finmme_item_correct
from finacumen.ft.eval.finmmr_eval import finmmr_is_correct
from finacumen.ft.eval.fintmm_eval import squad_em


def _score_native(target: dict, predicted: str) -> bool:
    """Route to the correct per-dataset native scorer."""
    ds = target.get("dataset", "")
    if ds.startswith("finmmr"):
        return finmmr_is_correct(target, predicted)
    if ds == "bizbench":
        tgt = dict(target)
        tgt.setdefault("task", "SEC-NUM")
        return bizbench_is_correct(tgt, predicted)
    if ds == "finmme":
        return finmme_item_correct(target, predicted)
    if ds == "fintmm":
        return squad_em(str(target.get("gold_answer", "")), predicted) > 0.5
    return False


class DSERVariant(ABC):
    """Abstract variant: maps a target dict to a result dict."""

    name: str = "base"

    @abstractmethod
    async def solve(self, target: dict) -> dict:
        """Run the variant on one target, return result in run_dser.py schema."""

    @staticmethod
    def build_result(
        target: dict,
        final_answer: str,
        *,
        extras: dict[str, Any] | None = None,
        latency_sec: float | None = None,
    ) -> dict:
        """Assemble the standard result dict."""
        return {
            "target_id": target["id"],
            "question": target.get("question", ""),
            "gold_answer": target.get("gold_answer", ""),
            "answer_type": target.get("answer_type", ""),
            "dataset": target.get("dataset", ""),
            "difficulty": target.get("difficulty"),
            "final_answer": final_answer,
            "correct": _score_native(target, final_answer),
            "latency_sec": latency_sec if latency_sec is not None else 0.0,
            **(extras or {}),
        }

    @staticmethod
    def stopwatch() -> "Stopwatch":
        return Stopwatch()


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.time()

    def elapsed(self) -> float:
        return round(time.time() - self._t0, 2)
