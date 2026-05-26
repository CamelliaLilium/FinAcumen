"""Self-contained native evaluation helpers for FinAcumen experiments.

This module is the shared scoring and aggregation layer used by both the core
package and the thin experiment CLIs.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


DATASETS = (
    "bizbench",
    "finmmr_easy",
    "finmmr_medium",
    "finmmr_hard",
    "fintmm",
    "finmme",
)

LETTER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*([^\s\n]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_targets(path: Path, dataset: str = "") -> dict[str, dict[str, Any]]:
    """Load target rows, using dataset adapter when available for correct id keys.
    Falls back to raw JSON with index-based keys if adapter unavailable."""
    if dataset:
        try:
            from finacumen.ft.dataset.adapters import load_dataset
            rows = load_dataset(dataset, split="test")
            return {str(row["id"]): row for row in rows}
        except Exception:
            pass
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [])
    result = {}
    for idx, row in enumerate(rows):
        key = str(row.get("id") or row.get("target_id") or "")
        if not key:
            key = f"{dataset}-{idx}" if dataset else str(idx)
        result[key] = row
    return result


def find_target_file(targets_root: Path, dataset: str) -> Path:
    candidates = [
        targets_root / f"{dataset}_test.json",
        targets_root / f"{dataset}.jsonl",
        targets_root / f"{dataset}.json",
        targets_root / dataset / "test.jsonl",
        targets_root / dataset / "test.json",
        targets_root / dataset / "data" / "test.jsonl",
        targets_root / dataset / "data" / "test.json",
    ]
    if dataset == "fintmm":
        candidates.insert(0, targets_root / "fintmm_test" / "fintmm_test.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No target file found for {dataset!r} under {targets_root}. "
        "Expected <dataset>.jsonl/json or <dataset>/test.jsonl/json."
    )


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_basic(value: Any) -> str:
    return re.sub(r"[\s$%,]", "", _norm_text(value)).lower()


def _first_number(value: Any) -> float | None:
    text = _norm_text(value)
    text = text.replace("−", "-").replace("×", "x")
    text = re.sub(r"(?i)\b(usd|rmb|dollars?|yuan|million|billion|thousand)\b", " ", text)
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _letter_set(value: Any) -> set[str]:
    text = _norm_text(value)
    match = ANSWER_RE.search(text)
    if match:
        text = match.group(1)
    compact = re.sub(r"[^A-Za-z]", "", text)
    if 1 <= len(compact) <= 8:
        return {c.upper() for c in compact}
    return {m.group(1).upper() for m in LETTER_RE.finditer(text)}


def _is_mcq(target: dict[str, Any], pred: Any) -> bool:
    answer_type = str(target.get("answer_type", "")).lower()
    if "mcq" in answer_type or "choice" in answer_type:
        return True
    gold_letters = _letter_set(target.get("gold_answer"))
    pred_letters = _letter_set(pred)
    return bool(gold_letters and pred_letters)


def bizbench_correct(target: dict[str, Any], pred: Any) -> bool:
    return _strip_basic(pred) == _strip_basic(target.get("gold_answer"))


def finmmr_correct(target: dict[str, Any], pred: Any) -> bool:
    gold = _norm_text(target.get("gold_answer")).lower()
    pred_text = _norm_text(pred).lower()
    if gold in {"yes", "no", "true", "false"}:
        mapping = {"true": "yes", "false": "no"}
        return mapping.get(pred_text, pred_text) == mapping.get(gold, gold)
    gold_num = _first_number(gold)
    pred_num = _first_number(pred)
    if gold_num is None or pred_num is None:
        return _strip_basic(pred) == _strip_basic(gold)
    tolerance = abs(gold_num) * 0.002
    return abs(pred_num - gold_num) <= tolerance


def finmme_correct(target: dict[str, Any], pred: Any) -> bool:
    if _is_mcq(target, pred):
        return _letter_set(pred) == _letter_set(target.get("gold_answer"))
    gold_num = _first_number(target.get("gold_answer"))
    pred_num = _first_number(pred)
    if gold_num is None or pred_num is None:
        return _strip_basic(pred) == _strip_basic(target.get("gold_answer"))
    tolerance = float(target.get("tolerance") or 0.0)
    return abs(pred_num - gold_num) <= tolerance


def fintmm_exact_correct(target: dict[str, Any], pred: Any) -> bool:
    """Fallback exact scorer.

    The paper protocol for FinTMMBench is LLM-as-judge. This exact scorer is
    deterministic and useful for smoke checks, but reports should label it as a
    fallback unless external judge annotations are supplied.
    """
    gold = _strip_basic(target.get("gold_answer"))
    guess = _strip_basic(pred)
    return bool(gold) and (gold == guess or gold in guess or guess in gold)


def score_native(dataset_dotted: str, target: dict[str, Any], pred: Any) -> bool:
    """Authoritative native scoring entry for FinAcumen experiments."""
    from finacumen.ft.eval.bizbench_eval import bizbench_is_correct
    from finacumen.ft.eval.finmme_eval import finmme_item_correct
    from finacumen.ft.eval.finmmr_eval import finmmr_is_correct
    from finacumen.ft.eval.fintmm_eval import squad_em

    pred_s = str(pred or "").strip()
    if not pred_s:
        return False

    if dataset_dotted.startswith("finmmr"):
        return finmmr_is_correct(target, pred_s)
    if dataset_dotted == "bizbench":
        tgt = dict(target)
        tgt.setdefault("task", "SEC-NUM")
        return bizbench_is_correct(tgt, pred_s)
    if dataset_dotted == "finmme":
        return finmme_item_correct(target, pred_s)
    if dataset_dotted == "fintmm":
        return squad_em(str(target.get("gold_answer", "")), pred_s) > 0.5
    raise ValueError(f"Unsupported dataset: {dataset_dotted!r}")


def native_correct(dataset: str, target: dict[str, Any], pred: Any) -> bool:
    """Alias used by result aggregators; raises on unsupported scoring paths."""
    return score_native(dataset, target, pred)


def fallback_correct(dataset: str, target: dict[str, Any], pred: Any) -> bool:
    """Deterministic fallback scorer for smoke checks that explicitly opt in."""
    if dataset.startswith("finmmr"):
        return finmmr_correct(target, pred)
    if dataset == "bizbench":
        return bizbench_correct(target, pred)
    if dataset == "finmme":
        return finmme_correct(target, pred)
    if dataset == "fintmm":
        return fintmm_exact_correct(target, pred)
    raise ValueError(f"Unsupported dataset: {dataset}")


def evaluate_results(
    dataset: str,
    results_path: Path,
    targets_path: Path,
    *,
    limit: int | None = None,
    judge_enabled: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl(results_path, limit=limit)
    targets = load_targets(targets_path, dataset=dataset)
    correct = 0
    n = 0
    missing_targets = 0
    for row in rows:
        target_id = str(row.get("target_id") or row.get("id"))
        target = targets.get(target_id)
        if target is None:
            missing_targets += 1
            continue
        n += 1
        if judge_enabled and dataset == "fintmm":
            if row.get("judge_correct"):
                correct += 1
        else:
            pred = row.get("final_answer", row.get("prediction", ""))
            if native_correct(dataset, target, pred):
                correct += 1
    pct = correct / n * 100 if n else math.nan
    return {
        "dataset": dataset,
        "n": n,
        "correct": correct,
        "accuracy": pct,
        "missing_targets": missing_targets,
        "results_path": str(results_path),
        "targets_path": str(targets_path),
        "fintmm_note": "LLM-as-judge (GPT-4o-mini)."
        if (judge_enabled and dataset == "fintmm")
        else (
            "Exact fallback; paper protocol requires LLM-as-judge."
            if dataset == "fintmm"
            else ""
        ),
    }
