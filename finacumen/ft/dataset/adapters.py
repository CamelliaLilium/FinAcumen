"""
Dataset adapters — map 4 financial benchmarks to a unified DSER target schema.

Unified target dict (aligned with historical ab_test/targets_finmmr_100.jsonl schema):
    id:             str           unique identifier
    question:       str           question text
    context:        str | None    background text
    options:        str | None    formatted options (for mcq)
    image_paths:    list[str]     local filesystem paths
    gold_answer:    str           gold answer (as string)
    answer_type:    str           "numerical" | "mcq" | "boolean" | "free_text"
    decimal_places: int | None    for numerical
    dataset:        str           "bizbench" | "finmmr_easy" | "finmmr_medium"
                                  | "finmmr_hard" | "fintmm" | "finmme"
    difficulty:     str | None    for finmmr

Datasets supported (entry point: `load_dataset(name, split="test")`):
    bizbench       → datasets/test/bizbench_test.json            (test, default)
                   → datasets/train/bizbench/data/train.json     (train)
    finmmr_easy    → datasets/test/finmmr_easy_test.json         (test)
                   → datasets/train/FinMMR/data/easy_validation_cot_prompt.json (train)
    finmmr_medium  → datasets/test/finmmr_medium_test.json       (test)
                   → datasets/train/FinMMR/data/medium_validation_cot_prompt.json (train)
    finmmr_hard    → datasets/test/finmmr_hard_test.json         (test)
                   → datasets/train/FinMMR/data/hard_validation_cot_prompt.json (train)
    fintmm         → datasets/test/fintmm_test/fintmm_test.json  (test)
                   → datasets/train/FinTMMBench/fintmm_train/fintmm_train.json (train)
    finmme         → datasets/test/finmme_test.json              (test)
                   → datasets/train/FinMME/data/train_split.json (train)

FinMMR special: the dataset ships no standalone train.json — its
{easy,medium,hard}_validation_cot_prompt.json files are used as training
source for memory-bank accumulation (300 items per difficulty, 900 total).

Per-dataset train/test JSON schemas are equivalent enough that the same
load_<dataset>() helpers work for both splits; the only variation we have
to tolerate is that finmmr validation files carry a superset of keys
(`answer`, `program`, `system_input`, ...) which the loader ignores, plus
an `answer` field that duplicates `ground_truth` when both are present.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional

from finacumen.ft.paths import FINACUMEN_PROJECT_ROOT, REPO_ROOT

DATASETS_ROOT = REPO_ROOT / "datasets"
FINMMR_IMAGES_ROOT = DATASETS_ROOT / "train" / "FinMMR" / "images"
FINMME_IMAGE_CACHE = FINACUMEN_PROJECT_ROOT / "workspace" / "finmme_images"


def _infer_numerical(answer: str) -> Optional[int]:
    """Infer decimal_places from a numerical answer string, or None."""
    s = str(answer).strip()
    if re.fullmatch(r"-?\d+", s):
        return 0
    m = re.fullmatch(r"-?\d+\.(\d+)", s)
    return len(m.group(1)) if m else None


def load_bizbench(path: Path, task_filter: str | None = None) -> list[dict]:
    """BizBench — SEC-NUM only. All entries are numerical extraction from SEC filings.

    task_filter is kept for API compatibility but is a no-op (only SEC-NUM data exists).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for idx, row in enumerate(raw):
        answer = str(row.get("answer", ""))
        answer_type = "numerical"
        targets.append({
            "id": f"bizbench-{idx}",
            "question": row["question"],
            "context": row.get("context"),
            "options": None,
            "image_paths": [],
            "gold_answer": answer,
            "answer_type": answer_type,
            "decimal_places": _infer_numerical(answer),
            "dataset": "bizbench",
            "difficulty": "SEC-NUM",
        })
    return targets


def load_finmmr(path: Path, difficulty: str) -> list[dict]:
    """Load finmmr_{easy,medium,hard}_test.json OR the validation_cot_prompt
    train variants. Both share `ground_truth` / `question` / `question_id` /
    `images` / `context`; validation also carries an `answer` duplicate which
    we fall back to only if `ground_truth` is missing."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for row in raw:
        gold = row.get("ground_truth")
        if gold is None:
            gold = row.get("answer")
        if isinstance(gold, float):
            gold_str = f"{gold:.6g}"
            dp = _infer_numerical(gold_str) if "." in gold_str else 0
            atype = "numerical"
        elif isinstance(gold, int):
            gold_str, dp, atype = str(gold), 0, "numerical"
        else:
            gold_str, dp, atype = str(gold), None, "free_text"

        # Remap /root/autodl-tmp/datasets/FinMMR/images/xxx → local path
        image_paths = []
        for img in row.get("images", []) or []:
            name = Path(img).name
            local = FINMMR_IMAGES_ROOT / name
            if local.exists():
                image_paths.append(str(local))

        targets.append({
            "id": row.get("question_id", f"finmmr-{difficulty}-?"),
            "question": row["question"],
            "context": row.get("context"),
            "options": None,
            "image_paths": image_paths,
            "gold_answer": gold_str,
            "answer_type": atype,
            "decimal_places": dp,
            "dataset": f"finmmr_{difficulty}",
            "difficulty": difficulty,
        })
    return targets


FINTMM_CHART_DIR = DATASETS_ROOT / "train" / "FinTMMBench" / "data" / "Chart"

# UUID index for resolving FinTMMBench source references to real data rows.
# Built lazily on first access.
_FINTMM_UUID_INDEX: dict[str, dict] | None = None


def _ensure_fintmm_uuid_index() -> dict[str, dict]:
    global _FINTMM_UUID_INDEX
    if _FINTMM_UUID_INDEX is not None:
        return _FINTMM_UUID_INDEX
    _FINTMM_UUID_INDEX = {}
    data_dir = DATASETS_ROOT / "train" / "FinTMMBench" / "data"
    for filename, label in [
        ("StockPrice.json", "StockPrice"),
        ("FinancialTable.json", "FinancialTable"),
        ("News.json", "News"),
    ]:
        fp = data_dir / filename
        if not fp.exists():
            continue
        for row in json.loads(fp.read_text(encoding="utf-8")):
            uid = row.get("uuid")
            if uid:
                _FINTMM_UUID_INDEX[uid] = row
    return _FINTMM_UUID_INDEX


def _resolve_fintmm_data_context(answers: list[dict]) -> str | None:
    """Resolve source UUIDs in answers to actual data rows, build context text."""
    index = _ensure_fintmm_uuid_index()
    parts: list[str] = []
    for ans in answers or []:
        if not isinstance(ans, dict):
            continue
        for src in ans.get("source") or []:
            if not isinstance(src, str):
                continue
            # Chart refs (Chart_TICKER_PERIOD_Kline) are handled by image_paths
            if src.startswith("Chart_"):
                continue
            row = index.get(src)
            if not row:
                continue
            company = row.get("Company", "")
            symbol = row.get("Symbol", "")
            date_val = row.get("Date", "")
            name = row.get("indicator_name", "")
            value = row.get("indicator_value", "")
            unit = row.get("unit", "")
            parts.append(f"{company} ({symbol}) {date_val}: {name}={value} {unit}".strip())
    return "\n".join(parts) if parts else None


def _resolve_fintmm_chart_paths(answers: list[dict]) -> list[str]:
    """Map ``Chart_<TICKER>_<PERIOD>_Kline`` source refs to filesystem paths.

    fintmm Chart-type questions reference K-line PNG files in
    ``datasets/FinTMMBench/data/Chart/<TICKER>_<PERIOD>_Kline.png``. The
    ``answers[0].source`` strings carry an extra ``Chart_`` prefix that we
    strip before resolving against disk. Non-Chart sources (StockPrice-...,
    News-...) are skipped because they are queryable via the
    financial_data_lookup tool, not visualized.
    """
    paths: list[str] = []
    for ans in answers or []:
        if not isinstance(ans, dict):
            continue
        for src in ans.get("source") or []:
            if not isinstance(src, str) or not src.startswith("Chart_"):
                continue
            stem = src[len("Chart_"):]
            fp = FINTMM_CHART_DIR / f"{stem}.png"
            if fp.exists():
                paths.append(str(fp))
    return paths


def load_fintmm(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for idx, row in enumerate(raw):
        answers = row.get("answers") or []
        gold = str(answers[0]["answer"]) if answers and isinstance(answers[0], dict) else ""
        dp = _infer_numerical(gold)
        atype = "numerical" if dp is not None else "free_text"
        image_paths = _resolve_fintmm_chart_paths(answers) if row.get("type") == "Chart" else []
        context = _resolve_fintmm_data_context(answers)
        targets.append({
            "id": row.get("uuid") or f"fintmm-{idx}",
            "question": row["question"],
            "context": context,
            "options": None,
            "image_paths": image_paths,
            "gold_answer": gold,
            "answer_type": atype,
            "decimal_places": dp,
            "dataset": "fintmm",
            "difficulty": row.get("type"),
        })
    return targets


def load_finmme(path: Path) -> list[dict]:
    """FinMME has base64-encoded image bytes inline. Decode on first load to a
    persistent cache directory so downstream code can use filesystem paths."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    FINMME_IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    targets = []
    for row in raw:
        qid = f"finmme-{row['id']}"
        image_paths = _materialize_finmme_image(row.get("image"), qid)

        q_type = (row.get("question_type") or "").lower()
        options = _format_options(row.get("options"))
        if "choice" in q_type or options:
            atype = "mcq"
        elif q_type in ("numeric", "numerical", "number"):
            atype = "numerical"
        else:
            atype = "free_text"

        answer = str(row.get("answer", ""))
        dp = _infer_numerical(answer) if atype == "numerical" else None

        targets.append({
            "id": qid,
            "question": row.get("question_text", ""),
            "context": row.get("verified_caption"),
            "options": options,
            "image_paths": image_paths,
            "gold_answer": answer,
            "answer_type": atype,
            "decimal_places": dp,
            "tolerance": row.get("tolerance"),
            "question_type": row.get("question_type"),
            "dataset": "finmme",
            "difficulty": None,
        })
    return targets


def _materialize_finmme_image(image_field, qid: str) -> list[str]:
    """Write inline base64 image bytes to cache dir, return local path."""
    if not image_field or not isinstance(image_field, dict):
        return []
    b64 = image_field.get("bytes")
    if not b64:
        return []
    cache_path = FINMME_IMAGE_CACHE / f"{qid}.png"
    if not cache_path.exists():
        cache_path.write_bytes(base64.b64decode(b64))
    return [str(cache_path)]


def _format_options(options) -> Optional[str]:
    if not options:
        return None
    if isinstance(options, str):
        return options
    if isinstance(options, list):
        return "\n".join(
            f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options)
        )
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    return str(options)


# ── Unified entry point ────────────────────────────────────────────────────

_TEST_PATHS = {
    "bizbench":      DATASETS_ROOT / "test" / "bizbench_test.json",
    "finmmr_easy":   DATASETS_ROOT / "test" / "finmmr_easy_test.json",
    "finmmr_medium": DATASETS_ROOT / "test" / "finmmr_medium_test.json",
    "finmmr_hard":   DATASETS_ROOT / "test" / "finmmr_hard_test.json",
    "fintmm":        DATASETS_ROOT / "test" / "fintmm_test" / "fintmm_test.json",
    "finmme":        DATASETS_ROOT / "test" / "finmme_test.json",
}

_TRAIN_PATHS = {
    "bizbench":      DATASETS_ROOT / "train" / "bizbench" / "data" / "train.json",
    "finmmr_easy":   DATASETS_ROOT / "train" / "FinMMR" / "data" / "easy_validation_cot_prompt.json",
    "finmmr_medium": DATASETS_ROOT / "train" / "FinMMR" / "data" / "medium_validation_cot_prompt.json",
    "finmmr_hard":   DATASETS_ROOT / "train" / "FinMMR" / "data" / "hard_validation_cot_prompt.json",
    "fintmm":        DATASETS_ROOT / "train" / "FinTMMBench" / "fintmm_train" / "fintmm_train.json",
    "finmme":        DATASETS_ROOT / "train" / "FinMME" / "data" / "train_split.json",
}

# Backwards-compat alias for callers that imported _PATHS directly. New code
# should use _TEST_PATHS / _TRAIN_PATHS explicitly.
_PATHS = _TEST_PATHS


def load_dataset(name: str, split: str = "test", task_filter: str | None = None) -> list[dict]:
    """Load a named dataset's split into unified target schema.

    split='test' is the default and maps to datasets/test/ (the files all
    previously-existing callers have always loaded). split='train' maps to
    the per-dataset train/validation JSON used by memory-agent's training
    phase — see the module docstring for the mapping.
    
    task_filter: for bizbench, kept for API compat (all entries are SEC-NUM).
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    paths = _TRAIN_PATHS if split == "train" else _TEST_PATHS
    path = paths.get(name)
    if path is None:
        raise ValueError(
            f"Unknown dataset: {name} (split={split}). "
            f"Available: {sorted(paths)}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if name == "bizbench":
        targets = load_bizbench(path, task_filter=task_filter)
    elif name.startswith("finmmr_"):
        targets = load_finmmr(path, name.split("_", 1)[1])
    elif name == "fintmm":
        targets = load_fintmm(path)
    elif name == "finmme":
        targets = load_finmme(path)
    else:
        raise ValueError(f"No loader for dataset: {name}")

    # Tag every target with its split so downstream analysis (training_log,
    # retrieval_judge, reasoning_quality_judge) can partition train vs test
    # without re-deriving from IDs.
    for t in targets:
        t["split"] = split
    return targets
