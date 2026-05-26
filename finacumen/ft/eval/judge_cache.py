"""Persistent GPT-4o-mini judge cache, keyed by (row, dataset, target_id).

Each (row, dataset) gets a JSONL file under results/main_table/judge_cache/.
Each line records one item's judge verdict so re-runs don't re-pay for OpenAI
calls. Records are append-only — when a new run produces a different prediction
for the same target_id, both verdicts are kept (most recent wins on lookup).

Usage:
    from finacumen.ft.eval.judge_cache import judge_with_cache
    yes_count, total = await judge_with_cache(
        row='qwen3vl_ours', dataset='finmme',
        items=[(target_dict, predicted_str), ...],
    )
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from finacumen.ft.eval.fintmm_eval import _make_judge_client, _JUDGE_PROMPT  # type: ignore


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "results/main_table/judge_cache"


def _cache_path(row: str, dataset: str) -> Path:
    return CACHE_DIR / f"{row}_{dataset}.jsonl"


def _pred_hash(pred: str) -> str:
    return hashlib.sha1(pred.encode("utf-8", errors="ignore")).hexdigest()[:12]


def load_cache(row: str, dataset: str) -> dict[str, dict]:
    """Return {target_id -> latest record dict}. Latest pred_hash wins."""
    p = _cache_path(row, dataset)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["target_id"]] = rec  # last write wins
            except Exception:
                continue
    return out


def append_records(row: str, dataset: str, records: list[dict]) -> None:
    p = _cache_path(row, dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _judge_single(client, model: str, question: str, gold: str, pred: str) -> bool:
    prompt = _JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("YES")
    except Exception:
        return False


async def judge_with_cache(
    row: str,
    dataset: str,
    items: list[tuple[dict, str]],
    concurrency: int = 4,
) -> tuple[int, int]:
    """Judge all items, using cache to skip already-evaluated ones.

    Returns (yes_count, total).
    """
    cache = load_cache(row, dataset)
    client, model = _make_judge_client()

    # Check cache; collect items that need fresh judging
    yes_count = 0
    total = 0
    fresh_targets: list[tuple[dict, str]] = []
    for target, pred in items:
        tid = target.get("id")
        if tid is None:
            continue
        gold = str(target.get("gold_answer", ""))
        ph = _pred_hash(pred)
        rec = cache.get(tid)
        if rec is not None and rec.get("pred_hash") == ph:
            total += 1
            if rec.get("yes"):
                yes_count += 1
        else:
            fresh_targets.append((target, pred))

    # Judge fresh items
    if fresh_targets:
        sem = asyncio.Semaphore(concurrency)

        async def _one(t, p):
            async with sem:
                question = str(t.get("question", ""))
                gold = str(t.get("gold_answer", ""))
                yes = await _judge_single(client, model, question, gold, p)
                return {
                    "target_id": t.get("id"),
                    "gold": gold,
                    "pred": p[:300],  # cap to keep cache file manageable
                    "pred_hash": _pred_hash(p),
                    "yes": yes,
                    "ts": int(time.time()),
                }

        verdicts = await asyncio.gather(*(_one(t, p) for t, p in fresh_targets))
        append_records(row, dataset, verdicts)
        for v in verdicts:
            total += 1
            if v.get("yes"):
                yes_count += 1

    return yes_count, total
