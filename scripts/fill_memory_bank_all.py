"""Fill ALL memory datasets to 300 entries using training-set question -> memory 1:1 mapping.

Reads training data via load_dataset(), identifies missing target_ids per bank,
runs collect() with cross-verify pipeline and concurrency control.

Usage:
    python scripts/fill_memory_bank_all.py

Datasets covered:
    fintmm (127->300), finmme (100->300),
    finmmr_easy (263->300), finmmr_hard (210->300), finmmr_medium (217->300)
    bizbench (645->skip)
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from finacumen.fm.collect import collect
from finacumen.fm.schema import Trace
from finacumen.ft.dataset.adapters import load_dataset
from finacumen.ft.logger import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = _REPO_ROOT / "memory"

TARGET = 300
MAX_CONCURRENT = 3  # each collect() spawns 8 parallel agents internally
PROGRESS_INTERVAL = 30  # seconds between progress reports
RATE_LIMIT_COOLDOWN = 15  # seconds to wait on 429


@dataclass
class Progress:
    dataset: str
    total: int = 0
    written: int = 0
    skipped: int = 0
    errors: int = 0
    rate_limits: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def done(self) -> int:
        return self.written + self.skipped

    @property
    def remaining(self) -> int:
        return self.total - self.done

    @property
    def elapsed_min(self) -> float:
        return (time.time() - self.started_at) / 60.0

    @property
    def rate_ph(self) -> float:
        if self.elapsed_min < 0.5:
            return 0
        return self.written / (self.elapsed_min / 60.0)


def _get_collected_ids(bank_path: Path) -> set[str]:
    if not bank_path.exists():
        return set()
    with open(bank_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["source"]["target_id"] for e in data.get("entries", [])}


def _count_errors(bank_path: Path) -> int:
    errors_path = bank_path / "collect_errors.jsonl"
    if not errors_path.exists():
        return 0
    return sum(1 for _ in errors_path.read_text(encoding="utf-8").strip().split("\n") if _)


def _progress_line(p: Progress) -> str:
    eta = ""
    if p.done > 0 and p.written > 0:
        eta_min = p.remaining / max(p.rate_ph or 1, 1) * 60.0
        eta = f"  ETA={eta_min:.0f}min"
    limits = f"  429s={p.rate_limits}" if p.rate_limits else ""
    return (
        f"[{p.dataset:>14s}] {p.done:3d}/{p.total} "
        f"written={p.written} skip={p.skipped} err={p.errors} "
        f"rate={p.rate_ph:.1f}/h{eta}{limits}"
    )


async def _progress_reporter(progresses: dict[str, Progress], stop_event: asyncio.Event):
    while not stop_event.is_set():
        await asyncio.sleep(PROGRESS_INTERVAL)
        if stop_event.is_set():
            break
        print()
        for p in progresses.values():
            print(_progress_line(p))
        print(flush=True)


def _prepare_targets(dataset_name: str, bank_name: str) -> list[dict]:
    """Load training data, exclude already-collected ids, return pending targets."""
    all_targets = load_dataset(dataset_name, split="train")
    bank_path = BANK_DIR / bank_name
    collected_ids = _get_collected_ids(bank_path / "meta.json")

    pending = [t for t in all_targets if t["id"] not in collected_ids]
    current = len(collected_ids)
    need = max(0, TARGET - current)

    if need == 0:
        return []

    buffer = max(10, int(need * 0.25))
    take = min(need + buffer, len(pending))
    selected = random.sample(pending, take)

    logger.info(
        f"{bank_name}: current={current}, need={need}, "
        f"pending_total={len(pending)}, selected={len(selected)} (buffer={buffer})"
    )
    return selected


async def fill_dataset(dataset_name: str, bank_name: str, prog: Progress, sem: asyncio.Semaphore) -> None:
    bank_path = BANK_DIR / bank_name
    bank_path.mkdir(parents=True, exist_ok=True)

    targets = _prepare_targets(dataset_name, bank_name)
    if not targets:
        logger.info(f"{bank_name}: already at target ({TARGET}+)")
        return

    prog.total = len(targets)
    logger.info(f"{bank_name}: {prog.total} targets to collect (concurrency={MAX_CONCURRENT})")

    async def _collect_one(target: dict) -> Optional[dict]:
        async with sem:
            trace = Trace(source_variant="__ft_only_with_trace__")
            try:
                entry = await collect(
                    target=target,
                    trace=trace,
                    final_answer="",
                    is_correct=False,
                    bank_dir=bank_path,
                )
                if entry is not None:
                    prog.written += 1
                    f_cnt = len(entry["experience"].get("findings", []))
                    c_cnt = len(entry["experience"].get("cautions", []))
                    logger.info(f"  OK {target['id']} (f={f_cnt} c={c_cnt})")
                else:
                    prog.skipped += 1
                    logger.warning(f"  SKIP {target['id']} (check collect_errors.jsonl)")
                return entry
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate" in err_msg.lower():
                    prog.rate_limits += 1
                    logger.warning(f"  RATE-LIMIT {target['id']}: {e} - cooling {RATE_LIMIT_COOLDOWN}s")
                    await asyncio.sleep(RATE_LIMIT_COOLDOWN)
                else:
                    prog.errors += 1
                    logger.error(f"  FAIL {target['id']}: {e}")
                return None

    tasks = [asyncio.create_task(_collect_one(t)) for t in targets]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"{bank_name}: done - {prog.written} written, {prog.skipped} skipped, {prog.errors} errors")


async def main() -> None:
    configs = [
        ("fintmm",        "fintmm"),
        ("finmme",        "finmme"),
        ("finmmr_easy",   "finmmr_easy"),
        ("finmmr_hard",   "finmmr_hard"),
        ("finmmr_medium", "finmmr_medium"),
    ]

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    stop_event = asyncio.Event()

    print("=== Memory Bank Fill (target: 300 per dataset) ===\n")

    # Pre-flight counts
    for _, bank_name in configs:
        bank_path = BANK_DIR / bank_name
        ids = _get_collected_ids(bank_path / "meta.json")
        errs = _count_errors(bank_path)
        need = max(0, TARGET - len(ids))
        status = "SKIP" if need == 0 else f"+{need}"
        print(f"  {bank_name:14s} {len(ids):4d} entries  {errs:3d} errors  -> {status}")

    print()

    progresses: dict[str, Progress] = {}
    for ds_name, bank_name in configs:
        progresses[bank_name] = Progress(dataset=bank_name)

    reporter = asyncio.create_task(_progress_reporter(progresses, stop_event))

    fill_tasks = [
        asyncio.create_task(fill_dataset(ds_name, bank_name, progresses[bank_name], sem))
        for ds_name, bank_name in configs
    ]
    await asyncio.gather(*fill_tasks, return_exceptions=True)

    stop_event.set()
    await reporter

    # Final report
    print("\n=== FINAL ===")
    total_written = 0
    for ds_name, bank_name in configs:
        p = progresses[bank_name]
        ids = _get_collected_ids(BANK_DIR / bank_name / "meta.json")
        errs = _count_errors(BANK_DIR / bank_name)
        total_written += p.written
        shortfall = max(0, TARGET - len(ids))
        status = "DONE" if shortfall == 0 else f"SHORT {shortfall}"
        print(f"  {bank_name:14s} {len(ids):4d}/{TARGET} entries  {errs:3d} errors  "
              f"(this run: +{p.written}w +{p.skipped}s +{p.errors}e)  {status}")
    print(f"\nTotal written this run: {total_written}")
    print(f"Total 429 rate-limit hits: {sum(p.rate_limits for p in progresses.values())}")


if __name__ == "__main__":
    asyncio.run(main())
