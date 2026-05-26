"""Fill memory banks to per-dataset targets using training-set question → memory 1:1 mapping.

Supports parallel execution across multiple API keys — each instance writes to
memory/<dataset>/meta.json, no cross-process conflicts.

Usage:
    python scripts/fill_memory_bank.py --datasets finmmr_easy,finmmr_hard,finmmr_medium
    python scripts/fill_memory_bank.py --datasets finmme --target finmme=600
    python scripts/fill_memory_bank.py --datasets fintmm --config config/config_key2.toml

Safety:
    - Verifies train/test ID separation at startup (fatal if overlap found)
    - bizbench filtered to SEC-NUM only
    - Already-collected target_ids are skipped (resumable)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_TARGETS = {
    "finmmr_easy":   300,
    "finmmr_hard":   300,
    "finmmr_medium": 300,
    "finmme":        500,
    "fintmm":        500,
}

MAX_CONCURRENT = 3
BUFFER_FACTOR = 0.25
RATE_LIMIT_COOLDOWN = 15
PROGRESS_INTERVAL = 30

_REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = _REPO_ROOT / "memory"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill memory banks to per-dataset targets")
    parser.add_argument(
        "--datasets", required=True,
        help="Comma-separated dataset names (finmmr_easy,finmmr_hard,finmmr_medium,finmme,fintmm,bizbench)",
    )
    parser.add_argument(
        "--target", action="append", default=[],
        help="Override target count: dataset=COUNT (can repeat). "
             f"Defaults: {DEFAULT_TARGETS}",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.toml (sets FINACUMEN_CONFIG_PATH before importing finacumen)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=MAX_CONCURRENT,
        help=f"Max concurrent collects (default: {MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--buffer-factor", type=float, default=BUFFER_FACTOR,
        help=f"Extra targets to pick as buffer (default: {BUFFER_FACTOR})",
    )
    return parser.parse_args()


def _parse_targets(args: argparse.Namespace) -> dict[str, int]:
    targets = dict(DEFAULT_TARGETS)
    for override in args.target:
        ds, _, count = override.partition("=")
        targets[ds.strip()] = int(count)
    return targets


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


def _verify_no_test_leak(dataset_name: str, test_ids: set[str]) -> None:
    """Fatal-exit if any test split ID appears in the train data for dataset_name."""
    from finacumen.ft.dataset.adapters import load_dataset
    from finacumen.ft.logger import logger

    train_data = load_dataset(dataset_name, split="train")
    train_ids = {t["id"] for t in train_data}
    overlap = train_ids & test_ids
    if overlap:
        logger.error(
            f"FATAL: {len(overlap)} test IDs leaked into {dataset_name} train data! "
            f"Sample: {sorted(overlap)[:10]}"
        )
        sys.exit(1)
    logger.info(f"  test-safety {dataset_name}: train={len(train_ids)} test={len(test_ids)} overlap=0 OK")


def _load_all_test_ids() -> set[str]:
    """Load all test split IDs from every dataset. Returns empty set on partial failure."""
    from finacumen.ft.dataset.adapters import load_dataset
    from finacumen.ft.logger import logger

    all_test = set()
    ds_names = ["bizbench", "finmme", "finmmr_easy", "finmmr_hard", "finmmr_medium", "fintmm"]
    for ds in ds_names:
        try:
            test_data = load_dataset(ds, split="test")
            ids = {t.get("id", "") for t in test_data}
            all_test.update(ids)
            logger.info(f"  loaded test IDs for {ds}: {len(ids)}")
        except Exception as e:
            logger.warning(f"  could not load test data for {ds}: {e}")
    return all_test


def _parse_dataset_list(raw: str) -> list[str]:
    return [d.strip() for d in raw.split(",") if d.strip()]


def _prepare_targets(
    dataset_name: str, bank_name: str, target_count: int, buffer_factor: float,
) -> list[dict]:
    from finacumen.ft.dataset.adapters import load_dataset
    from finacumen.ft.logger import logger

    all_targets = load_dataset(dataset_name, split="train")

    if dataset_name == "bizbench":
        all_targets = [t for t in all_targets if t.get("task") == "SEC-NUM"]
        logger.info(f"  bizbench SEC-NUM filter: {len(all_targets)} remaining")

    bank_path = BANK_DIR / bank_name
    collected_ids = _get_collected_ids(bank_path / "meta.json")
    pending = [t for t in all_targets if t["id"] not in collected_ids]
    current = len(collected_ids)
    need = max(0, target_count - current)

    if need == 0:
        return []

    buffer = max(10, int(need * buffer_factor))
    take = min(need + buffer, len(pending))
    selected = random.sample(pending, take)

    logger.info(
        f"{bank_name}: current={current}, target={target_count}, need={need}, "
        f"pending_total={len(pending)}, selected={len(selected)} (buffer={buffer})"
    )
    return selected


async def fill_dataset(
    dataset_name: str, bank_name: str, target_count: int, prog: Progress,
    sem: asyncio.Semaphore, max_concurrent: int, buffer_factor: float,
) -> None:
    from finacumen.fm.collect import collect
    from finacumen.fm.schema import Trace
    from finacumen.ft.logger import logger

    bank_path = BANK_DIR / bank_name
    bank_path.mkdir(parents=True, exist_ok=True)

    targets = _prepare_targets(dataset_name, bank_name, target_count, buffer_factor)
    if not targets:
        logger.info(f"{bank_name}: already at target ({target_count})")
        return

    prog.total = len(targets)
    logger.info(f"{bank_name}: {prog.total} targets to collect (concurrency={max_concurrent})")

    async def _collect_one(target: dict) -> Optional[dict]:
        async with sem:
            try:
                entry = await collect(
                    target=target,
                    trace=Trace(source_variant="__ft_only_with_trace__"),
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
                    logger.warning(f"  RATE-LIMIT {target['id']}: {e} — cooling {RATE_LIMIT_COOLDOWN}s")
                    await asyncio.sleep(RATE_LIMIT_COOLDOWN)
                else:
                    prog.errors += 1
                    logger.error(f"  FAIL {target['id']}: {e}")
                return None

    tasks = [asyncio.create_task(_collect_one(t)) for t in targets]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"{bank_name}: done — {prog.written} written, {prog.skipped} skipped, {prog.errors} errors")


async def run(args: argparse.Namespace) -> None:
    from finacumen.ft.logger import logger

    ds_list = _parse_dataset_list(args.datasets)
    targets = _parse_targets(args)

    # Validate all requested datasets have targets
    for ds in ds_list:
        if ds not in targets:
            logger.error(f"No target defined for dataset '{ds}'. Add --target {ds}=COUNT")
            sys.exit(1)
        if ds not in DEFAULT_TARGETS and ds != "bizbench":
            logger.warning(f"Unknown dataset '{ds}' — target will be used but no default exists")

    sem = asyncio.Semaphore(args.max_concurrent)
    stop_event = asyncio.Event()

    print("=" * 60)
    print("Memory Bank Fill — Per-Dataset Targets")
    print("=" * 60)

    # Step 1: Test safety — load all test IDs and verify no overlap
    print("\n[1/3] Loading test IDs for safety check...")
    test_ids = _load_all_test_ids()
    for ds in ds_list:
        _verify_no_test_leak(ds, test_ids)

    # Step 2: Pre-flight counts
    print("\n[2/3] Pre-flight counts:")
    for ds in ds_list:
        bank_path = BANK_DIR / ds
        ids = _get_collected_ids(bank_path / "meta.json")
        errs = _count_errors(bank_path)
        need = max(0, targets[ds] - len(ids))
        status = "DONE" if need == 0 else f"NEED +{need}"
        print(f"  {ds:14s} {len(ids):4d}/{targets[ds]} entries  {errs:3d} errors  -> {status}")

    # Filter out already-complete datasets
    active_ds = [
        ds for ds in ds_list
        if max(0, targets[ds] - len(_get_collected_ids(BANK_DIR / ds / "meta.json"))) > 0
    ]
    if not active_ds:
        print("\nAll datasets at target. Nothing to do.")
        return

    # Step 3: Fill
    print(f"\n[3/3] Filling {len(active_ds)} datasets (concurrency={args.max_concurrent})...")
    print()

    progresses: dict[str, Progress] = {ds: Progress(dataset=ds) for ds in active_ds}

    reporter = asyncio.create_task(_progress_reporter(progresses, stop_event))

    fill_tasks = [
        asyncio.create_task(
            fill_dataset(ds, ds, targets[ds], progresses[ds], sem, args.max_concurrent, args.buffer_factor)
        )
        for ds in active_ds
    ]
    await asyncio.gather(*fill_tasks, return_exceptions=True)

    stop_event.set()
    await reporter

    # Final report
    print("\n" + "=" * 60)
    print("FINAL")
    print("=" * 60)
    total_written = 0
    for ds in ds_list:
        ids = _get_collected_ids(BANK_DIR / ds / "meta.json")
        errs = _count_errors(BANK_DIR / ds)
        p = progresses.get(ds)
        if p:
            total_written += p.written
            shortfall = max(0, targets[ds] - len(ids))
            status = "DONE" if shortfall == 0 else f"SHORT {shortfall}"
            print(f"  {ds:14s} {len(ids):4d}/{targets[ds]} entries  {errs:3d} errors  "
                  f"(this run: +{p.written}w +{p.skipped}s +{p.errors}e)  {status}")
        else:
            shortfall = max(0, targets[ds] - len(ids))
            status = "DONE" if shortfall == 0 else f"SHORT {shortfall}"
            print(f"  {ds:14s} {len(ids):4d}/{targets[ds]} entries  {errs:3d} errors  {status}")
    print(f"\nTotal written this run: {total_written}")
    print(f"Total 429 rate-limit hits: {sum(p.rate_limits for p in progresses.values())}")


def main() -> None:
    args = _parse_args()
    if args.config:
        os.environ["FINACUMEN_CONFIG_PATH"] = os.path.abspath(args.config)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
