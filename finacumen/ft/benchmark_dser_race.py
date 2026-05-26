#!/usr/bin/env python3
"""
benchmark.py — unified harness for FinAcumen variant evaluation.

Loads a named dataset, routes each target through a variant
(baseline-raw, ft-only, finacumen), and writes per-target
results as JSONL.

Usage:
    python -m finacumen.ft.benchmark --variant finacumen --dataset fintmm --limit 30
    python -m finacumen.ft.benchmark --variant baseline-raw --dataset bizbench
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from dotenv import load_dotenv

from finacumen.ft.dataset.adapters import load_dataset
from finacumen.ft.logger import logger
from finacumen.ft.paths import DEFAULT_MEMORY_BANK_DIR, FINACUMEN_PROJECT_ROOT, MEMORY_ROOT, REPO_ROOT
from finacumen.ft.variant.base import DSERVariant

# Load .env — repository root then finacumen/.
for _env in (
    REPO_ROOT / ".env",
    FINACUMEN_PROJECT_ROOT / ".env",
):
    if _env.exists():
        load_dotenv(_env)

RESULTS_ROOT_DEFAULT = FINACUMEN_PROJECT_ROOT / "results"


_VARIANT = "finacumen.ft.variant"
VARIANT_MODULES = {
    "baseline-raw": f"{_VARIANT}.baseline_raw",
    "ft-only":      f"{_VARIANT}.ft_only",
    "finacumen":    f"{_VARIANT}.finacumen",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FinAcumen benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_MODULES.keys()))
    p.add_argument(
        "--dataset", required=True,
        choices=["bizbench", "finmmr_easy", "finmmr_medium", "finmmr_hard", "fintmm", "finmme"],
    )
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: results/<dataset>/<variant>/")
    p.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_BANK_DIR,
                   help="Bank directory containing meta.json + emb.npy.")
    p.add_argument("--memory-root", type=Path, default=MEMORY_ROOT,
                   help="Root directory for auto-discovering additional bank directories.")
    p.add_argument("--memory-k-max", type=int, default=3,
                   help="Max experiences injected per query.")
    p.add_argument("--use-relevance", action="store_true",
                    help="Enable LLM relevance review on retrieved memory entries.")
    p.add_argument("--relevance-filter", action="store_true", default=False,
                    help="Filter out not-useful memory entries after relevance annotation (default: keep all, annotate only).")
    p.add_argument("--collect-concurrency", type=int, default=4,
                   help="Async collect semaphore bound.")
    p.add_argument("--memory-mode", type=str, default="train",
                    choices=["train", "test", "eval"],
                    help="train = retrieve + collect; test = retrieve only + stats bump; eval = retrieve only (fully read-only, safe for parallel).")
    p.add_argument("--memory-strategy", type=str, default="E",
                    choices=["A", "B", "C", "D", "E"],
                    help="Injection strategy: A=Q-only, B=Q+A, C=Q+A+Experience, D=Q+A+Annotation, E=Q+A+Exp+Annotation.")
    p.add_argument("--memory-config", type=str, default=None,
                    choices=["B1", "B2", "B3", "B4"],
                    help="System-prompt config (auto-derived from strategy if not set).")
    p.add_argument("--dataset-split", type=str, default="test",
                    choices=["train", "test"],
                    help="Which dataset split to load.")
    p.add_argument("--task", type=str, default=None,
                    help="Filter by task type (bizbench is SEC-NUM only)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the first question and exit without API calls")
    return p.parse_args()


def build_variant(args: argparse.Namespace) -> DSERVariant:
    module_path = VARIANT_MODULES[args.variant]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Variant module '{module_path}' not found. ({e})"
        )
    if not hasattr(module, "build_variant"):
        raise SystemExit(
            f"Variant module '{module_path}' must expose a build_variant(args) factory."
        )
    return module.build_variant(args)


async def run_benchmark(args: argparse.Namespace) -> None:
    targets = load_dataset(args.dataset, split=args.dataset_split, task_filter=args.task)
    if args.offset:
        targets = targets[args.offset:]
    if args.limit:
        targets = targets[:args.limit]

    if args.dry_run:
        _dry_run(targets[0], args)
        return

    output_dir = args.output_dir or (RESULTS_ROOT_DEFAULT / args.dataset / args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["target_id"])
            except Exception:
                pass
    pending = [t for t in targets if t["id"] not in done_ids]

    logger.info(
        f"Benchmark | variant={args.variant} dataset={args.dataset} "
        f"| targets={len(targets)} pending={len(pending)}"
    )

    variant = build_variant(args)
    open_mode = "a" if args.resume and results_path.exists() else "w"
    n_correct = 0
    total_seen = len(targets) - len(pending)
    if open_mode == "a":
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("correct"):
                    n_correct += 1
            except Exception:
                pass

    with open(results_path, open_mode, encoding="utf-8") as out:
        for i, target in enumerate(pending, 1):
            try:
                result = await variant.solve(target)
            except Exception as e:
                logger.exception(f"Variant failed on {target['id']}: {e}")
                result = DSERVariant.build_result(
                    target, "", extras={"error": str(e)}, latency_sec=0.0
                )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            total_seen += 1
            if result.get("correct"):
                n_correct += 1
            status = "OK" if result.get("correct") else "XX"
            logger.info(
                f"  [{i}/{len(pending)}] {target['id']} -> {str(result.get('final_answer', ''))!r} "
                f"{status} | acc = {n_correct}/{total_seen} = {n_correct/max(total_seen,1):.3f} "
                f"| {result.get('latency_sec', 0)}s"
            )

    if hasattr(variant, "finalize"):
        try:
            await variant.finalize()
        except Exception as e:
            logger.warning(f"variant.finalize() raised: {type(e).__name__}: {e}")

    n_total = total_seen
    accuracy = n_correct / max(n_total, 1)

    # ── fintmm: LLM-as-judge re-score ──────────────────────────────────────
    if args.dataset == "fintmm" and n_total > 0:
        try:
            from finacumen.ft.eval.fintmm_eval import _make_judge_client, llm_judge
            client, model = _make_judge_client()
            logger.info(f"LLM judge re-scoring {n_total} fintmm results with {model}...")
            updated = 0
            results_lines = results_path.read_text(encoding="utf-8").splitlines()
            all_results = [json.loads(l) for l in results_lines if l.strip()]
            for r in all_results:
                judged = await llm_judge(
                    question=r.get("question", ""),
                    gold=r.get("gold_answer", ""),
                    pred=r.get("final_answer", ""),
                    client=client,
                    model=model,
                )
                if judged != r.get("correct", False):
                    r["correct"] = judged
                    r["llm_judge_overridden"] = True
                    updated += 1
                else:
                    r["llm_judge_overridden"] = False
            # Rewrite results
            with open(results_path, "w", encoding="utf-8") as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_correct = sum(1 for r in all_results if r.get("correct"))
            accuracy = n_correct / n_total
            logger.info(f"  LLM judge updated {updated} results → {n_correct}/{n_total} = {accuracy:.4f}")
        except Exception as e:
            logger.warning(f"LLM judge re-score failed: {e}")

    summary = {
        "variant": args.variant,
        "dataset": args.dataset,
        "n": n_total,
        "correct": n_correct,
        "accuracy": round(accuracy, 4),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"\n{args.variant} / {args.dataset}: {n_correct}/{n_total} = {accuracy:.4f}")
    logger.info(f"Results: {results_path}")


def _dry_run(target: dict, args: argparse.Namespace) -> None:
    logger.info(f"Dry-run for {target['id']} (dataset={target.get('dataset')})")
    logger.info(f"Question: {target.get('question', '')[:200]}")
    logger.info(f"Answer type: {target.get('answer_type')}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
