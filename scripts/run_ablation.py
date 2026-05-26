"""Ablation experiment: 5 configs x N questions.

Configs:
  A  - no memory (baseline)
  B1 - Q + A + Experience (no annotation)
  B2 - Q + A + Annotation (no experience)
  B3 - Annotation only (self-contained)
  B4 - Q + A + E + Annotation (all fields)

Phase 1: annotate(all, fields=B4, filter_useful=True) -> keeps only useful entries.
B1/B2/B3/B4 each get their own annotate call with config-specific fields_shown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from finacumen.ft.dataset.adapters import load_dataset
from finacumen.ft.variant.finacumen import FinAcumenVariant, _build_problem_text
from finacumen.ft.logger import logger
from finacumen.fm.retrieve import retrieve
from finacumen.fm.relevance import annotate as relevance_annotate

_THINK_RE = re.compile(r"<Think>\s*(.*?)\s*</Think>", re.DOTALL | re.IGNORECASE)

FIELDS_SHOWN = {
    "B1": "Question + Answer + Experience (Findings + Cautions)",
    "B2": "Question + Answer + Annotation (MUST/DO NOT directives)",
    "B3": "Annotation only (self-contained MUST/DO NOT directives)",
    "B4": "Question + Answer + Experience + Annotation",
}


def _build_variant_ns(config: str, output_dir: Path, memory_dir: Path | str = ""):
    if not memory_dir:
        memory_dir = _REPO_ROOT / "memory"
    memory_dir = Path(memory_dir).resolve()
    return argparse.Namespace(
        variant="finacumen", dataset="bizbench", memory_mode="test",
        memory_dir=memory_dir, memory_strategy="C", memory_k_max=3,
        use_relevance=(config != "A"), memory_config=config,
        collect_concurrency=1, memory_root=None,
        retrieval_file=None, output_dir=output_dir,
    )


async def run_one(variant, target: dict) -> dict:
    result = await variant.solve(target)
    return {"target_id": target.get("id", ""), "question": target.get("question", "")[:200],
            "gold": str(target.get("gold_answer", "")), "predicted": result.get("final_answer", ""),
            "correct": result.get("correct", False), "memory_mode": result.get("memory_mode", "?"),
            "memory_entry_ids": result.get("memory_entry_ids", []), "steps": int(result.get("solve_steps", 0) or 0)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--datasets", default="bizbench,finmmr_easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--target-ids", default=None)
    args = parser.parse_args()

    ds_names = [d.strip() for d in args.datasets.split(",")]
    per_ds = max(1, args.limit // len(ds_names))
    rng = random.Random(args.seed)

    targets = []
    for ds_name in ds_names:
        ds_targets = load_dataset(ds_name, split="test")
        rng.shuffle(ds_targets)
        sampled = ds_targets[:per_ds]
        for t in sampled:
            t["_dataset"] = ds_name
        targets.extend(sampled)
    rng.shuffle(targets)

    if args.target_ids:
        # Reload all targets for targeted filtering
        all_targets = []
        for ds_name in ds_names:
            ds_all = load_dataset(ds_name, split="test")
            for t in ds_all:
                t["_dataset"] = ds_name
            all_targets.extend(ds_all)
        tid_set = set(args.target_ids.split(","))
        targets = [t for t in all_targets if str(t.get("id", "")) in tid_set]
        logger.info(f"Filtered to {len(targets)} targets by --target-ids")

    targets = targets[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = Path(args.memory_dir).resolve()

    # Load pre-computed retrieval cache if provided
    retrieval_cache: dict[str, dict] = {}
    if args.retrieval_file:
        rp = Path(args.retrieval_file)
        for line in rp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            retrieval_cache[row["target_id"]] = row
        logger.info(f"Loaded {len(retrieval_cache)} retrieval cache entries from {args.retrieval_file}")

    # Build variants
    va = FinAcumenVariant(_build_variant_ns("A", out_dir, mem_dir))
    vb1 = FinAcumenVariant(_build_variant_ns("B1", out_dir, mem_dir))
    vb2 = FinAcumenVariant(_build_variant_ns("B2", out_dir, mem_dir))
    vb3 = FinAcumenVariant(_build_variant_ns("B3", out_dir, mem_dir))
    vb4 = FinAcumenVariant(_build_variant_ns("B4", out_dir, mem_dir))

    results_all = []
    for i, t in enumerate(targets):
        tid = t.get("id", f"q{i}")
        ds = t.get("_dataset", "?")
        q = t.get("question", "")[:100]
        gold = str(t.get("gold_answer", ""))
        logger.info(f"Q{i+1}/{args.limit}: {tid} [{ds}] | Q: {q} | Gold: {gold}")

        # Retrieve and Phase 1 annotate (common for all B configs)
        if tid in retrieval_cache:
            cached = retrieval_cache[tid]
            retrieval = {
                "mode": cached.get("mode", "no-memory"),
                "experiences": cached.get("experiences", []),
                "scores": cached.get("scores", []),
            }
        else:
            retrieval = retrieve(t, k_max=3, bank_dir=Path(mem_dir))
        entries_raw = retrieval.get("experiences", [])
        mem_mode = retrieval.get("mode", "no-memory")
        entries_phase1 = []
        phase1_judgments = []

        if mem_mode == "with-memory" and len(entries_raw) >= 1:
            problem_text = _build_problem_text(t)
            entries_phase1 = await relevance_annotate(
                problem_text, entries_raw,
                fields_shown=FIELDS_SHOWN["B4"],
                filter_useful=True,
            )
            for e in entries_phase1:
                phase1_judgments.append({
                    "useful": e.get("_useful", True),
                    "annotation": e.get("_annotation", "")[:200],
                })

        row = {"qid": tid, "dataset": ds, "question": q, "gold": gold,
               "mem_mode": mem_mode, "n_retrieved": len(entries_raw),
               "n_phase1": len(entries_phase1), "phase1": phase1_judgments}

        # Run A (no memory)
        ra = await run_one(va, t)
        row["A"] = {"answer": ra["predicted"], "correct": ra["correct"], "steps": ra["steps"]}
        logger.info(f"  A      -> '{str(ra['predicted'])[:50]}' {'OK' if ra['correct'] else 'XX'}")

        # Run B1 (Q+A+E)
        if entries_phase1:
            vb1._override_entries = entries_phase1
        rb1 = await run_one(vb1, t)
        row["B1"] = {"answer": rb1["predicted"], "correct": rb1["correct"], "steps": rb1["steps"],
                      "entries": len(rb1.get("memory_entry_ids", []))}
        logger.info(f"  B1(QAE)-> '{str(rb1['predicted'])[:50]}' {'OK' if rb1['correct'] else 'XX'}")

        # Run B2 (Q+A+Annotation)
        if entries_raw and mem_mode == "with-memory":
            entries_b2 = await relevance_annotate(
                _build_problem_text(t), entries_raw,
                fields_shown=FIELDS_SHOWN["B2"], filter_useful=True,
            )
            vb2._override_entries = entries_b2 if entries_b2 else entries_raw
        else:
            vb2._override_entries = entries_raw
        rb2 = await run_one(vb2, t)
        row["B2"] = {"answer": rb2["predicted"], "correct": rb2["correct"], "steps": rb2["steps"],
                      "entries": len(rb2.get("memory_entry_ids", []))}
        logger.info(f"  B2(QA+)-> '{str(rb2['predicted'])[:50]}' {'OK' if rb2['correct'] else 'XX'}")

        # Run B3 (Annotation only)
        if entries_raw and mem_mode == "with-memory":
            entries_b3 = await relevance_annotate(
                _build_problem_text(t), entries_raw,
                fields_shown=FIELDS_SHOWN["B3"], filter_useful=True,
            )
            vb3._override_entries = entries_b3 if entries_b3 else entries_raw
        else:
            vb3._override_entries = entries_raw
        rb3 = await run_one(vb3, t)
        row["B3"] = {"answer": rb3["predicted"], "correct": rb3["correct"], "steps": rb3["steps"],
                      "entries": len(rb3.get("memory_entry_ids", []))}
        logger.info(f"  B3(Ao )-> '{str(rb3['predicted'])[:50]}' {'OK' if rb3['correct'] else 'XX'}")

        # Run B4 (All)
        if entries_phase1:
            vb4._override_entries = entries_phase1
        rb4 = await run_one(vb4, t)
        row["B4"] = {"answer": rb4["predicted"], "correct": rb4["correct"], "steps": rb4["steps"],
                      "entries": len(rb4.get("memory_entry_ids", []))}
        logger.info(f"  B4(All)-> '{str(rb4['predicted'])[:50]}' {'OK' if rb4['correct'] else 'XX'}")

        results_all.append(row)

    # Write JSONL
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results_all:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Generate analysis
    configs = ["A", "B1", "B2", "B3", "B4"]
    lines = ["# Ablation Experiment", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"Datasets: {args.datasets} | Sample: {args.limit} questions (seed={args.seed})", ""]

    # Summary table
    lines.append("## Summary")
    lines.append("| Config | Accuracy | Avg Entries | Avg Steps |")
    lines.append("|--------|----------|-------------|-----------|")
    for cfg in configs:
        correct = sum(1 for r in results_all if r.get(cfg, {}).get("correct"))
        entries = [r.get(cfg, {}).get("entries", 0) for r in results_all]
        steps = [r.get(cfg, {}).get("steps", 0) for r in results_all]
        avg_entries = sum(entries) / len(entries) if entries else 0
        avg_steps = sum(steps) / len(steps) if steps else 0
        lines.append(f"| {cfg} | {correct}/{args.limit} ({correct/args.limit:.0%}) | {avg_entries:.1f} | {avg_steps:.1f} |")

    # Per-question analysis
    lines.extend(["", "## Per-Question Analysis", ""])
    for i, r in enumerate(results_all):
        lines.append(f"### Q{i+1}: {r['question'][:100]}")
        lines.append(f"**Gold**: {r['gold']} | **Mem**: {r['mem_mode']} | "
                     f"Retrieved: {r['n_retrieved']} | Phase1 kept: {r['n_phase1']}")
        lines.append("")

        if r["phase1"]:
            lines.append("**Phase 1 judgments:**")
            for j, pj in enumerate(r["phase1"]):
                useful = "\u2713 KEPT" if pj["useful"] else "\u2717 DROPPED"
                lines.append(f"  Entry {j}: {useful} | {pj['annotation'][:120]}")
            lines.append("")

        lines.append("| Config | Answer | Correct | Steps | Entries |")
        lines.append("|--------|--------|---------|-------|---------|")
        for cfg in configs:
            d = r.get(cfg, {})
            ok = "\u2713" if d.get("correct") else "\u2717"
            lines.append(f"| {cfg} | {str(d.get('answer',''))[:60]} | {ok} | {d.get('steps',0)} | {d.get('entries','-')} |")
        lines.append("\n---\n")

    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report: {out_dir / 'analysis.md'}")
    for cfg in configs:
        correct = sum(1 for r in results_all if r.get(cfg, {}).get("correct"))
        logger.info(f"  {cfg}: {correct}/{args.limit}")


if __name__ == "__main__":
    asyncio.run(main())
