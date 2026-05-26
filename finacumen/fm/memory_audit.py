"""Memory audit — automated metrics + LLM quality review.

Usage: python -m finacumen.fm.memory_audit [bank_dir] [--samples N]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from finacumen.fm.bank import load_meta
from finacumen.ft.llm import LLM

# ── Sampling config per dataset ──────────────────────────────────────────────

_SAMPLE_COUNTS = {
    "bizbench": 10,
    "finmmr_easy": 5,
    "finmmr_medium": 5,
    "finmmr_hard": 5,
    "fintmm": 10,
    "finmme": 10,
}

# ── LLM review prompt ────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
<Evaluate>
Review this memory bank entry and score it on 4 dimensions. Return ONLY a JSON object.

<Entry>
  <Question>{question}</Question>
  <GoldAnswer>{gold_answer}</GoldAnswer>
  <Findings>{findings}</Findings>
  <Cautions>{cautions}</Cautions>
</Entry>

Scoring rubric (0-4 each):
1. Generalization: Does this entry leak specific years/entity names?
   - 0: Full of source-specific values (company names, years, numbers)
   - 2: Some generalization but still has specific references
   - 4: Fully generic, reusable on any problem of the same shape

2. Actionability: Can an agent apply this on first read?
   - 0: Vague imperative ("be careful", "pay attention") — no concrete action
   - 2: Has some actionable steps but missing critical details
   - 4: Immediately actionable — agent knows exactly what to do

3. Format compliance: Findings and Cautions properly separated?
   - 0: Wrong format (guidance in Cautions, guard rules in Findings, or all mixed)
   - 1: One side empty (all Findings no Cautions, or vice versa)
   - 2: Both sides properly populated with appropriate content

4. Overall quality: How useful is this entry?
   - 0: Harmful (would mislead the agent)
   - 1: Useless (too vague to apply, or describes a failure without a rule)
   - 2: Marginal (barely helpful)
   - 3: Useful (clear guidance for a specific scenario)
   - 4: Excellent (would genuinely improve agent performance)

Return:
<Answer>
{{"generalization": 0-4, "actionability": 0-4, "format": 0-2, "overall": 0-4, "note": "one-sentence comment"}}
</Answer>
</Evaluate>"""


# ── Automated metrics ────────────────────────────────────────────────────────

def _auto_metrics(entries: list[dict]) -> dict:
    if not entries:
        return {"total": 0}

    ds_counts: dict[str, int] = {}
    has_findings = 0
    has_cautions = 0
    both_empty = 0
    exp_lens = []
    analysis_lens = []
    use_counts = []

    for e in entries:
        ds = e.get("source", {}).get("dataset", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

        exp = e.get("experience", {})
        f = exp.get("findings", [])
        c = exp.get("cautions", [])
        if f:
            has_findings += 1
        if c:
            has_cautions += 1
        if not f and not c:
            both_empty += 1

        exp_lens.append(sum(len(s) for s in f + c))
        analysis_lens.append(len(str(e.get("analysis", ""))))

        st = e.get("stats", {})
        use_counts.append(st.get("use_count", 0))

    n = len(entries)
    return {
        "total": n,
        "per_dataset": ds_counts,
        "has_findings": has_findings,
        "has_cautions": has_cautions,
        "both_empty": both_empty,
        "avg_experience_chars": round(sum(exp_lens) / n, 1) if n else 0,
        "avg_analysis_chars": round(sum(analysis_lens) / n, 1) if n else 0,
        "max_use_count": max(use_counts),
        "avg_use_count": round(sum(use_counts) / n, 2) if n else 0,
        "dead_entries": sum(1 for u in use_counts if u == 0),
    }


# ── LLM review ───────────────────────────────────────────────────────────────

def _sample_entries(entries: list[dict], n: int) -> list[dict]:
    """Stratified sample: try to get balanced per-dataset."""
    by_ds: dict[str, list[dict]] = {}
    for e in entries:
        ds = e.get("source", {}).get("dataset", "unknown")
        by_ds.setdefault(ds, []).append(e)

    sampled = []
    for ds, target_n in _SAMPLE_COUNTS.items():
        pool = by_ds.get(ds, [])
        if pool:
            k = min(target_n, len(pool))
            sampled.extend(random.sample(pool, k))
    # If not enough, fill with random from remaining
    if len(sampled) < n:
        remaining = [e for e in entries if e not in set(id(x) for x in sampled)]
        extra = min(n - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra))
    return sampled[:n]


async def _judge_entry(entry: dict) -> dict:
    exp = entry.get("experience", {})
    findings_text = "; ".join(exp.get("findings", [])[:3])
    cautions_text = "; ".join(exp.get("cautions", [])[:3])
    prompt = _JUDGE_PROMPT.format(
        question=entry.get("question", "")[:300],
        gold_answer=entry.get("gold_answer", ""),
        findings=findings_text or "(none)",
        cautions=cautions_text or "(none)",
    )

    llm = LLM("memory_judge")
    try:
        resp = await llm.ask(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        import re
        m = re.search(r"<Answer>\s*(\{.*?\})\s*</Answer>", resp or "", re.DOTALL)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return {"generalization": -1, "actionability": -1, "format": -1, "overall": -1, "note": "judge failed"}


# ── Report generation ────────────────────────────────────────────────────────

def _build_report(
    meta_path: Path,
    metrics: dict,
    reviews: list[dict],
    review_entries: list[dict],
) -> str:
    lines = [
        "# Memory Quality Report",
        f"\n**Bank**: `{meta_path}`",
        f"**Total entries**: {metrics['total']}",
        "",
        "## Automated Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total entries | {metrics['total']} |",
        f"| Dead entries (use_count=0) | {metrics['dead_entries']} |",
        f"| Max use_count | {metrics['max_use_count']} |",
        f"| Avg use_count | {metrics['avg_use_count']} |",
        f"| Entries with Findings | {metrics['has_findings']} |",
        f"| Entries with Cautions | {metrics['has_cautions']} |",
        f"| Empty entries (no Findings or Cautions) | {metrics['both_empty']} |",
        f"| Avg experience chars | {metrics['avg_experience_chars']} |",
        f"| Avg analysis chars | {metrics['avg_analysis_chars']} |",
        "",
        "## Per-Dataset Distribution",
        "",
    ]
    for ds, cnt in sorted(metrics.get("per_dataset", {}).items()):
        lines.append(f"| {ds} | {cnt} |")

    lines.extend([
        "",
        "## LLM Quality Review (deepseek-v4-pro)",
        "",
        "| # | Dataset | Generalization | Actionability | Format | Overall | Note |",
        "|---|---------|---------------|--------------|--------|---------|------|",
    ])

    scores = {"generalization": [], "actionability": [], "format": [], "overall": []}
    for i, (review, entry) in enumerate(zip(reviews, review_entries), 1):
        ds = entry.get("source", {}).get("dataset", "?")
        g = review.get("generalization", "?")
        a = review.get("actionability", "?")
        f = review.get("format", "?")
        o = review.get("overall", "?")
        n = review.get("note", "").replace("|", "/")
        lines.append(f"| {i} | {ds} | {g} | {a} | {f} | {o} | {n} |")
        for k in scores:
            if isinstance(review.get(k), (int, float)) and review[k] >= 0:
                scores[k].append(review[k])

    if any(scores[k] for k in scores):
        lines.extend([
            "",
            "| Dimension | Mean |",
            "|-----------|------|",
        ])
        for k, v in scores.items():
            if v:
                lines.append(f"| {k} | {round(sum(v)/len(v), 2)} |")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_audit(bank_dir: str, samples: int = 45) -> str:
    meta_path = Path(bank_dir) / "meta.json"
    entries = load_meta(meta_path)
    if not entries:
        return "# Memory Quality Report\n\n**No entries found.**"

    # 1. Automated metrics
    metrics = _auto_metrics(entries)

    # 2. LLM review
    review_entries = _sample_entries(entries, samples)
    print(f"Audit: reviewing {len(review_entries)} entries...")
    reviews = [_judge_entry(e) for e in review_entries]
    import asyncio
    reviews = await asyncio.gather(*reviews)

    # 3. Build report
    report = _build_report(meta_path, metrics, reviews, review_entries)

    report_path = Path(bank_dir) / "quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written: {report_path}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bank_dir", nargs="?", default=None, help="Bank directory")
    p.add_argument("--samples", type=int, default=45, help="LLM review samples")
    args = p.parse_args()
    if args.bank_dir is None:
        args.bank_dir = Path(__file__).resolve().parents[2] / "finacumen" / "memory" / "main"
    else:
        args.bank_dir = Path(args.bank_dir).resolve()

    import asyncio
    asyncio.run(run_audit(args.bank_dir, args.samples))


if __name__ == "__main__":
    main()
