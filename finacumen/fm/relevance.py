"""Memory annotator — writes personalized guidance for each entry. Never filters by default.

Can accept `fields_shown` to tailor annotations to what the solving agent will see.
Now outputs `useful: true/false` for Phase 1 entry-level judgment.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger


class AnnotationIncompleteError(RuntimeError):
    """LLM returned fewer results than entries — skip this question."""
    pass

ANNOTATE_PROMPT = """\
<Purpose>
You are a memory annotator. For each past experience entry, write a DIRECTIVE annotation telling the solving agent exactly what to do and what to avoid. ALL entries will be shown — your job is to annotate, not to filter.
</Purpose>

<Context>
The solving agent will see entries formatted as: {fields_shown}
Tailor your annotation to what context the agent will have.
If only Annotation is shown, make it fully self-contained (mention the current problem's specific entities, values, and expected steps).
</Context>

<Current_Problem>
{problem_text}
</Current_Problem>

<Entries>
{entries_text}
</Entries>

<Task>
For EVERY entry, write 1-3 directive sentences using MUST / DO NOT.
- Extract the entry's methodology (extraction pattern, computation logic, verification step)
  and adapt it to the current problem's specific row labels, column names, and numeric values.
- NEVER write an empty annotation.

<AMBITUITY_WARNING — Financial Vocabulary>
Identical financial terms often carry DIFFERENT meanings depending on statement context:
  "Deferred income taxes" on a cash flow statement   = period CHANGE in deferred tax
  "Deferred tax assets" in a balance sheet note       = cumulative ASSET balance
  "Revenue" vs "Net sales" vs "Operating revenue"      = may differ in deduction scope
  "Term" in a lease disclosure vs "term" in a loan     = different contractual meaning

When the memory entry uses vocabulary similar to but semantically different from
the current problem, the annotation MUST explicitly warn about the distinction:
  RIGHT: "MUST extract the value from the 'Deferred income taxes' row in the cash flow
         table. DO NOT assume this refers to a deferred tax asset balance — the cash
         flow line represents the period-over-period change, not the cumulative."
  WRONG: "This entry is about deferred tax assets, not relevant."
</AMBITUITY_WARNING>

<Generalization_Rule>
Map the memory's METHODOLOGY, not its literal entity names:
  WRONG: "MUST extract 'Impact of curtailments' from 2021 column"
  RIGHT: "MUST extract the value from the matching row label in the current table
          for the relevant period, then compute as the question requires."

Set "useful": false ONLY when the task TYPE is fundamentally different
(chart reading vs. table lookup vs. narrative extraction). Otherwise, useful MUST be true.
</Generalization_Rule>
</Task>

<Output>
Return ONLY:
<Answer>
[
  {{
    "i": 0,
    "useful": true,
    "annotation": "MUST extract values from the Context table row labels that match the question, assign as python variables. DO NOT compute without verifying row-label match."
  }},
  {{
    "i": 1,
    "useful": true,
    "annotation": "MUST verify whether the table row's header matches the question's exact term. DO NOT conflate similar-sounding financial terms — confirm the statement type (cash flow vs. balance sheet) before extracting."
  }}
]
</Answer>
</Output>"""


async def annotate(
    problem_text: str,
    entries: list[dict],
    fields_shown: str = "Question + Answer + Experience + Annotation",
    filter_useful: bool = False,
) -> list[dict]:
    """Annotate entries with personalized guidance.

    Args:
        problem_text: Current problem text (truncated to 2000 chars).
        entries: Retrieved memory entries.
        fields_shown: Description of what fields the solving agent will see.
        filter_useful: If True, drop entries where useful==False.

    Returns:
        All entries (if filter_useful=False) or only useful ones (if filter_useful=True).
        Each entry gets `_annotation` and `_useful` fields.
        Falls back to original entries on any failure.
    """
    if not entries:
        return []

    entries_text = _format_entries(entries)
    problem_text_short = problem_text[:2000]
    prompt = ANNOTATE_PROMPT.format(
        fields_shown=fields_shown,
        problem_text=problem_text_short,
        entries_text=entries_text,
    )

    try:
        llm = LLM("dser")
        response = await llm.ask(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        )
        results = _parse_json_array(response or "")
    except Exception:
        logger.warning("annotate: LLM call failed, falling back to raw entries")
        return entries

    if not results:
        logger.warning("annotate: JSON parse failed, falling back to raw entries")
        return entries

    # Build result map by index (LLM may return fewer results than entries)
    result_map: dict[int, dict] = {}
    for r in (results or []):
        if isinstance(r, dict):
            result_map[r.get("i", -1)] = r

    # Check: did LLM return results for ALL entries?
    missing = [i for i in range(len(entries)) if i not in result_map]
    if missing:
        raise AnnotationIncompleteError(
            f"LLM returned {len(results)} results for {len(entries)} entries. "
            f"Missing indices: {missing}."
        )

    # All present — assign annotation to each entry
    for i, entry in enumerate(entries):
        r = result_map[i]
        entry["_useful"] = r.get("useful", True)
        entry["_annotation"] = r.get("annotation", "")
        if not entry["_annotation"].strip():
            raise AnnotationIncompleteError(
                f"annotate: entry {i} has empty _annotation. "
                f"useful={entry.get('_useful')}."
            )

    n_annotated = sum(1 for e in entries if e.get("_annotation", "").strip())
    n_useful = sum(1 for e in entries if e.get("_useful", True))
    logger.info(f"annotate: {n_annotated}/{len(entries)} annotated, {n_useful}/{len(entries)} useful")

    if filter_useful:
        filtered = [e for e in entries if e.get("_useful", True)]
        if not filtered and entries:
            logger.info("annotate: all entries marked not-useful, keeping top-1")
            filtered = [entries[0]]
        return filtered

    return entries


def _format_entries(entries: list[dict]) -> str:
    parts: list[str] = []
    for i, e in enumerate(entries):
        exp = e.get("experience", "")
        if isinstance(exp, dict):
            lines: list[str] = []
            findings = exp.get("findings", [])
            if findings:
                lines.append("Findings:")
                lines.extend(f"  - {f}" for f in findings)
            cautions = exp.get("cautions", [])
            if cautions:
                lines.append("Cautions:")
                lines.extend(f"  - {c}" for c in cautions)
            exp_str = "\n".join(lines)
        else:
            exp_str = str(exp)
        parts.append(
            f"<Entry_{i}>\n"
            f"  <OriginalQuestion>{e.get('question', '')[:800]}</OriginalQuestion>\n"
            f"  <Experience>{exp_str[:1200]}</Experience>\n"
            f"</Entry_{i}>"
        )
    return "\n".join(parts)


_ANSWER_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.DOTALL | re.IGNORECASE)


def _parse_json_array(text: str) -> Optional[list]:
    m = _ANSWER_RE.search(text)
    inner = (m.group(1) if m else text).strip()
    if inner.startswith("```"):
        inner = re.sub(r"^```\w*\s*", "", inner)
        inner = re.sub(r"\s*```$", "", inner)
    try:
        result = json.loads(inner)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None
