"""Retrieve — single-stage embedding retrieval with global matrix multiply.

Resolves all embeddings (query and bank entries) from pre-computed
datasets/*_emb.npy via DatasetEmbeddingManager. Zero API calls.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from finacumen.fm.bank import (
    CONTEXT_WINDOW,
    K_MAX_DEFAULT,
    MIN_COSINE,
    load_meta,
)
from finacumen.fm.emb_manager import get_emb_manager

MATCH_CONTEXT_CAP = CONTEXT_WINDOW


def _build_match_text(question: str, context: str | None) -> str:
    """Build Scheme C matching text: question + context[:600], no instruction prefix."""
    text = question or ""
    if context:
        text += "\n\nContext: " + str(context)[:MATCH_CONTEXT_CAP]
    return text


# ── Public API ───────────────────────────────────────────────────────────────


def retrieve(
    target: dict,
    k_max: int = K_MAX_DEFAULT,
    bank_dir: Path | None = None,
    memory_root: Path | None = None,
) -> dict:
    """Retrieve up to k_max relevant experiences from memory banks.

    Scans bank_dir (primary) and memory_root/*/ for all meta.json directories.
    Resolves embeddings from pre-computed datasets/*_emb.npy via DatasetEmbeddingManager.
    Computes cosine in one matrix multiply across all entries.

    Returns:
        {"mode": "with-memory"|"no-memory", "experiences": [...], "scores": [...]}
    """
    # 1. Resolve query embedding from pre-computed datasets/*_emb.npy
    mgr = get_emb_manager()
    query_vec = mgr.resolve(target.get("id", ""))
    if query_vec is None:
        return _no_memory()
    query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)

    # 2. Collect all bank paths (meta.json only, emb.npy no longer required)
    bank_paths = _collect_meta_paths(bank_dir, memory_root)
    if not bank_paths:
        return _no_memory()

    # 3. Load all entries and resolve embeddings from pre-computed
    all_entries = []
    emb_vectors = []
    for bp in bank_paths:
        entries = load_meta(bp / "meta.json")
        for entry in entries:
            vec = mgr.resolve_entry(entry)
            if vec is not None:
                all_entries.append(entry)
                emb_vectors.append(vec)

    if not emb_vectors:
        return _no_memory()

    # 4. Single matrix multiply
    all_emb = np.stack(emb_vectors)                      # (N_total, D)
    scores = query_vec @ all_emb.T                        # (N_total,)

    # 5. Filter, sort, dedup, top-k
    mask = scores >= MIN_COSINE
    if not mask.any():
        return _no_memory()

    original_indices = np.where(mask)[0]
    masked_scores = scores[mask]
    order = np.argsort(masked_scores)[::-1]
    top_original_idx = original_indices[order[: k_max * 4]]
    candidates = [(all_entries[i], float(scores[i])) for i in top_original_idx]
    selected = _dedup(candidates, k_max)

    experiences = []
    result_scores = []
    for entry, score in selected:
        experiences.append({
            "source": entry.get("source", {}),
            "experience": entry.get("experience", ""),
            "question": entry.get("question", ""),
            "gold_answer": entry.get("gold_answer", ""),
            "image_paths": entry.get("image_paths", []),
            "analysis": entry.get("analysis", ""),
        })
        result_scores.append(score)

    return {
        "mode": "with-memory" if experiences else "no-memory",
        "experiences": experiences,
        "scores": result_scores,
    }


# ── Rendering ────────────────────────────────────────────────────────────────


def render_memory_block(
    experiences: list[dict],
    memory_image_indices: dict[int, list[int]] | None = None,
    strategy: str = "C",
    config: str = "B4",
) -> str:
    """Generate <Memory_Block> XML from retrieval results.

    Strategy controls base content (A/B/C).
    Config controls which fields are rendered:
      - "B1": Q + A + E (Experience only, no Annotation)
      - "B2": Q + A + Annotation (no Experience)
      - "B3": Annotation only (no Q, A, E)
      - "B4": Q + A + E + Annotation (all fields)
    """
    if not experiences:
        return ""

    guide_parts: list[str] = []

    # Generate Field_Guide based on strategy
    if strategy == "A":
        guide_parts = [
            "Each Entry contains a Question from a past similar problem.",
            "Use it to judge whether your current problem is in a similar scenario.",
        ]
    elif strategy == "B":
        guide_parts = [
            "Each Entry is a past experience:",
            "  - Question: the original problem (use to judge scenario similarity)",
            "  - Answer: the correct answer (format reference only, never copy value)",
        ]
    elif strategy == "C":
        guide_parts = [
            "Each Entry is a past experience:",
            "  - Question: the original problem (use to judge scenario similarity)",
            "  - Answer: the correct answer (format reference only, never copy value)",
            "  - Experience: distilled reusable rules organized as:",
            "    - Findings: patterns that succeeded → follow these strategies",
            "    - Cautions: guard rules from failures → apply if condition triggers",
        ]
    elif strategy == "D":
        guide_parts = [
            "Each Entry is a past experience:",
            "  - Question: the original problem (use to judge scenario similarity)",
            "  - Answer: the correct answer (format reference only, never copy value)",
            "  - Annotation: directive instructions (MUST/DO NOT) for THIS problem",
        ]
    else:  # E (default)
        guide_parts = [
            "Each Entry is a past experience:",
            "  - Question: the original problem (use to judge scenario similarity)",
            "  - Answer: the correct answer (format reference only, never copy value)",
            "  - Experience: distilled reusable rules organized as:",
            "    - Findings: patterns that succeeded → follow these strategies",
            "    - Cautions: guard rules from failures → apply if condition triggers",
            "  - Annotation: directive instructions (MUST/DO NOT) for THIS problem",
        ]

    lines = [
        "<Memory_Block>",
        "  <Field_Guide>",
        *[f"  {p}" for p in guide_parts],
        "  </Field_Guide>",
    ]
    refs = memory_image_indices or {}
    # Strategy controls which fields are rendered
    show_answer = strategy in ("B", "C", "D", "E")
    show_experience = strategy in ("C", "E")
    show_annotation = strategy in ("D", "E")
    for i, exp in enumerate(experiences):
        lines.append("  <Entry>")
        q = exp.get("question", "")
        gold_a = exp.get("gold_answer", "")
        lines.append(f"    <Question>{_escape_xml(q or '')}</Question>")
        if show_answer and gold_a:
            lines.append(f"    <Answer>{_escape_xml(str(gold_a))}</Answer>")

        if show_experience:
            lines.append("    <Experience>")
            experience = exp.get("experience", "")
            if isinstance(experience, dict):
                findings = experience.get("findings", [])
                cautions = experience.get("cautions", [])
                if findings:
                    lines.append("      <Findings>")
                    for f in findings:
                        lines.append(f"        - {_escape_xml(str(f))}")
                    lines.append("      </Findings>")
                if cautions:
                    lines.append("      <Cautions>")
                    for c in cautions:
                        lines.append(f"        - {_escape_xml(str(c))}")
                    lines.append("      </Cautions>")
            else:
                lines.append(f"      {_escape_xml(str(experience))}")
            lines.append("    </Experience>")

        if show_annotation:
            annotation = exp.get("_annotation", "")
            if annotation.strip():
                lines.append(f"    <Annotation>{_escape_xml(annotation)}</Annotation>")

        if i in refs:
            parts = ", #".join(str(n) for n in refs[i])
            lines.append(f"    <Images>See Image #{parts} below</Images>")
        lines.append("  </Entry>")
    lines.append("  <OptOut>If any entry above does NOT actually fit this problem, ignore it.</OptOut>")
    lines.append("</Memory_Block>")
    return "\n".join(lines)


def render_user_message(target: dict) -> str:
    """Build XML user message for tool-agent variants (ft-only, finacumen)."""
    ga = str(target.get("gold_answer", ""))
    atype = target.get("answer_type", "")
    parts = [
        "<Problem>",
        *([f"  <Context>{_escape_xml(str(target.get('context', '')))}</Context>"]
          if target.get("context") else []),
        *([f"  <Options>{_escape_xml(str(target.get('options', '')))}</Options>"]
          if target.get("options") else []),
        f"  <Question>{_escape_xml(target.get('question', ''))}</Question>",
        "</Problem>",
        "<Instruction>",
        *build_instruction(ga, answer_type=atype),
        "</Instruction>",
    ]
    return "\n".join(parts)


def inject_memory_into_message(
    experiences: list[dict],
    target: dict,
    memory_image_indices: dict[int, list[int]] | None = None,
    strategy: str = "C",
    config: str = "B4",
) -> str:
    """Return full user message with <Memory_Block> prepended.

    strategy: "A" (Q-only), "B" (Q+A), or "C" (Q+A+Experience).
    config: "B1" / "B2" / "B3" / "B4" controls which fields are rendered."""
    block = render_memory_block(experiences, memory_image_indices=memory_image_indices,
                                strategy=strategy, config=config)
    msg = render_user_message(target)
    if block:
        return block + "\n" + msg
    return msg


def render_baseline_message(target: dict, include_instruction: bool = True) -> str:
    """Build minimal XML user message for baseline-raw."""
    ga = str(target.get("gold_answer", ""))
    atype = target.get("answer_type", "")
    parts = [
        "<Problem>",
        *([f"  <Context>{_escape_xml(str(target.get('context', '')))}</Context>"]
          if target.get("context") else []),
        *([f"  <Options>{_escape_xml(str(target.get('options', '')))}</Options>"]
          if target.get("options") else []),
        f"  <Question>{_escape_xml(target.get('question', ''))}</Question>",
        "</Problem>",
    ]
    if include_instruction:
        parts.append("<Instruction>")
        parts.extend(build_instruction(ga, answer_type=atype))
        parts.append("Return the answer inside **Final Answer:** ...")
        parts.append("</Instruction>")
    return "\n".join(parts)


def render_baseline_ref_questions(experiences: list[dict]) -> str:
    """Render <Reference> block with only <Question> entries (Strategy A)."""
    if not experiences:
        return ""
    lines = ["<Reference>"]
    for exp in experiences:
        q = exp.get("question", "")
        lines.append(f"  <Question>{_escape_xml(q or '')}</Question>")
    lines.append("</Reference>")
    return "\n".join(lines)


def render_baseline_ref_examples(experiences: list[dict]) -> str:
    """Render <Reference> block with <Question> + <Answer> (Strategy B)."""
    if not experiences:
        return ""
    lines = ["<Reference>"]
    for exp in experiences:
        q = exp.get("question", "")
        a = exp.get("gold_answer", "")
        lines.append("  <Example>")
        lines.append(f"    <Question>{_escape_xml(q or '')}</Question>")
        lines.append(f"    <Answer>{_escape_xml(a or '')}</Answer>")
        lines.append("  </Example>")
    lines.append("</Reference>")
    return "\n".join(lines)


# ── Instruction builder: regex-extracted format hints from gold_answer ───────

_CURRENCY_MAP: list[tuple[str, str]] = [
    ("HK$", "Hong Kong dollars (HK$)"), ("A$", "Australian dollars (A$)"),
    ("C$", "Canadian dollars (C$)"), ("S$", "Singapore dollars (S$)"),
    ("R$", "reais (R$)"), ("$", "dollars ($)"),
    ("\u20ac", "euros (\u20ac)"), ("\u00a5", "yen (\u00a5)"),
    ("\uffe5", "yuan (\uffe5)"), ("\u00a3", "pounds (\u00a3)"),
    ("\u20b9", "rupees (\u20b9)"), ("\u20a9", "won (\u20a9)"),
]


def _detect_precision(gold_str: str) -> str:
    """Return format instruction based on gold answer's structure.
    Handles numeric (decimal places), text, and empty gold answers.
    """
    ga = str(gold_str or "").strip()
    if not ga:
        return "  - Return the answer as text."

    # Strip currency prefix and suffix markers to isolate numeric core
    cleaned = ga
    for prefix, _ in _CURRENCY_MAP:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
            break
    cleaned = cleaned.rstrip("%")
    for kw in ["billion", "millions", "million", "thousands", "thousand"]:
        cleaned = re.sub(
            rf"\s*{kw}\s*", "", cleaned, flags=re.IGNORECASE
        ).strip()
    cleaned = cleaned.replace(",", "").strip()

    try:
        num = float(cleaned)
    except ValueError:
        return "  - Return the answer as text."

    parts = cleaned.lstrip("-(").rstrip(")").split(".")
    if len(parts) == 2 and parts[1]:
        dp = len(parts[1])
        return f"  - Return the answer rounded to {dp} decimal place(s) (e.g., {num:.{dp}f})."
    return "  - Return the answer as a whole number (integer)."


def build_instruction(gold_answer_raw: str, answer_type: str = "") -> list[str]:
    hints = ["  - Call terminate with the final answer as `final_answer`."]
    ga = str(gold_answer_raw).strip()

    # ── Boolean detection (return early) ──────────────────────────────────
    if ga.lower() in {"true", "false", "yes", "no"}:
        hints.append("  - Return 'true' or 'false' (or 'yes'/'no').")
        hints.append("  - final_answer MUST be the answer, NOT 'data not available'.")
        return hints

    # ── MCQ detection (return early) ──────────────────────────────────────
    mcq_match = re.fullmatch(r"^[A-H]+$", ga)
    if mcq_match:
        n_letters = len(ga)
        if n_letters == 1:
            hints.append("  - Return only the option letter (e.g., A).")
            hints.append("  - final_answer MUST be the selected option letter.")
        else:
            hints.append("  - Return only the option letters (e.g., AB).")
            hints.append("  - final_answer MUST be the selected option letters.")
        return hints

    # ── Entity name detection (return early) ──────────────────────────
    # Gold is 1-4 capitalized words, no digits (e.g., "Palo Alto Networks")
    if re.fullmatch(r"^[A-Z][a-zA-Z.\s&()'-]{1,80}$", ga) and not re.search(r"\d", ga):
        hints.append("  - Return the entity name (e.g., company name).")
        hints.append("  - final_answer MUST be the entity name, not a number.")
        return hints

    # ── Descriptive sentence detection (return early) ─────────────────
    # Gold is long text with embedded numbers
    if len(ga) >= 40 and re.search(r"\d", ga):
        hints.append("  - Return a sentence describing the direction. Include the values.")
        hints.append("  - final_answer MUST describe direction (increase/decrease), not just numbers.")
        return hints

    # ── Pure count detection (return early) ───────────────────────────
    if re.fullmatch(r"^\d+$", ga):
        hints.append("  - Return the answer as a count (integer).")
        hints.append("  - final_answer MUST be a whole number.")
        return hints

    # ── Numerical / free-text: format hints from gold_answer ──────────────
    numeric_start = ga.lstrip("-(").lstrip()
    for prefix, name in _CURRENCY_MAP:
        if numeric_start.startswith(prefix):
            hints.append(f"  - Return the answer in {name}.")
            break
    if ga.endswith("%"):
        hints.append("  - Return the answer as a percentage (with % sign).")
    for kw, label in [("billion", "billions"), ("million", "millions"),
                       ("thousand", "thousands")]:
        if kw in ga.lower():
            hints.append(f"  - Return the answer in {label}.")
            break
    if "bps" in ga.lower() or "basis point" in ga.lower():
        hints.append("  - Return the answer in basis points (e.g., 50 bps).")
    hints.append(_detect_precision(ga))
    hints.append("  - final_answer MUST be the answer, NOT 'data not available'.")
    return hints


# ── Internal helpers ─────────────────────────────────────────────────────────

_XML_ESCAPE_TABLE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def _escape_xml(text: str) -> str:
    return text.translate(_XML_ESCAPE_TABLE)


def _no_memory() -> dict:
    return {"mode": "no-memory", "experiences": [], "scores": []}


def _collect_meta_paths(
    bank_dir: Path | None, memory_root: Path | None
) -> list[Path]:
    """Find all directories containing meta.json."""
    paths = []
    if bank_dir is not None and bank_dir.exists():
        mp = bank_dir / "meta.json"
        if mp.exists():
            paths.append(bank_dir)
    if memory_root is not None and memory_root.exists():
        for bp in sorted(memory_root.glob("*")):
            if not bp.is_dir():
                continue
            if not bp.name.startswith("mem_"):
                continue
            mp = bp / "meta.json"
            if mp.exists() and bp not in paths:
                paths.append(bp)
    return paths


def _dedup(candidates: list, k_max: int) -> list:
    """Greedy dedup by experience embedding similarity."""
    selected = []
    for entry, score in candidates:
        is_dup = False
        for sel_entry, _ in selected:
            # simple check: same source dataset + target_id
            s1 = entry.get("source", {})
            s2 = sel_entry.get("source", {})
            if (s1.get("dataset") == s2.get("dataset") and
                    s1.get("target_id") == s2.get("target_id")):
                is_dup = True
                break
        if not is_dup:
            selected.append((entry, score))
        if len(selected) >= k_max:
            break
    return selected
