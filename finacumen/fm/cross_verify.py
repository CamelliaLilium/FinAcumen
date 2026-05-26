"""Cross-verify memory collection: 8×ft-only parallel agents → summary agent scoring + writing.

Replaces single-chain reflect (v1) with multi-path verification for higher-quality
experience extraction. Only triggered during finacumen train-mode collect.

Pipeline:
  sample_paths     — K=8 parallel ToolCallAgent runs (ft-only config, T=0.7)
  score_paths      — code pre-checks correctness, LLM scores only wrong paths (T=0.1)
  write_experience — single summary agent writes analysis + findings + cautions (T=0.3)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.config import config as ft_config
from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.schema import ToolChoice
from finacumen.ft.tool import Terminate, ToolCollection
from finacumen.ft.tool.financial_data_lookup import FinancialDataLookup
from finacumen.ft.tool.ocr import OcrExtract, set_step_images_for_ocr
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.variant.ft_only import FT_NEXT_STEP_PROMPT, FT_SYSTEM_PROMPT
from finacumen.fm.retrieve import render_user_message

# ── Constants ─────────────────────────────────────────────────────────────────

K_PARALLEL = 8
T_REASONING = 0.7
T_SCORING = 0.1
T_WRITING = 0.3
MAX_STEPS = 16
MAX_OBSERVE = 4000
WRITE_MAX_TOKENS = 8192

# ── LLM instances ────────────────────────────────────────────────────────────

_summary_llm: Optional[LLM] = None


def _get_summary_llm() -> LLM:
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = LLM("dser")
    return _summary_llm


def _make_agent_llm() -> LLM:
    cfg = copy.deepcopy(ft_config.llm["dser"])
    cfg.temperature = T_REASONING
    return LLM("__cv_agent__", llm_config={"default": cfg, "__cv_agent__": cfg})


async def _preprocess_cv_ocr(image_paths: list[str]) -> str | None:
    from finacumen.fm.ocr_preprocess import preprocess_images
    return await preprocess_images(image_paths)


# ── Prompts ───────────────────────────────────────────────────────────────────

SCORE_PATHS_PROMPT = """\
<Purpose>
Score each INCORRECT reasoning attempt based on how close the reasoning is to the correct approach.
Each attempt below has a wrong answer — evaluate the thinking quality (0-0.99), not the answer.
0.99 = reasoning almost correct, only a minor error. 0.0 = completely off-track.
</Purpose>

<Input>
  <Problem>{problem_text}</Problem>
  <CorrectAnswer>{gold_answer}</CorrectAnswer>
  <Attempts>
{attempts_text}
  </Attempts>
</Input>

<Scoring_Guide>
- Judge the thinking process, not the final answer (all answers here are wrong)
- Higher score = reasoning closer to the correct solution approach
- Lower score = fundamental misunderstanding or irrelevant reasoning
- Give a short reason for each score
</Scoring_Guide>

<Output>
Return ONLY:
<Answer>
[
  {{"score": 0.85, "reason": "short"}},
  {{"score": 0.30, "reason": "short"}}
]
</Answer>
</Output>"""

WRITE_EXPERIENCE_PROMPT = """\
<Purpose>
Write analysis (problem-specific derivation) then distill experience (generalized reusable rules).
CRITICAL: SUBSEQUENT AGENTS WILL ADOPT YOUR EXPERIENCE AS TRUTH. Only write what you are absolutely confident about.
</Purpose>

<Workflow>
Step A — confirm answer: The correct answer is provided in <Input><CorrectAnswer>. State it first, then derive why it is correct.
Step B — analysis: Derive the solution step-by-step from the available data. Explain: what information is available, which data fields to use, what calculations to perform, why this approach yields the CorrectAnswer.
  Purpose of analysis: to help you thoroughly understand this problem, so you can better generalize experience for similar scenarios.
Step C — findings: Generalize successful strategies from analysis and correct paths. See format below.
Step D — cautions: Generalize guard rules from analysis and incorrect paths. See format below.
Step E — verify: Re-read each rule against the traces — is it actually supported? Does it match the problem information?
</Workflow>

<analysis_guide>
IMPORTANT: The CorrectAnswer provided in <Input> is the ground truth. Your analysis MUST:
1. State the correct answer first: "The correct answer is X."
2. Derive step-by-step how to arrive at this answer from the available data.
3. If your derivation contradicts the CorrectAnswer, re-examine your assumptions — the CorrectAnswer is authoritative.
Write problem-specific derivation, do NOT generalize:
- What the available data looks like: fields in images/tables/text, numeric values present
- What the question asks: quantity, unit, precision
- Step-by-step: extract which value → apply which formula → intermediate result → final answer
- Why this approach works
This is REQUIRED. It grounds your findings/cautions in real data. Later agents use your experience — the analysis proves it is correct.
</analysis_guide>

<findings_guide>
Write generalized reusable strategies. Strip ALL specific values:
- Format: "To <goal>, use/do <method>."
- Do NOT include specific numbers, years, or entity names from this problem
- Trigger: what information format triggers this? What question type triggers this?
- Method: what to do step by step? Which tool? Which formula? What to check?
- Each finding is one sentence, self-contained and executable

Examples:
- "To compute YoY growth rate from a table of period values, use python_execute: assign each period's value to a variable, compute (new - old) / old * 100, verify unit matches question requirement."
- "To extract chart values, call ocr_extract with use_context_image=true, verify axis labels and units, assign values as python variables before computing."
</findings_guide>

<cautions_guide>
Write generalized guard rules. Strip ALL specific values:
- Format: "When <detectable condition>, <corrective action>."
- Condition must be detectable from question/context text (not hindsight)
- Action must be concrete: what to check, how to check, what to do if it fails
- Do NOT include specific numbers, years, or entity names from this problem

Examples:
- "When context provides data in millions but question requires billions, convert by dividing by 1000 before finalizing."
- "When financial_data_lookup returns empty rows, retry with relaxed date range before concluding data is unavailable."
</cautions_guide>

<Critical_All_Wrong>
If ALL attempts scored below 1.0:
- Reason: your experience will be injected into future agents as factual guidance.
- Write ONLY cautions about pitfalls you are absolutely certain of.
- Do NOT fabricate findings from incorrect reasoning.
- Prefer one high-confidence caution over three uncertain ones.
- If unsure whether a pattern is truly a pitfall, omit it.
</Critical_All_Wrong>

<Input>
  <Problem>{problem_text}</Problem>
  <CorrectAnswer>{gold_answer}</CorrectAnswer>
  <Attempts>{attempts_text}</Attempts>
  <Scores>{scores_json}</Scores>
</Input>

<Output>
Return ONLY:
<Think>
analysis: [problem-specific derivation]
→ From above, generalize findings:
→ From above, generalize cautions:
→ Verify: does each item have trace support? Is the condition detectable?
</Think>
<Answer>
{{
  "analysis": "<<REPLACE_WITH_YOUR_PROBLEM_DERIVATION_FROM_THINK_ABOVE>>",
  "findings": ["To ..., use/do ..."],
  "cautions": ["When ..., ..."]
}}
</Answer>
</Output>"""

# ── Public API ────────────────────────────────────────────────────────────────


async def collect_verified(target: dict) -> Optional[dict]:
    """Run cross-verify pipeline on a target problem.

    Returns {"experience": {"findings": [...], "cautions": [...]}, "analysis": str}
    or None if any stage fails irrecoverably.
    """
    gold = str(target.get("gold_answer", ""))

    # Step 1 — parallel reasoning (8×ft-only ToolCallAgent, T=0.7)
    paths = await sample_paths(target)
    if not paths:
        logger.warning(f"cross_verify: sample_paths returned empty for {target.get('id')}")
        return None

    n_paths = len(paths)
    logger.info(f"cross_verify: {n_paths}/{K_PARALLEL} agents completed for {target.get('id')}")

    # Step 2 — score paths (code pre-checks correct, LLM scores only wrong)
    try:
        paths = await score_paths(target, paths, gold)
    except Exception as e:
        logger.error(f"cross_verify: score_paths failed for {target.get('id')}: {e}")
        return None

    n_correct = sum(1 for p in paths if p.get("score") == 1.0)
    logger.info(f"cross_verify: {n_correct}/{n_paths} paths correct for {target.get('id')}")

    # Step 3 — write experience (single summary agent, T=0.3)
    try:
        experience = await write_experience(target, paths, gold)
    except Exception as e:
        logger.error(f"cross_verify: write_experience failed for {target.get('id')}: {e}")
        return None

    analysis = experience.get("analysis", "")

    return {
        "experience": {
            "findings": experience.get("findings", []),
            "cautions": experience.get("cautions", []),
        },
        "analysis": analysis,
    }


# ── Step 1: Parallel Reasoning ────────────────────────────────────────────────


async def sample_paths(
    target: dict,
    k: int = K_PARALLEL,
) -> list[dict]:
    """Run K independent ToolCallAgent ReAct loops (ft-only config, no memory).

    Returns list of {"trajectory": str, "answer": str}.
    """
    ocr_lock = asyncio.Lock()

    tasks = [asyncio.create_task(_run_one_agent(target, ocr_lock)) for _ in range(k)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"cross_verify: agent #{i + 1} failed: {r}")
            continue
        if r and r.get("answer"):
            valid.append(r)

    if not valid:
        logger.error("cross_verify: all agent runs failed")
        return []

    return valid


async def _run_one_agent(target: dict, ocr_lock: asyncio.Lock) -> dict:
    """Create and run one ToolCallAgent, return trajectory + answer."""
    agent = ToolCallAgent(
        name="cv-agent",
        description="Cross-verify reasoning agent (ft-only, no memory).",
        system_prompt=FT_SYSTEM_PROMPT,
        next_step_prompt=FT_NEXT_STEP_PROMPT,
        llm=_make_agent_llm(),
        available_tools=ToolCollection(
            PythonExecute(),
            FinancialDataLookup(),
            OcrExtract(),
            Terminate(),
        ),
        tool_choices=ToolChoice.AUTO,
        max_steps=MAX_STEPS,
        max_observe=MAX_OBSERVE,
    )

    user_msg = render_user_message(target)

    image_paths = target.get("image_paths") or []
    encoded_images: list[str] = []
    for ip in image_paths[:3]:
        p = Path(ip)
        if p.exists():
            encoded_images.append(base64.b64encode(p.read_bytes()).decode("ascii"))

    async with ocr_lock:
        if encoded_images:
            set_step_images_for_ocr(encoded_images)

    # OCR pre-processing
    ocr_text = await _preprocess_cv_ocr(target.get("image_paths") or [])
    if ocr_text:
        user_msg = ocr_text + "\n\n" + user_msg

    try:
        await agent.run(
            request=user_msg,
            base64_images=encoded_images if encoded_images else None,
        )
    except Exception as e:
        logger.warning(f"cross_verify: agent run failed: {e}")

    answer = _extract_agent_answer(agent)
    trajectory = _build_trajectory_text(agent)

    return {"trajectory": trajectory, "answer": answer}


def _extract_agent_answer(agent: ToolCallAgent) -> str:
    """Extract answer from agent's terminate tool call."""
    for msg in reversed(agent.memory.messages):
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if tc.function.name == "terminate":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                ans = args.get("final_answer") or args.get("status") or args.get("answer") or args.get("result")
                if ans:
                    return str(ans).strip()
    return ""


def _build_trajectory_text(agent: ToolCallAgent) -> str:
    """Build full trajectory text from agent message history. No truncation."""
    parts: list[str] = []
    for msg in agent.memory.messages:
        role = getattr(msg, "role", "")

        if role == "assistant":
            content = str(getattr(msg, "content", "") or "")
            if content:
                think_m = _THINK_RE.search(content)
                if think_m:
                    parts.append(f"[Reasoning]\n{think_m.group(1).strip()}")
                else:
                    parts.append(f"[Assistant]\n{content.strip()}")

            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                fn_name = getattr(tc.function, "name", "")
                fn_args = getattr(tc.function, "arguments", "") or "{}"
                parts.append(f"[Tool Call: {fn_name}]\n{fn_args}")

        elif role == "tool":
            content = str(getattr(msg, "content", "") or "")
            tool_name = getattr(msg, "name", "")
            if content:
                label = f"[Tool Output: {tool_name}]" if tool_name else "[Tool Output]"
                parts.append(f"{label}\n{content}")

    return "\n\n".join(parts)


# ── Step 2: Scoring ──────────────────────────────────────────────────────────


async def score_paths(
    target: dict,
    paths: list[dict],
    gold_answer: str,
) -> list[dict]:
    """Score each path: code pre-checks correctness → score=1.0.
    Only wrong paths are scored by LLM (0-0.99).
    """
    # Pre-check: code-determined correctness
    wrong_paths: list[dict] = []
    for p in paths:
        ans = p.get("answer", "")
        if _check_correct(ans, gold_answer):
            p["score"] = 1.0
            p["reason"] = "correct"
        else:
            wrong_paths.append(p)

    # If no wrong paths, all done
    if not wrong_paths:
        return paths

    # LLM scores only wrong paths
    problem_text = _build_problem_text(target)
    attempts_text = _format_attempts(wrong_paths)
    prompt = SCORE_PATHS_PROMPT.format(
        problem_text=problem_text,
        gold_answer=gold_answer,
        attempts_text=attempts_text,
    )

    llm = _get_summary_llm()
    try:
        response = await llm.ask(
            messages=[{"role": "user", "content": prompt}],
            temperature=T_SCORING,
            max_tokens=4096,
        )
    except Exception as e:
        _save_score_failure_dump(target, wrong_paths, None, repr(e))
        raise RuntimeError(f"score_paths: LLM call failed for {target.get('id')}: {e}") from e

    scores = _parse_json_array(response or "")
    if not scores or len(scores) != len(wrong_paths):
        _save_score_failure_dump(target, wrong_paths, response or "",
                                 f"JSON parse returned {len(scores) if scores else 0} scores for {len(wrong_paths)} paths")
        raise ValueError(
            f"score_paths: expected {len(wrong_paths)} scores, "
            f"got {len(scores) if scores else 0} for {target.get('id')}"
        )

    for p, s in zip(wrong_paths, scores):
        if not isinstance(s, dict):
            raise TypeError(f"score_paths: score item is not dict: {type(s).__name__} for {target.get('id')}")
        p["score"] = float(s["score"])
        p["reason"] = str(s["reason"])

    return paths


# ── Step 3: Experience Writing ────────────────────────────────────────────────


async def write_experience(
    target: dict,
    paths: list[dict],
    gold_answer: str,
) -> dict:
    """Single summary agent call: write analysis, findings, and cautions."""
    problem_text = _build_problem_text(target)
    attempts_text = _format_attempts(paths)
    scores_data = [
        {"attempt": i + 1, "score": p["score"], "reason": p["reason"]}
        for i, p in enumerate(paths)
    ]
    scores_json = json.dumps(scores_data, ensure_ascii=False)

    prompt = WRITE_EXPERIENCE_PROMPT.format(
        problem_text=problem_text,
        gold_answer=gold_answer,
        attempts_text=attempts_text,
        scores_json=scores_json,
    )

    llm = _get_summary_llm()
    try:
        response = await llm.ask(
            messages=[{"role": "user", "content": prompt}],
            temperature=T_WRITING,
            max_tokens=WRITE_MAX_TOKENS,
        )
    except Exception as e:
        raise RuntimeError(
            f"write_experience: LLM call failed for {target.get('id')}"
        ) from e

    result = _parse_json_answer(response or "")
    if result is None:
        _save_raw_write_response(target, response or "")
        raise ValueError(f"write_experience: JSON parse failed for {target.get('id')}")

    # Deduplicate while preserving order
    findings = list(dict.fromkeys(f.strip() for f in result.get("findings", []) if f.strip()))
    cautions = list(dict.fromkeys(c.strip() for c in result.get("cautions", []) if c.strip()))

    if not findings and not cautions:
        _save_raw_write_response(target, response or "")
        raise ValueError(f"write_experience: empty findings and cautions for {target.get('id')}")

    # analysis: use JSON field, fall back to <Think> block if model copied template
    analysis = result.get("analysis", "")
    _PLACEHOLDER_MARKERS = ("REPLACE_WITH", "Problem-specific derivation", "bank archive only")
    if not analysis or any(m in analysis for m in _PLACEHOLDER_MARKERS):
        think_text = _extract_think_block(response or "")
        if think_text:
            m = re.search(
                r"(?:^|\n)\s*(?:analysis|Step A)[:\-]\s*(.+?)(?:\n\s*(?:→|From|Step|findings|###))",
                think_text, re.DOTALL | re.IGNORECASE,
            )
            if m:
                analysis = m.group(1).strip()
        if not analysis:
            analysis = think_text.strip()[:800] if think_text else ""

    return {
        "analysis": analysis,
        "findings": findings,
        "cautions": cautions,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<Think>\s*(.*?)\s*</Think>", re.DOTALL | re.IGNORECASE)


def _build_problem_text(target: dict) -> str:
    parts = []
    ctx = target.get("context", "")
    opts = target.get("options", "")
    q = target.get("question", "")

    if ctx:
        parts.append(f"Context: {ctx}")
    if opts:
        parts.append(f"Options: {opts}")
    parts.append(f"Question: {q}")
    return "\n".join(parts)


def _format_attempts(paths: list[dict]) -> str:
    """Format paths into an XML-like attempts block with scores."""
    parts = []
    for i, p in enumerate(paths, 1):
        score = p.get("score")
        reason = p.get("reason", "")
        if score == 1.0:
            header = f"# --- Attempt #{i} (correct) ---"
        elif score is not None:
            header = f"# --- Attempt #{i} (score: {score}, reason: {reason}) ---"
        else:
            header = f"# --- Attempt #{i} ---"
        parts.append(f"{header}\n\n{p.get('trajectory', '')}\n")
    return "\n".join(parts)


def _extract_answer(trajectory: str) -> str:
    m = _ANSWER_RE.search(trajectory)
    if m:
        return m.group(1).strip()
    lines = trajectory.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("<") and not line.startswith("#"):
            return line
    return ""


def _extract_think_block(text: str) -> str:
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _check_correct(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False

    pred = predicted.strip().lower()
    g = gold.strip().lower()

    if pred == g:
        return True

    pred_num = _try_parse_number(pred)
    gold_num = _try_parse_number(g)
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return abs(pred_num) < 1e-9
        if abs(pred_num - gold_num) / abs(gold_num) < 0.01:
            return True
        # Percent sign mismatch: un-normalize the % side and compare raw values.
        pred_has_pct = predicted.rstrip().endswith("%")
        gold_has_pct = gold.rstrip().endswith("%")
        if pred_has_pct != gold_has_pct:
            alt_pred = pred_num * 100.0 if pred_has_pct else pred_num
            alt_gold = gold_num * 100.0 if gold_has_pct else gold_num
            if alt_gold == 0:
                if abs(alt_pred) < 1e-9:
                    return True
            elif abs(alt_pred - alt_gold) / abs(alt_gold) < 0.01:
                return True

    return False


_NUM_CLEAN_RE = re.compile(r"[,$%€¥£\s]")
_MAGNITUDE_MAP = {
    "billion": 1e9, "billions": 1e9,
    "million": 1e6, "millions": 1e6,
    "thousand": 1e3, "thousands": 1e3,
}


def _try_parse_number(text: str) -> Optional[float]:
    t = text.strip().lower()

    for sym in ["$", "€", "¥", "£", "₹", "₩", "hk$", "a$", "c$", "s$", "r$"]:
        if t.startswith(sym):
            t = t[len(sym):].strip()
            break

    is_pct = t.endswith("%")
    if is_pct:
        t = t[:-1].strip()

    magnitude = 1.0
    for word, mag in _MAGNITUDE_MAP.items():
        if t.endswith(word):
            t = t[:-len(word)].strip()
            magnitude = mag
            break

    t = _NUM_CLEAN_RE.sub("", t)
    t = t.strip(")(").strip()

    if not t:
        return None

    try:
        val = float(t)
        if is_pct:
            val = val / 100.0
        return val * magnitude
    except (ValueError, OverflowError):
        return None


def _parse_json_answer(text: str) -> Optional[dict]:
    m = re.search(r"<Answer>\s*(.*?)\s*</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    inner = m.group(1).strip()

    if inner.startswith("```"):
        inner = re.sub(r"^```\w*\s*", "", inner)
        inner = re.sub(r"\s*```$", "", inner)

    try:
        return json.loads(inner)
    except json.JSONDecodeError:
        repaired = _repair_json(inner)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.warning(f"cross_verify: failed to parse JSON from: {inner[:200]}")
            return None


def _parse_json_array(text: str) -> Optional[list]:
    """Parse a JSON array from <Answer>...</Answer> or raw text."""
    m = re.search(r"<Answer>\s*(.*?)\s*</Answer>", text, re.DOTALL | re.IGNORECASE)
    inner = (m.group(1) if m else text).strip()

    if inner.startswith("```"):
        inner = re.sub(r"^```\w*\s*", "", inner)
        inner = re.sub(r"\s*```$", "", inner)

    try:
        result = json.loads(inner)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        repaired = _repair_json(inner)
        try:
            result = json.loads(repaired)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    logger.warning(f"cross_verify: failed to parse JSON array from: {inner[:200]}")
    return None


# ── JSON Repair ──────────────────────────────────────────────────────────────

_JSON_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_json(text: str) -> str:
    """Fix common LLM JSON errors: trailing commas before } or ]."""
    return _JSON_TRAILING_COMMA_RE.sub(r"\1", text)


# ── Failure Dumps (for offline recovery) ────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]
_SCORE_FAILURES_DIR = _ROOT / "finacumen" / "memory" / "v8_unified" / "_score_failures"
_WRITE_FAILURES_DIR = _ROOT / "finacumen" / "memory" / "v8_unified" / "_write_failures"


def _save_score_failure_dump(
    target: dict, wrong_paths: list[dict],
    raw_response: Optional[str], error_msg: str,
) -> None:
    """Save agent trajectories and raw LLM response before aborting."""
    target_id = target.get("id", "unknown")
    _SCORE_FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "target_id": target_id,
        "error": error_msg,
        "wrong_path_count": len(wrong_paths),
        "wrong_paths": [
            {
                "answer": p.get("answer", ""),
                "trajectory": (p.get("trajectory", "") or "")[:3000],
            }
            for p in wrong_paths
        ],
        "raw_response": (raw_response or "")[:8000],
    }
    dump_path = _SCORE_FAILURES_DIR / f"{target_id}.json"
    dump_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"cross_verify: score failure dump saved to {dump_path}")


def _save_raw_write_response(target: dict, raw_response: str) -> None:
    """Save write_experience raw LLM response for offline JSON repair."""
    target_id = target.get("id", "unknown")
    _WRITE_FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = _WRITE_FAILURES_DIR / f"{target_id}.txt"
    dump_path.write_text(raw_response, encoding="utf-8")
    logger.info(f"cross_verify: write failure dump saved to {dump_path}")
