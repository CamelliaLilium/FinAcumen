"""finacumen variant — full pipeline: memory retrieval + injection (Q+A+E)
+ agent loop + tools + cross-verify collection.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Optional

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.schema import ToolChoice
from finacumen.ft.tool import Terminate, ToolCollection
from finacumen.ft.tool.financial_data_lookup import FinancialDataLookup
from finacumen.ft.tool.ocr import OcrExtract, set_step_images_for_ocr
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.variant.base import DSERVariant
from finacumen.ft.variant.ft_only import FTOnlyVariant
from finacumen.fm import bank
from finacumen.fm.collect import collect as collect_experience
from finacumen.fm.relevance import AnnotationIncompleteError
from finacumen.fm.retrieve import inject_memory_into_message, retrieve
from finacumen.fm import trace_adapter

FINACUMEN_SYSTEM_PROMPT = """\
<Purpose>
You are FinAcumen, a financial reasoning agent with access to computational tools, external financial data, and lessons distilled from past similar problems. Solve each financial question step by step, using tools as needed, and call terminate with the final answer as `final_answer` when done.
</Purpose>

<Available_Tools>
- `python_execute` — Execute Python code for arithmetic, string parsing, and table joins on data you already have. Variables persist across calls within the same session.
- `financial_data_lookup` — Query an offline knowledge base of US-stock data (FinTMMBench): daily OHLCV stock prices, news articles, and quarterly financial indicators. FinancialTable values are strings: cast with int() or float() before comparison.
  TRIGGER: Whenever the question mentions a company/ticker symbol along with a date, news, sentiment, P/E, revenue, or any quarterly indicator AND the required data is NOT already provided in Context.
  RULE: ONLY call this tool when Context lacks the specific data the question needs. If Context already provides structured financial data (e.g. \"Company (TICKER) DATE: indicator=VALUE\"), use python_execute directly.
- `ocr_extract` — Re-read a chart, table, or scanned figure as clean markdown. Pass `use_context_image=true` to OCR the current image. BACKUP only — use sparingly when vision-read values are unreliable.
- `terminate` — Submit the final answer. Call with the `final_answer` parameter set to the answer value.
</Available_Tools>

<Memory_Handling>
The input may contain a `<Memory_Block>` before the `<Problem>` block. Each `<Entry>` has:
- `<Question>`: The original problem that produced this experience. Use it to judge whether your current problem is in a similar scenario (industry, data source, problem shape).
- `<Answer>`: The correct answer to that original problem. Use as a format reference — not to copy the value, but to understand what kind of output your problem expects (number, text, percentage, etc.).
- `<Experience>`: Distilled reusable rules organized as `<Findings>` and `<Cautions>`:
    - `<Findings>`: Patterns that succeeded on similar problems — follow these strategies when applicable.
    - `<Cautions>`: Guard rules from past failures — if the condition triggers, apply the rule exactly.

The block ends with an `<OptOut>` — use it only when an entry truly does not fit.

<Entity_Check>
For each Annotation, search your ENTIRE Context for the entity it references:
  1. Entity nowhere in Context → Annotation is for a different problem → IGNORE.
  2. Entity appears but the surrounding text describes a different concept
     than what the Annotation assumes → IGNORE.
  3. Entity appears with matching meaning → APPLY.
When checking, search the full Context — do not stop at the first occurrence.
A later occurrence may reveal the entity is used differently from what the
Annotation expects. When in doubt, IGNORE.
</Entity_Check>
</Memory_Handling>

<Think_Steps>
<Think>
### 1. Input Inventory (MANDATORY — do NOT skip to computation)
Output a complete inventory before any tool call. Use this format inside <Think>:

[NARRATIVE] Every standalone number with its unit and context, one per line.
  Example: "- average spread across Level 3 power delivery locations: $3.33"
[TABLES] ALL row labels with ALL column values. Do NOT summarize — list each cell.
[IMAGES] ALL fields from OCR/vision. Flag any ambiguous readings.
[TERM_MATCH] Map the question's key terms to your inventory entries.
  If a term has NO direct match, list candidate matches explicitly:
  "WARNING: 'Forward Power Basis' unmatched. Candidates: Forward powerprice ($8.86-$481),
  average spread ($3.33). Will disambiguate in Strategy."

If any section is incomplete, you have not finished reading. Go back and re-read.
DO NOT proceed to Strategy until the inventory covers ALL of Context.

### 2. Experience Applicability (if Memory_Block present)
{step2_text}

### 3. Problem Understanding
- What quantity? Unit? Precision? Check Instruction.
- Resolve TERM_MATCH warnings: which candidate fits the question's intent?
- If the question uses a financial term with inherent ambiguity (e.g., "term",
  "basis", "rate", "spread"), search for ALL sentences in Context that define or
  describe it. Do NOT stop at the first match — there may be another sentence
  further in the Context that provides a different or more complete definition.
  Compare all matches before choosing one.
- Output TYPE: Is the question asking for an ENTITY NAME (e.g., "Which company..."),
  a NUMERIC VALUE (e.g., "What was the price..."), a DESCRIPTIVE SENTENCE
  (e.g., "How did the trend evolve..."), a VERDICT (e.g., "Is X greater than Y?"),
  or NEWS SENTIMENT (e.g., "Is there any positive/negative news about...").
  Match your output to the expected type:
  * ENTITY → terminate(final_answer="Company Name") — NEVER output a number alone
  * NUMERIC → terminate(final_answer="VALUE UNIT") — always include unit
  * DESCRIPTIVE → terminate(final_answer="Direction: description with numbers")
  * VERDICT → terminate(final_answer="Yes/No" or "true/false")
  * NEWS → terminate(final_answer="Yes" or "No") after checking retrieved articles

### 4. Strategy
- Simple → tool directly. Multi-step → sub-steps with python_execute.
- Write ALL values as python variables before computing (e.g. rev_2022 = 125.4).
- Check unit conversion (millions↔billions, %↔decimal).
- BEFORE terminate: verify final_answer matches Output TYPE:
  ENTITY type → answer MUST be a company name, never a number.
  DESCRIPTIVE type → answer MUST describe direction with values.
  VERDICT type → answer MUST be Yes/No or true/false.
  NEWS type → answer MUST be Yes or No.

### 5. Next Action
- Tool call or terminate.
</Think>
After </Think>, emit the tool call. No &lt;Answer&gt; block.
</Think_Steps>

<Guiding_Checks>
Before calling terminate, verify:
- INVENTORY COMPLETENESS: did you extract numbers from narrative text AND tables?
  If any section was skipped, do NOT terminate — return to Step 1.
- Chart/figure: assign values as python variables before computing.
- Unit alignment: millions↔billions /1000; thousands↔millions /1000; ratio↔percent x100.
- Rounding: apply requested decimal places from Instruction.
- EXTERNAL-DATA CHECK: Context first; use financial_data_lookup only when Context lacks data.
- Every caution Entry: verify condition fires, apply rule.
- AMBIGUITY CHECK: If the context contains financial phrasing like "net of",
  "gross of", "including", "excluding", or "subject to", verify you understand
  the direction (addition vs subtraction vs qualification) before computing.
- PROXY CHECK: If an exact metric (e.g., EPS, P/B) cannot be computed because a
  required field is missing (e.g., net income, book value), consider using a
  closely related available metric as a reasonable proxy. State your assumption
  when you do this. Better to answer with a proxy than to give up.
- TYPE-MATCH CHECK: Before terminate, verify your final_answer matches your
  Output TYPE. ENTITY question + numeric final_answer → WRONG → go back and
  output the entity name instead. DESCRIPTIVE question + bare number → WRONG →
  add direction. VERDICT question + bare number → WRONG → output Yes/No.
</Guiding_Checks>

<Invariants>
- NEVER skip narrative paragraphs to jump to tables. Context is one document — prose AND tables must both be read.
- "data not available" is never true. If you think data is missing, re-read from the beginning — you missed a section.
- NEVER submit placeholder text as final_answer: avoid 'I will terminate', 'data not available', 'insufficient data', empty strings.
- final_answer MUST contain the actual answer value, not a meta-status.
</Invariants>"""

STEP2_TEXTS = {
    "A": (
        "Each Entry contains only a <Question> from a past similar problem.\n"
        "Use it to judge whether your current problem is in a similar scenario."
    ),
    "B": (
        "Each Entry contains <Question> and <Answer>.\n"
        "- <Question>: judge scenario similarity.\n"
        "- <Answer>: the correct answer — use as a format reference."
    ),
    "B1": (
        "Each Entry contains <Question>, <Answer>, and <Experience> (Findings + Cautions).\n"
        "- <Findings>: patterns that succeeded on similar problems — follow these strategies.\n"
        "- <Cautions>: guard rules from past failures — if the condition triggers, apply the rule."
    ),
    "B2": (
        "Each Entry contains <Question>, <Answer>, and <Annotation>.\n"
        "- <Annotation>: directive instructions (MUST/DO NOT) pre-screened for THIS problem.\n"
        "Follow the Annotation's instructions directly."
    ),
    "B3": (
        "Each Entry contains ONLY an <Annotation> — directive instructions (MUST/DO NOT).\n"
        "The Annotation is self-contained. Follow its instructions directly — it tells you exactly what to do."
    ),
    "B4": (
        "Each Entry contains <Question>, <Answer>, <Experience>, and <Annotation>.\n"
        "- <Annotation>: directive instructions (MUST/DO NOT) for THIS problem — follow these first.\n"
        "- <Experience>: Findings (successful patterns) and Cautions (guard rules) — use as supplement."
    ),
}

def build_system_prompt(config: str = "B4") -> str:
    step2 = STEP2_TEXTS.get(config, STEP2_TEXTS["B4"])
    return FINACUMEN_SYSTEM_PROMPT.format(step2_text=step2)

FIELDS_SHOWN = {
    "B1": "Question + Answer + Experience (Findings + Cautions)",
    "B2": "Question + Answer + Annotation (MUST/DO NOT directives)",
    "B3": "Annotation only (self-contained MUST/DO NOT directives)",
    "B4": "Question + Answer + Experience + Annotation",
}

FINACUMEN_NEXT_STEP_PROMPT = """\
<Next_Action_Policy>
1. Inventory all inputs (Context, Options, Question, Instruction, Memory_Block).
2. Memory_Block: check scenario similarity. Skip if question type differs.
3. Arithmetic needed → python_execute.
4. Multi-step → break into sub-steps with python_execute.
5. Company/ticker + date + indicator NOT in Context → financial_data_lookup.
6. Chart image ambiguous → ocr_extract with use_context_image=true.
7. Before terminate → verify unit alignment, convert if needed.
8. Every caution Entry → verify condition fires, apply rule.
9. When you reach a conclusion, call terminate IMMEDIATELY with the right format:
   - NUMERIC:     terminate(final_answer="VALUE UNIT")
   - ENTITY:      terminate(final_answer="Entity Name")
   - DESCRIPTIVE: terminate(final_answer="Direction: description with values")
   - VERDICT:     terminate(final_answer="Yes" or "No")
   - NEWS:        terminate(final_answer="Yes" or "No")
10. If python_execute just produced output that answers the question →
    call terminate NOW. No commentary. No repeat computation.
11. STEP LIMIT: If this is step 8 or later and you have NOT yet produced an
    answer, you MUST call terminate with your best answer NOW. Do NOT start
    a new tool call after step 8. Do NOT call financial_data_lookup after
    step 4 if it has already been called twice.
</Next_Action_Policy>"""

_THINK_RE = re.compile(r"<Think>\s*(.*?)\s*</Think>", re.DOTALL | re.IGNORECASE)
_THINK_CAP = 2500
_DEFAULT_MEMORY_DIR = Path("memory")
_DEFAULT_K_MAX = 3
_DEFAULT_COLLECT_CONCURRENCY = 1
_STRATEGY_CONFIG = {"A": "B1", "B": "B1", "C": "B1", "D": "B2", "E": "B4"}
_STRATEGY_STEP2 = {"A": "A", "B": "B", "C": "B1", "D": "B2", "E": "B4"}


async def _preprocess_ocr(image_paths: list[str]) -> str | None:
    from finacumen.fm.ocr_preprocess import preprocess_images
    return await preprocess_images(image_paths)


def _encode_image_paths(image_paths: list[str]) -> list[str]:
    """Encode up to 3 image files as base64 strings, skip missing files."""
    encoded = []
    for ip in image_paths[:3]:
        p = Path(ip)
        if p.exists():
            encoded.append(base64.b64encode(p.read_bytes()).decode("ascii"))
    return encoded


def _extract_think_block(agent) -> str:
    messages = list(getattr(getattr(agent, "memory", None), "messages", []) or [])
    for msg in reversed(messages):
        if getattr(msg, "role", "") != "assistant":
            continue
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        m = _THINK_RE.search(str(content))
        if m:
            return m.group(1).strip()[:_THINK_CAP]
        trimmed = str(content).strip()
        if trimmed:
            return trimmed[:_THINK_CAP]
    return ""


def _count_py_execute_calls(agent) -> int:
    n = 0
    for msg in getattr(getattr(agent, "memory", None), "messages", []) or []:
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                if tc.function.name == "python_execute":
                    n += 1
            except Exception:
                pass
    return n


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


def _extract_final_answer(agent: ToolCallAgent) -> str:
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
    for msg in reversed(agent.memory.messages):
        if getattr(msg, "role", "") == "assistant" and msg.content:
            m = re.search(r"\*\*Final Answer:\*\*\s*(.+?)(?:\n|$)", str(msg.content))
            if m:
                return m.group(1).strip()
    for msg in reversed(agent.memory.messages):
        if getattr(msg, "role", "") == "assistant" and msg.content:
            text = str(msg.content).strip()
            for line in reversed(text.splitlines()):
                line = line.strip()
                if not line:
                    continue
                if re.match(r'^</?[A-Za-z_|][^>]*>$', line):
                    continue
                # Prefer "final_answer: value" or "answer = value" patterns
                fm = re.match(r"(?:final_answer|voted_answer|answer)\s*[:=]\s*(.+)", line, re.IGNORECASE)
                if fm:
                    return fm.group(1).strip()
                if len(line) <= 200:
                    return line
    # Fallback: last computed value from exhausted agent
    forced = getattr(agent, "_forced_answer", None) or ""
    if forced:
        return str(forced).strip()
    return ""


class _FTOnlyWithTraceCapture(FTOnlyVariant):
    """Subclass that captures live ToolCallAgent for trace extraction."""

    name = "__ft_only_with_trace__"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._last_agent: Optional[ToolCallAgent] = None

    def _build_agent(self) -> ToolCallAgent:
        agent = super()._build_agent()
        self._last_agent = agent
        return agent


class FinAcumenVariant(DSERVariant):
    name = "finacumen"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        memory_dir = getattr(args, "memory_dir", None)
        self.bank_dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self.k_max = int(getattr(args, "memory_k_max", None) or _DEFAULT_K_MAX)
        self.use_relevance = bool(getattr(args, "use_relevance", False))
        self.relevance_filter = bool(getattr(args, "relevance_filter", False))

        raw_strategy = getattr(args, "memory_strategy", "E") or "E"
        if raw_strategy not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"--memory-strategy must be 'A'/'B'/'C'/'D'/'E', got {raw_strategy!r}")
        self.memory_strategy = raw_strategy

        self.config = getattr(args, "memory_config", None) or _STRATEGY_CONFIG.get(self.memory_strategy, "B4")
        self._override_entries: list[dict] | None = None
        concurrency = int(getattr(args, "collect_concurrency", None) or _DEFAULT_COLLECT_CONCURRENCY)
        self._collect_sem = asyncio.Semaphore(max(1, concurrency))
        self._pending: list[asyncio.Task] = []

        raw_mode = getattr(args, "memory_mode", "train") or "train"
        if raw_mode not in {"train", "test", "eval"}:
            raise ValueError(f"--memory-mode must be 'train'/'test'/'eval', got {raw_mode!r}")
        self.memory_mode = raw_mode

        memory_root = getattr(args, "memory_root", None)
        self.memory_root = Path(memory_root) if memory_root else None

        retrieval_file = getattr(args, "retrieval_file", None)
        self._retrieval_cache: dict[str, dict] = {}
        if retrieval_file:
            rp = Path(retrieval_file)
            if rp.is_file():
                for line in rp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._retrieval_cache[str(row["target_id"])] = row
            else:
                raise FileNotFoundError(f"retrieval_file not found: {retrieval_file}")

        self.base = _FTOnlyWithTraceCapture(args)
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    async def solve(self, target: dict) -> dict:
        # ── TRAIN mode: write-only, no retrieval ─────────────────────────
        if self.memory_mode == "train":
            retrieval = {"mode": "train-write-only", "experiences": [], "scores": []}
            from finacumen.fm.retrieve import render_user_message
            user_msg = render_user_message(target)
        else:
            # ── TEST mode: read-only, with retrieval ─────────────────────
            if self._override_entries is not None:
                retrieval = {"mode": "with-memory", "experiences": self._override_entries, "scores": []}
                self._override_entries = None  # consume once
            else:
                tid_cache = str(target.get("id", ""))
                if tid_cache and tid_cache in self._retrieval_cache:
                    cached = self._retrieval_cache[tid_cache]
                    retrieval = {
                        "mode": cached.get("mode", "no-memory"),
                        "experiences": cached.get("experiences", []),
                        "scores": cached.get("scores", []),
                    }
                else:
                    retrieval = retrieve(
                        target, k_max=self.k_max,
                        bank_dir=self.bank_dir,
                        memory_root=self.memory_root,
                    )

            if self.use_relevance and retrieval["mode"] == "with-memory":
                # Skip annotation if entries are already pre-annotated
                if not (retrieval["experiences"] and retrieval["experiences"][0].get("_annotation")):
                    from finacumen.fm.relevance import annotate as relevance_annotate
                    try:
                        retrieval["experiences"] = await relevance_annotate(
                            _build_problem_text(target),
                            retrieval["experiences"],
                            fields_shown=FIELDS_SHOWN.get(self.config, "Question + Answer + Experience + Annotation"),
                            filter_useful=self.relevance_filter,
                        )
                    except AnnotationIncompleteError as e:
                        logger.warning(
                            f"annotation incomplete for {target.get('id')}: {e} — skipping"
                        )
                        return {
                            "target_id": str(target.get("id", "")),
                            "correct": False,
                            "final_answer": "",
                            "variant": self.name,
                            "memory_mode": retrieval.get("mode", "no-memory"),
                            "memory_strategy": self.memory_strategy,
                            "memory_scores": retrieval.get("scores", []),
                            "memory_entry_ids": [],
                            "annotation_skipped": True,
                        }
                if not retrieval["experiences"]:
                    retrieval["mode"] = "no-memory"

            memory_image_indices: dict[int, list[int]] = {}
            mem_image_paths: list[str] = []

            if retrieval["mode"] == "with-memory":
                img_idx = 1
                for i, exp in enumerate(retrieval["experiences"]):
                    paths = exp.get("image_paths") or []
                    existing = [p for p in paths[:3] if Path(p).exists()]
                    if existing:
                        memory_image_indices[i] = list(range(img_idx, img_idx + len(existing)))
                        mem_image_paths.extend(existing)
                        img_idx += len(existing)
                user_msg = inject_memory_into_message(
                    retrieval["experiences"], target,
                    memory_image_indices=memory_image_indices if memory_image_indices else None,
                    strategy=self.memory_strategy,
                    config=self.config,
                )
            else:
                from finacumen.fm.retrieve import render_user_message
                user_msg = render_user_message(target)

        agent = self._build_agent_for_target(target)
        agent._gold_answer_hint = target.get("gold_answer", "")
        self.base._last_agent = agent

        target_image_paths = target.get("image_paths") or []
        encoded_images = _encode_image_paths(target_image_paths)
        # test mode also encodes memory image paths
        if self.memory_mode == "test" and retrieval["mode"] == "with-memory":
            encoded_images[:0] = _encode_image_paths(mem_image_paths)

        if encoded_images:
            set_step_images_for_ocr(encoded_images)

        # OCR pre-processing: extract exact values from images before agent loop
        ocr_text = await _preprocess_ocr(target.get("image_paths") or [])
        if ocr_text:
            user_msg = ocr_text + "\n\n" + user_msg

        try:
            await agent.run(user_msg, base64_images=encoded_images if encoded_images else None)
        except Exception as e:
            logger.exception(f"finacumen agent failed on {target['id']}: {e}")

        answer = _extract_final_answer(agent)

        # bump_stats: test mode only
        if self.memory_mode == "test" and retrieval["mode"] == "with-memory":
            meta_path = self.bank_dir / "meta.json"
            for entry in retrieval["experiences"]:
                src = entry.get("source", {})
                try:
                    bank.bump_stats(meta_path, dataset=src.get("dataset", ""),
                                    target_id=src.get("target_id", ""),
                                    use_delta=1, hit_delta=0)
                except Exception:
                    pass

        result = self.build_result(
            target, answer,
            extras={
                "variant": self.name,
                "python_calls": _count_py_execute_calls(agent),
                "steps": agent.current_step,
                "think_block_text": _extract_think_block(agent),
            },
            latency_sec=None,
        )

        is_correct = bool(result.get("correct"))
        if self.memory_mode == "test" and retrieval["mode"] == "with-memory" and is_correct:
            meta_path = self.bank_dir / "meta.json"
            for entry in retrieval["experiences"]:
                src = entry.get("source", {})
                try:
                    bank.bump_stats(meta_path, dataset=src.get("dataset", ""),
                                    target_id=src.get("target_id", ""),
                                    use_delta=0, hit_delta=1)
                except Exception:
                    pass

        # collect: train mode only (write-only)

        if self.memory_mode == "train":
            trace = trace_adapter.build_trace(self.base, result)
            self._pending.append(
                asyncio.create_task(self._run_collect(target, trace, result))
            )

        result["memory_mode"] = retrieval["mode"]
        result["memory_strategy"] = self.memory_strategy
        result["memory_scores"] = retrieval.get("scores", [])
        result["memory_entry_ids"] = [
            f"{e.get('source', {}).get('dataset', '')}:{e.get('source', {}).get('target_id', '')}"
            for e in retrieval.get("experiences", [])
        ]
        result["variant"] = self.name
        result["solve_steps"] = int(getattr(agent, "current_step", 0) or 0)
        result["py_execute_calls"] = _count_py_execute_calls(agent)
        result["think_block_text"] = _extract_think_block(agent)
        return result

    def _build_agent_for_target(self, target: dict) -> ToolCallAgent:
        return ToolCallAgent(
            name="finacumen-agent",
            description="FinAcumen full-pipeline agent.",
            system_prompt=build_system_prompt(_STRATEGY_STEP2.get(self.memory_strategy, "B4")),
            next_step_prompt=FINACUMEN_NEXT_STEP_PROMPT,
            llm=LLM("dser"),
            available_tools=ToolCollection(
                PythonExecute(),
                FinancialDataLookup(),
                OcrExtract(),
                Terminate(),
            ),
            tool_choices=ToolChoice.AUTO,
            max_steps=16,
            max_observe=4000,
        )

    async def _run_collect(self, target: dict, trace, result: dict) -> None:
        async with self._collect_sem:
            await collect_experience(
                target=target,
                trace=trace,
                final_answer=str(result.get("final_answer", "")),
                is_correct=bool(result.get("correct")),
                bank_dir=self.bank_dir,
            )

    async def finalize(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
            self._pending.clear()

        errors_path = self.bank_dir / "collect_errors.jsonl"
        if errors_path.exists():
            try:
                with errors_path.open("r", encoding="utf-8") as f:
                    n = sum(1 for _ in f)
                if n > 0:
                    logger.warning(f"finacumen: {n} collect failures at {errors_path}")
            except Exception:
                pass


def build_variant(args: argparse.Namespace) -> DSERVariant:
    return FinAcumenVariant(args)
