"""ft-only variant — ToolCallAgent loop with PythonExecute + FinancialDataLookup
+ OcrExtract + Terminate. No memory, no DSER. Baseline for tool capability.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.schema import ToolChoice
from finacumen.ft.tool import Terminate, ToolCollection
from finacumen.ft.tool.financial_data_lookup import FinancialDataLookup
from finacumen.ft.tool.ocr import OcrExtract, set_step_images_for_ocr
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.variant.base import DSERVariant
from finacumen.fm.retrieve import render_user_message

FT_SYSTEM_PROMPT = """\
<Purpose>
You are a financial reasoning assistant. Solve a financial question step by step, using python_execute for computations, and call terminate with the final answer as `final_answer` once done.
</Purpose>

<Available_Tools>
- `python_execute` — Run Python code for computation, string parsing, table joins. Variables persist across calls.
- `financial_data_lookup` — Query offline US-stock data: daily OHLCV prices, news articles, quarterly financial indicators. Params: source (stock_price|news|financial_table), symbol, date, indicator. FinancialTable values are strings: cast with int()/float() before comparison. ONLY call when Context lacks the required data. If Context already provides structured financial data (e.g. \"Company (TICKER) DATE: indicator=VALUE\"), use python_execute directly.
- `ocr_extract` — OCR chart/table → clean markdown text via DeepSeek-OCR. Pass `use_context_image=true` to OCR the current image.
- `terminate` — Submit final answer as `final_answer`.
</Available_Tools>

<Think_Steps>
<Think>
### 1. Input Inventory
- Read all available inputs: Context, Options, Question.
- Read the Instruction block for format requirements (decimal places, units, percentage, etc.).

### 2. Problem Understanding
- What quantity? What unit? What precision? Check Instruction for formatting.
- If the question uses a financial term with inherent ambiguity (e.g., "term",
  "basis", "rate", "spread"), search for ALL sentences in Context that define or
  describe it. Do NOT stop at the first match — there may be another sentence
  further in the Context that provides a different or more complete definition.
  Compare all matches before choosing one.

### 3. Tool Decision
Decide which tool:
- python_execute: if arithmetic, string parsing, or table computation is needed.
- financial_data_lookup: if the question references a company/ticker + date/news/sentiment/quarterly metric AND the answer is NOT already in Context.
- ocr_extract: if chart/image values are ambiguous or contradict across attempts.

### 4. Compute
- If reading values from Context/table/figure, write them out as python variables (e.g. rev_2022 = 125.4) before computing.
- Detect whether unit conversion is needed (millions↔billions, percent↔decimal, etc.).

### 5. Next Action
- Tool call, or terminate with the final answer.
</Think>
After </Think>, emit the corresponding tool call. No &lt;Answer&gt; block — the answer lives in terminate's `final_answer` argument.
</Think_Steps>

<Guiding_Checks>
Before calling terminate, verify:
- Chart/figure: first read numbers from the image and assign them as python variables before computing.
- Unit alignment: millions->billions /1000; thousands->millions /1000; ratio->percent x100. Convert in python_execute if source and target units differ.
- Rounding: apply any requested decimal places from Instruction.
- EXTERNAL-DATA CHECK (CRITICAL): first check if Context already provides the required data (e.g. \"Company (TICKER) DATE: indicator=VALUE\"). Only call financial_data_lookup if Context lacks the specific data.
</Guiding_Checks>

<Invariants>
- NEVER submit placeholder text as the final_answer: avoid 'I will terminate', 'data not available', 'insufficient data', empty strings.
- final_answer MUST contain the actual answer value, not a meta-status.
</Invariants>"""

FT_NEXT_STEP_PROMPT = """\
<Next_Action_Policy>
1. Inventory: what inputs are available? (Context, Options, Question, Instruction for format requirements).
2. If arithmetic computation is needed -> call `python_execute`.
3. If the question references a company/ticker symbol on a specific date, news/sentiment/events for a company over a period, or a quarterly indicator (revenue/margin/P/E/etc.) and the answer is NOT already in Context -> MUST call `financial_data_lookup` before claiming data is unavailable.
4. If chart image values are ambiguous or contradict across attempts -> call `ocr_extract` with `use_context_image=true`.
5. Before terminating -> verify unit alignment per Instruction; convert in python_execute if source and target units differ.
6. Call `terminate` with the final answer as `final_answer`.
    - final_answer MUST be the determined answer.
    - NEVER use 'I will terminate', 'data not available', or 'insufficient data' as final_answer.
</Next_Action_Policy>"""


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
                fm = re.match(r"(?:final_answer|voted_answer|answer)\s*[:=]\s*(.+)", line, re.IGNORECASE)
                if fm:
                    return fm.group(1).strip()
                if len(line) <= 200:
                    return line
    forced = getattr(agent, "_forced_answer", None) or ""
    if forced:
        return str(forced).strip()
    return ""


class FTOnlyVariant(DSERVariant):
    name = "ft-only"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def _build_agent(self) -> ToolCallAgent:
        return ToolCallAgent(
            name="ft-only-finance",
            description="Finance agent with tools (no memory, no DSER).",
            system_prompt=FT_SYSTEM_PROMPT,
            next_step_prompt=FT_NEXT_STEP_PROMPT,
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

    async def solve(self, target: dict) -> dict:
        watch = self.stopwatch()
        agent = self._build_agent()
        agent._gold_answer_hint = target.get("gold_answer", "")
        user_msg = render_user_message(target)

        image_paths = target.get("image_paths") or []
        encoded_images = []
        if image_paths:
            for ip in image_paths[:3]:
                p = Path(ip)
                if p.exists():
                    encoded_images.append(base64.b64encode(p.read_bytes()).decode("ascii"))
        if encoded_images:
            set_step_images_for_ocr(encoded_images)

        # OCR pre-processing
        from finacumen.fm.ocr_preprocess import preprocess_images as _preprocess_ocr
        ocr_text = await _preprocess_ocr(target.get("image_paths") or [])
        if ocr_text:
            user_msg = ocr_text + "\n\n" + user_msg

        try:
            await agent.run(user_msg, base64_images=encoded_images if encoded_images else None)
        except Exception as e:
            logger.exception(f"ft-only failed on {target['id']}: {e}")

        answer = _extract_final_answer(agent)
        py_calls = sum(
            1 for msg in agent.memory.messages
            for tc in (getattr(msg, "tool_calls", None) or [])
            if tc.function.name == "python_execute"
        )
        return self.build_result(
            target, answer,
            extras={
                "variant": self.name,
                "python_calls": py_calls,
                "steps": agent.current_step,
            },
            latency_sec=watch.elapsed(),
        )


def build_variant(args: argparse.Namespace) -> DSERVariant:
    return FTOnlyVariant(args)
