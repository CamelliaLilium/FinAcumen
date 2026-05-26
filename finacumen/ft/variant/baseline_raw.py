"""baseline-raw variant — pure LLM single-shot, no tools, no agent loop.
Full multimodal input: question + context + images (+ optional retrieved examples).

Strategy A: Reference = similar past problem Questions only.
Strategy B: Reference = similar past problem Questions + Answers.
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.variant.base import DSERVariant
from finacumen.fm.retrieve import render_baseline_ref_questions, render_baseline_ref_examples

SYSTEM_PROMPT = """\
You will receive:
- <Reference>: Similar past problems for guidance.
- <Problem>: The current question with Context and Question.

Output ONLY: **Final Answer:** <your answer>"""

_ANSWER_RE = re.compile(r"\*\*Final Answer:\*\*\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if len(last) <= 200:
            return last
    return ""


def _encode_images(target: dict) -> list[str]:
    image_paths = target.get("image_paths") or []
    encoded = []
    for ip in image_paths[:3]:
        p = Path(ip)
        if p.exists():
            encoded.append(base64.b64encode(p.read_bytes()).decode("ascii"))
    return encoded


class BaselineRawVariant(DSERVariant):
    name = "baseline-raw"
    _system_prompt = SYSTEM_PROMPT

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        strategy = getattr(args, "memory_strategy", "A") or "A"
        self._strategy = strategy
        self._ref_experiences: list[dict] = []

        # Load retrieval cache if provided
        retrieval_file = getattr(args, "retrieval_file", None)
        if retrieval_file:
            import json
            rp = Path(retrieval_file)
            if rp.is_file():
                self._retrieval_cache = {}
                for line in rp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._retrieval_cache[str(row["target_id"])] = row

    async def solve(self, target: dict) -> dict:
        watch = self.stopwatch()
        # Load reference examples from retrieval cache
        tid = str(target.get("id", ""))
        ref_exps = []
        if hasattr(self, "_retrieval_cache") and tid in self._retrieval_cache:
            cached = self._retrieval_cache[tid]
            ref_exps = cached.get("experiences", [])
        self._ref_experiences = ref_exps

        user_text = self._build_user_message(target)
        return await self._call_llm(target, user_text, watch)

    def _build_user_message(self, target: dict) -> str:
        if self._strategy == "A" and self._ref_experiences:
            ref_block = render_baseline_ref_questions(self._ref_experiences)
            from finacumen.fm.retrieve import render_baseline_message
            prob = render_baseline_message(target, include_instruction=False)
            return ref_block + "\n" + prob if ref_block else prob
        elif self._strategy == "B" and self._ref_experiences:
            ref_block = render_baseline_ref_examples(self._ref_experiences)
            from finacumen.fm.retrieve import render_baseline_message
            prob = render_baseline_message(target, include_instruction=False)
            return ref_block + "\n" + prob if ref_block else prob
        else:
            from finacumen.fm.retrieve import render_baseline_message
            return render_baseline_message(target, include_instruction=False)

    async def _call_llm(self, target: dict, user_text: str, watch) -> dict:
        user_message = {"role": "user", "content": user_text}
        encoded = _encode_images(target)
        if encoded:
            user_message["base64_images"] = encoded

        try:
            response = await LLM("dser").ask(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    user_message,
                ],
                temperature=0.0,
            )
        except Exception as e:
            logger.exception(f"{self.name} LLM failed on {target['id']}: {e}")
            response = ""

        answer = _extract_answer(response or "")
        return self.build_result(target, answer, latency_sec=watch.elapsed())

    async def finalize(self) -> None:
        pass


def build_variant(args: argparse.Namespace) -> DSERVariant:
    return BaselineRawVariant(args)
