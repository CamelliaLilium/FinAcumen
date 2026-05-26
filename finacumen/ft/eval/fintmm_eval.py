"""
FinTMMBench evaluation: SQuAD v1.1 EM/F1 + async LLM-as-judge.

Source: arXiv 2503.05185 (FinTMMBench, ACM MM 2025)
SQuAD normalization: Rajpurkar et al. 2016 canonical evaluate-v2.0.py
LLM judge: GPT-4o-mini (per FinTMMBench paper, arXiv 2503.05185).
Resolved via the [llm.judge] config profile, falling back to env
OPENAI_API_KEY + OpenAI endpoint. NOT Qwen3-VL-8B — avoid self-evaluation
bias since the solver runs on Qwen3-VL-8B.
"""
from __future__ import annotations

import asyncio
import os
import re
import string
from collections import Counter
from typing import Any

from openai import AsyncOpenAI


# -- SQuAD v1.1 EM / F1  (Rajpurkar 2016 canonical) --

def squad_normalize(text: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    s = text.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s, flags=re.UNICODE)
    return ' '.join(s.split())


def squad_em(gold: str, pred: str) -> float:
    """SQuAD exact-match: 1.0 if normalized strings are equal, else 0.0."""
    return float(squad_normalize(gold) == squad_normalize(pred))


def squad_f1(gold: str, pred: str) -> float:
    """SQuAD token-level F1 over normalized bag-of-words."""
    gold_toks = squad_normalize(gold).split()
    pred_toks = squad_normalize(pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# -- LLM-as-judge --

_JUDGE_PROMPT = """\
[JUDGE_V3_ULTRASTRICT]
You are a strict evaluator checking for EXACT answer matches.

Question: {question}
Ground truth gold answer: {gold}
Model prediction to check: {pred}

DECISION RULES — answer NO if ANY of these apply:

A. The prediction is a NUMBER but gold is TEXT (or vice versa) → NO.
   "9.54" for "Palo Alto Networks" → NO.
   "2736.79 USD" for "Alphabet Inc. (GOOG)" → NO.

B. The prediction omits the UNIT (USD, EUR, %, million, billion) that gold includes → NO.
   "42.91" for "42.91 USD" → NO.
   "131000000" for "131,000,000 EUR" → NO.

C. The prediction's NUMERIC VALUE differs from gold by ANY amount → NO.
   "1.37" for "1.36" → NO.
   "-28125000" for "-21875000" → NO.

D. The prediction says a DIFFERENT entity name than gold → NO.
   "Fastenal" for "Ansys" → NO.
   "Qualcomm" for "Palo Alto Networks" → NO.
   
E. The prediction is tool-call text, placeholder, or non-answer → NO.
   "Call python_execute..." → NO.
   "Insufficient information" when gold has a specific answer → NO.
   "Cannot determine" → NO.

F. The prediction contradicts gold direction (increase vs decrease, yes vs no) → NO.

ONLY answer YES when:
- Numeric value, unit, entity, direction ALL match gold exactly
- OR text answer states the SAME factual claim with NO added/contradictory info
- Entity name equivalence is accepted ONLY for universally recognized abbreviations:
  "Alphabet Inc. (GOOG)" = "Alphabet" (YES)
  "Starbucks (SBUX)" = "Starbucks" (YES)

Answer with exactly ONE word: YES or NO\
"""


def _make_judge_client() -> tuple[AsyncOpenAI, str]:
    """Return (AsyncOpenAI client, model) for the FinTMMBench LLM-as-judge.

    Resolution order:
      1. ``[llm.judge]`` profile from ``config/config.toml`` — uses its
         base_url, model, and api_key. ``OPENAI_API_KEY`` env var overrides
         the config api_key when set (so secrets stay out of the repo).
      2. Fall back to ``OPENAI_API_KEY`` env + OpenAI endpoint + gpt-4o-mini.
    """
    try:
        from finacumen.ft.config import config  # type: ignore
        llm_cfg = config.llm.get("judge")
        if llm_cfg is not None:
            api_key = os.environ.get("OPENAI_API_KEY", "") or llm_cfg.api_key
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=llm_cfg.base_url,
            )
            return client, llm_cfg.model
    except Exception:
        pass

    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )
    return client, "gpt-4o-mini"


async def llm_judge(
    question: str,
    gold: str,
    pred: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o-mini",
) -> bool:
    """Call LLM judge; return True if prediction is factually correct."""
    prompt = _JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    answer = response.choices[0].message.content or ""
    return answer.strip().upper().startswith("YES")


async def llm_judge_batch(
    targets: list[dict[str, Any]],
    predictions: list[str],
    concurrency: int = 4,
    model: str = "gpt-4o-mini",
) -> list[bool]:
    """Judge all (target, prediction) pairs with bounded concurrency.

    targets: list of dicts with keys 'question', 'gold_answer'
    predictions: parallel list of model output strings
    """
    client, _default_model = _make_judge_client()
    effective_model = model  # caller controls the model override

    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(target: dict[str, Any], pred: str) -> bool:
        async with semaphore:
            return await llm_judge(
                question=str(target.get("question", "")),
                gold=str(target.get("gold_answer", "")),
                pred=pred,
                client=client,
                model=effective_model,
            )

    tasks = [_judge_one(t, p) for t, p in zip(targets, predictions)]
    return list(await asyncio.gather(*tasks))


# -- Aggregate metrics (EM + F1 only; LLM-judge is a separate async call) --

def compute_fintmm_metrics(
    targets: list[dict[str, Any]],
    predictions: list[str],
) -> dict[str, float | int]:
    """Compute SQuAD EM and F1 over all (target, prediction) pairs.

    Returns {'em': float, 'f1': float, 'n': int}.
    LLM-judge accuracy is a separate async call via llm_judge_batch.
    """
    n = len(targets)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "n": 0}

    total_em = 0.0
    total_f1 = 0.0
    for target, pred in zip(targets, predictions):
        gold = str(target.get("gold_answer", ""))
        total_em += squad_em(gold, pred)
        total_f1 += squad_f1(gold, pred)

    return {
        "em": total_em / n,
        "f1": total_f1 / n,
        "n": n,
    }


# -- Smoke tests (no API calls) --

if __name__ == "__main__":
    # canonical SQuAD v1.1: "a" is an article → removed; "The answer is A!" → "answer is"
    assert squad_normalize("The answer is A!") == "answer is", (
        f"normalize failed: {squad_normalize('The answer is A!')!r}"
    )
    assert squad_em("Constellation Energy", "constellation energy") == 1.0
    assert squad_em("Yes", "no") == 0.0
    assert squad_f1("The total is 4.7 billion", "4.7 billion") > 0.5
    assert squad_f1("abc def", "xyz") == 0.0
    print("All smoke tests passed.")
