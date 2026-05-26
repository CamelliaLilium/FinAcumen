"""
FinMMR exact evaluation protocol — arXiv 2508.04625
Source: FinMMR benchmark evaluate/utils/evaluation_utils.py

Magnitude words (million/billion/k/etc.) are stripped WITHOUT scaling.
Assumption: pred and gold use the same scale within each item.
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Optional


def within_eps(pred: float, gt: float) -> bool:
    """0.2% relative tolerance (paper's exact definition)."""
    eps = abs(gt) * 0.002
    return (gt - eps) <= pred <= (gt + eps)


# Normalization regexes
_CURRENCY_RE = re.compile(r'[£€¥$]')
_MAGNITUDE_RE = re.compile(r'\b(?:trillion|billion|million|thousand|[mbkt])\b', re.IGNORECASE)
_UNIT_RE = re.compile(r'\b(?:usd|rmb|cny|eur|gbp)\b', re.IGNORECASE)
_PCT_DEG_RE = re.compile(r'[%°]')
_ARITH_ALLOWED = re.compile(r'^[\d\s.\+\-\*/\(\)]+$')
_ARITH_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> Optional[float]:
    """Parse and evaluate a simple arithmetic expression via AST (no exec/eval)."""
    if not _ARITH_ALLOWED.match(expr):
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            l, r = ev(node.left), ev(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, ast.Div) and r == 0:
                return None
            return _ARITH_OPS[type(node.op)](l, r)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
            o = ev(node.operand)
            return None if o is None else _ARITH_OPS[type(node.op)](o)
        return None

    return ev(tree)


_LATEX_BOXED_RE = re.compile(r'\\boxed\s*\{\s*([^{}]+?)\s*\}')
_LAST_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def normalize_numeric_answer(text: str) -> Optional[float]:
    """Strip currency/magnitude/unit/percent, parse float or arithmetic expr.

    Robust to common LLM output formats:
      - LaTeX \\boxed{X} (extracts X)
      - Prose like "The final answer is 42" (takes last number)
      - Arithmetic expressions like "1.2 * 1000"
    """
    if not text:
        return None
    raw = str(text).strip()

    # 1. Try LaTeX \boxed{...} first — extract content
    m = _LATEX_BOXED_RE.search(raw)
    if m:
        raw = m.group(1)

    s = _CURRENCY_RE.sub('', raw.lower())
    s = _MAGNITUDE_RE.sub('', s)
    s = _UNIT_RE.sub('', s)
    s = _PCT_DEG_RE.sub('', s)
    s = s.replace(',', '').strip()
    if not s:
        return None

    # 2. Direct float
    try:
        return float(s)
    except ValueError:
        pass

    # 3. Arithmetic expr (safe AST eval)
    v = _safe_eval(s)
    if v is not None:
        return v

    # 4. Fallback: last number in cleaned string (handles prose like "answer is 42")
    nums = _LAST_NUMBER_RE.findall(s)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


_BOOL_TRUE = {'yes', 'true', '1'}
_BOOL_FALSE = {'no', 'false', '0'}


def normalize_boolean_answer(text: str) -> Optional[bool]:
    """yes/true/1 → True; no/false/0 → False; else None."""
    if not text:
        return None
    s = str(text).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def finmmr_is_correct(target: dict, predicted: str) -> bool:
    """Score one FinMMR item. target needs 'answer_type' and 'gold_answer'.

    answer_type must be 'numerical' or 'boolean'. FinMMR has no MCQ items;
    if 'mcq' is encountered, boolean fallback is tried first, else ValueError.
    """
    atype = target.get('answer_type', '').strip().lower()
    gold_raw = str(target.get('gold_answer', '')).strip()
    pred_raw = str(predicted).strip()

    if atype == 'numerical':
        gv = normalize_numeric_answer(gold_raw)
        pv = normalize_numeric_answer(pred_raw)
        return False if gv is None or pv is None else within_eps(pv, gv)

    if atype == 'boolean':
        gb = normalize_boolean_answer(gold_raw)
        pb = normalize_boolean_answer(pred_raw)
        return False if gb is None or pb is None else pb == gb

    if atype == 'mcq':
        gb = normalize_boolean_answer(gold_raw)
        pb = normalize_boolean_answer(pred_raw)
        if gb is not None and pb is not None:
            return pb == gb
        raise ValueError(
            f"answer_type='mcq' invalid for FinMMR and boolean fallback failed "
            f"(gold={gold_raw!r}, pred={pred_raw!r})"
        )

    raise ValueError(f"finmmr_is_correct: unknown answer_type={atype!r}")


if __name__ == '__main__':
    # within_eps: gt=99.0, eps=0.198 → range [98.802, 99.198]
    assert not within_eps(98.7, 99.0),  "0.3% diff → outside 0.2%"
    assert within_eps(98.82, 99.0),     "0.18% diff → inside 0.2%"
    assert not within_eps(98.5, 99.0),  "0.5% diff → outside 0.2%"

    # normalize_numeric_answer: magnitude stripped without scaling
    assert normalize_numeric_answer('$ 1,234.5 million') == 1234.5
    assert normalize_numeric_answer('1.2 * 1000') == 1200.0
    assert normalize_numeric_answer('42%') == 42.0
    assert normalize_numeric_answer('€ 2.5 billion') == 2.5

    # normalize_boolean_answer
    assert normalize_boolean_answer('yes') is True
    assert normalize_boolean_answer('FALSE') is False
    assert normalize_boolean_answer('maybe') is None

    # finmmr_is_correct
    n = {'answer_type': 'numerical', 'gold_answer': '99.0'}
    assert finmmr_is_correct(n, '98.82') and not finmmr_is_correct(n, '98.5')
    b = {'answer_type': 'boolean', 'gold_answer': 'yes'}
    assert finmmr_is_correct(b, 'true') and not finmmr_is_correct(b, 'no')

    print("All smoke tests passed.")
