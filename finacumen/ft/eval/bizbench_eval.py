"""
BizBench evaluation — SEC-NUM exact match.
All BizBench data is SEC-NUM only; no subtask routing needed.

Source: arXiv 2311.06602 (BizBench, ACL 2024).
Normalization mirrors Kensho's prompt.py (strips $, %, ,).
"""
from __future__ import annotations

import re

# Characters stripped before exact-match comparison
_STRIP_RE = re.compile(r'[$%,]')


def normalize_secnum(text: str) -> str:
    """Lowercase, strip currency/percent/comma separators, trim whitespace."""
    if not text:
        return ''
    return _STRIP_RE.sub('', str(text)).strip().lower()


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def bizbench_is_correct(target: dict, predicted: str) -> bool:
    """SEC-NUM: normalize then exact string or numeric match."""
    gold = str(target.get('gold_answer', '')).strip()
    pred = str(predicted).strip()
    if not pred:
        return False

    gn = normalize_secnum(gold)
    pn = normalize_secnum(pred)
    if gn == pn:
        return True
    gf, pf = _try_float(gn), _try_float(pn)
    if gf is not None and pf is not None:
        return gf == pf
    return False
