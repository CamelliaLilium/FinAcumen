"""
FinMME evaluation — FinScore formula.
Source: arXiv 2505.24714 + github.com/luo-junyu/FinMME eval.py

FinScore = domain_normed_accuracy * (1 - hallucination_rate)
  domain_normed_accuracy = macro-avg of per-domain accuracy
  hallucination_rate     = error rate on multiple_choice items only

NOTE: Our raw data LACKS knowledge_domain tags. When all targets miss this
field, domain_normed_accuracy falls back to micro-accuracy and FinScore is
NOT comparable to paper baselines.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

# ── MCQ extraction ────────────────────────────────────────────────────────────

_ANSWER_COLON_RE = re.compile(r'(?i)Answer\s*:\s*([^\s\n]+)')
_ANSWER_IS_RE = re.compile(
    r'(?i)(?:the\s+)?answer(?:\s+is)?\s*[:\s]\s*([A-N][,\s]*(?:[A-N][,\s]*)*)'
)
_VALID = set('ABCDEFGHIJKLMN')


def extract_mcq_answer(text: str) -> str:
    """Apply FinMME regex (Answer: <token>); try natural 'answer is X' form; fall back to full text."""
    s = str(text)
    m = _ANSWER_COLON_RE.search(s) or _ANSWER_IS_RE.search(s)
    return m.group(1) if m else text


def _lset(s: str) -> set:
    return {c for c in s.upper() if c in _VALID}


# ── Numerical parsing ─────────────────────────────────────────────────────────

_STRIP_RE = re.compile(r'[,\$£€¥%\s]')
_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')


def _parse_float(text: str) -> Optional[float]:
    """Strip commas/currency/% then parse float; fall back to first numeric token."""
    cleaned = _STRIP_RE.sub('', str(text).strip())
    try:
        return float(cleaned)
    except ValueError:
        m = _NUM_RE.search(cleaned)
        return float(m.group()) if m else None


# ── Per-item scoring ──────────────────────────────────────────────────────────

def finmme_item_correct(target: dict, predicted: str) -> bool:
    """Per-item correctness.

    Source-of-truth fields:
      - target['question_type'] ∈ {'single_choice','multiple_choice','numerical'}
        (FinMME paper's nomenclature, present in raw data)
      - target['answer_type'] ∈ {'mcq','numerical', ...}
        (our adapter's normalized form, present whenever question_type is missing)

    Bug fix 2026-04-30: when question_type is None (our adapter strips it
    on load) we used to fall through to a strict full-string lower-case
    comparison, so 'Answer: A' was judged != 'A' even though regex extraction
    would have recovered 'A'. Now we route via answer_type as a second key.
    """
    qtype = target.get('question_type') or ''
    atype = (target.get('answer_type') or '').lower()
    gold = str(target.get('gold_answer', '')).strip()
    pred = str(predicted).strip()

    is_mcq = (qtype in ('single_choice', 'multiple_choice')) or (atype == 'mcq')
    is_numeric = (qtype == 'numerical') or (atype in ('numerical', 'number'))

    if is_mcq:
        return _lset(extract_mcq_answer(pred)) == _lset(gold)

    if is_numeric:
        tol = float(target.get('tolerance') or 0.0)
        gv, pv = _parse_float(gold), _parse_float(pred)
        if gv is None or pv is None:
            return False
        return abs(pv - gv) <= tol + 1e-9  # 1e-9 guards float representation drift

    return pred.lower() == gold.lower()


# ── FinScore aggregation ──────────────────────────────────────────────────────

def compute_finscore(results: list[dict], targets_by_id: dict) -> dict:
    """Compute FinScore.

    Args:
        results: list of {'target_id': str, 'correct': bool}
        targets_by_id: target_id -> target dict (with 'question_type',
            optionally 'knowledge_domain')

    Returns:
        {'finscore', 'domain_normed_accuracy', 'hallucination_rate',
         'per_domain_accuracy', 'n', 'n_multi', 'n_domains', 'warning'}
    """
    n = len(results)
    _empty = dict(finscore=0.0, domain_normed_accuracy=0.0, hallucination_rate=0.0,
                  per_domain_accuracy={}, n=0, n_multi=0, n_domains=0, warning=None)
    if n == 0:
        return _empty

    rows = []
    miss = 0
    for r in results:
        t = targets_by_id.get(r['target_id'], {})
        d = t.get('knowledge_domain')
        if d is None:
            miss += 1
        rows.append({'correct': bool(r['correct']),
                     'qtype': t.get('question_type', ''),
                     'domain': d})

    warning: Optional[str] = None
    all_missing = miss == n

    if all_missing:
        warning = ('Missing knowledge_domain field; falling back to micro-accuracy. '
                   'FinScore is NOT comparable to paper baselines without domain tags.')
        for row in rows:
            row['domain'] = '__micro__'
    elif miss > 0:
        for row in rows:
            if row['domain'] is None:
                row['domain'] = 'unknown'

    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[row['domain']].append(row['correct'])

    per_domain = {d: sum(v) / len(v) for d, v in buckets.items()}

    if all_missing:
        dna = sum(r['correct'] for r in rows) / n
        per_domain_out: dict = {}
    else:
        dna = sum(per_domain.values()) / len(per_domain)
        per_domain_out = per_domain

    multi = [r for r in rows if r['qtype'] == 'multiple_choice']
    n_multi = len(multi)
    hr = sum(1 for r in multi if not r['correct']) / n_multi if n_multi else 0.0

    return {
        'finscore': dna * (1.0 - hr),
        'domain_normed_accuracy': dna,
        'hallucination_rate': hr,
        'per_domain_accuracy': per_domain_out,
        'n': n,
        'n_multi': n_multi,
        'n_domains': len(buckets),
        'warning': warning,
    }


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    assert extract_mcq_answer('Let me think... Answer: A,B') == 'A,B'
    assert finmme_item_correct({'question_type': 'single_choice', 'gold_answer': 'A'},
                                'The answer is A.')
    assert not finmme_item_correct({'question_type': 'multiple_choice', 'gold_answer': 'AB'},
                                    'AC')
    assert finmme_item_correct({'question_type': 'numerical', 'gold_answer': '1.23',
                                 'tolerance': 0.01}, '1.22')
    assert not finmme_item_correct({'question_type': 'numerical', 'gold_answer': '1.23'},
                                    '1.24')
    out = compute_finscore([{'target_id': 'x', 'correct': True}],
                           {'x': {'knowledge_domain': 'foo', 'question_type': 'numerical'}})
    assert 'finscore' in out and out['finscore'] == 1.0
    print('All smoke tests passed.')
    print(f'Sample output: {out}')
