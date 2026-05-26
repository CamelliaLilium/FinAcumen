"""
Answer extraction helpers — MCQ letter extraction and number extraction.

These are pure string-processing utilities used by evaluation and tool code.
Not an eval protocol; they extract structured answers from free-form model output.
"""
from __future__ import annotations

import re
from typing import Optional


# ── MCQ extraction ─────────────────────────────────────────────────────────

_LETTER_IN_TOKEN_RE = re.compile(r'\b([A-F])\b')
_PAREN_LETTER_RE = re.compile(r'\(([A-F])\)')
_ANSWER_PREFIX_RE = re.compile(
    r'(?:answer|option|choice)(?:\s*(?:is|:)?)\s*\(?([A-F])\)?',
    re.IGNORECASE,
)
_LETTER_LIST_RE = re.compile(r'(?<![A-Za-z])([A-F](?:[\s,;]+[A-F])+)(?![A-Za-z])')


def extract_mcq_letters(text: str) -> Optional[str]:
    """Return sorted uppercase letter string (e.g. 'ABC') or None if unclear.

    Strategy (in order, first yield wins for common prefixes):
      1. Pure letter+separator string ("ABC", "A B C", "A,B,C")
      2. 'Final Answer:' / 'Answer:' / 'Option' prefix followed by letter(s)
      3. Letter-colon form like "B: 12.5%" — take the letter (option label)
      4. Last non-empty line is a single letter or short letter list
      5. 'The answer is (C)' / parenthesized letters / 'C.' / 'C)' patterns
      6. Letter lists "A, B, D" anywhere in text
    """
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    # 1. Direct: pure-letter-and-separators string, e.g. "ABC", "A B C", "A,B,C"
    stripped = re.sub(r'[\s,;]+', '', s)
    if 1 <= len(stripped) <= 6 and all(c.upper() in 'ABCDEF' for c in stripped):
        return ''.join(sorted(set(stripped.upper())))

    # 2. **Final Answer:** / Answer: / Final answer = prefix — capture trailing letters
    m = re.search(
        r'(?:final\s*answer|answer)[\s:*]*(?:is\s*)?([A-F](?:[\s,;]*[A-F])*)',
        s, re.IGNORECASE,
    )
    if m:
        letters = re.findall(r'[A-F]', m.group(1).upper())
        if letters:
            return ''.join(sorted(set(letters)))

    # 3. Letter-colon form: "B: 12.5%" means option B. Find at start of line.
    for line in s.splitlines():
        line = line.strip().lstrip('*').strip()
        m = re.match(r'^([A-F])\s*[:\-]', line)
        if m:
            # Accept only if there's just one such option label (avoid full option list)
            return m.group(1).upper()

    # 4. Last non-empty line: single letter or short letter list
    lines = [ln.strip().strip('*').strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        cleaned = re.sub(r'[\s,;]+', '', last)
        if 1 <= len(cleaned) <= 6 and all(c.upper() in 'ABCDEF' for c in cleaned):
            return ''.join(sorted(set(cleaned.upper())))
        # Single letter on last line possibly with punctuation
        m = re.match(r'^[\(]?([A-F])[\)\.\s]', last + ' ')
        if m:
            return m.group(1).upper()

    # 5. Natural-language patterns
    letters: list[str] = []
    for m in _ANSWER_PREFIX_RE.finditer(s):
        letters.append(m.group(1).upper())
    for m in _PAREN_LETTER_RE.finditer(s):
        letters.append(m.group(1).upper())
    # "C." / "C)" at line starts (ignore option lists by taking only first)
    for line in s.splitlines():
        m = re.match(r'\s*([A-F])[.)]', line)
        if m:
            letters.append(m.group(1).upper())
            break
    if letters:
        # For prefix patterns, take the FIRST hit only — it's the answer
        return letters[0]

    # 6. Letter lists last-ditch
    for m in _LETTER_LIST_RE.finditer(s):
        letters.extend(re.findall(r'[A-F]', m.group(1)))
    if letters:
        return ''.join(sorted(set(letters)))

    return None


# ── Number extraction ──────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _strip_thousands(text: str) -> str:
    """Drop US-style thousand separator commas before regex extraction.

    Without this, "600,000" tokenizes as ['600', '000'] and extract_number
    returns 0 instead of 600000 — a v0 audit bug that silently failed eval
    on bizbench items where gold was formatted with thousand separators.
    """
    return str(text).replace(',', '')


def extract_number(text: str) -> Optional[float]:
    """Return the LAST numeric token in text, or None."""
    if not text:
        return None
    m = _NUMBER_RE.findall(_strip_thousands(text))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None
