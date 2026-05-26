"""
smart_is_correct — improved evaluation that handles common model-output format
mismatches that the strict string-equality is_correct undercounts as errors.

Error-analysis driven (Round 0/1 data):
- FinMME MCQ (multi-select): model outputs "A,B,C" or "The answer is AB"; gold is "ABC"
- FinTMM free_text: gold is "... dividend yield is 2.28%"; model outputs "2.28%"
- FinMMR numerical: model outputs close but not within strict decimal_places tolerance;
  paper likely allows some relative tolerance

This is a post-hoc evaluation fix — it does NOT change model outputs, only how
we score them for final reporting.

``native_is_correct`` routes to headline per-dataset scorers used in paper-native columns.
"""
from __future__ import annotations

import re
from typing import Optional

from finacumen.ft.eval.bizbench_eval import bizbench_is_correct
from finacumen.ft.eval.finmme_eval import finmme_item_correct
from finacumen.ft.eval.finmmr_eval import finmmr_is_correct
from finacumen.ft.eval.fintmm_eval import squad_em

# ── MCQ extraction ─────────────────────────────────────────────────────────

_LETTER_IN_TOKEN_RE = re.compile(r"\b([A-F])\b")
_PAREN_LETTER_RE = re.compile(r"\(([A-F])\)")
_ANSWER_PREFIX_RE = re.compile(
    r"(?:answer|option|choice)(?:\s*(?:is|:)?)\s*\(?([A-F])\)?",
    re.IGNORECASE,
)
_LETTER_LIST_RE = re.compile(r"(?<![A-Za-z])([A-F](?:[\s,;]+[A-F])+)(?![A-Za-z])")


def extract_mcq_letters(text: str) -> Optional[str]:
    """Return sorted uppercase letter string (e.g. 'ABC') or None if unclear."""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    stripped = re.sub(r"[\s,;]+", "", s)
    if 1 <= len(stripped) <= 6 and all(c.upper() in "ABCDEF" for c in stripped):
        return "".join(sorted(set(stripped.upper())))

    m = re.search(
        r"(?:final\s*answer|answer)[\s:*]*(?:is\s*)?([A-F](?:[\s,;]*[A-F])*)",
        s,
        re.IGNORECASE,
    )
    if m:
        letters = re.findall(r"[A-F]", m.group(1).upper())
        if letters:
            return "".join(sorted(set(letters)))

    for line in s.splitlines():
        line = line.strip().lstrip("*").strip()
        m = re.match(r"^([A-F])\s*[:\-]", line)
        if m:
            return m.group(1).upper()

    lines = [ln.strip().strip("*").strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        cleaned = re.sub(r"[\s,;]+", "", last)
        if 1 <= len(cleaned) <= 6 and all(c.upper() in "ABCDEF" for c in cleaned):
            return "".join(sorted(set(cleaned.upper())))
        m = re.match(r"^[\(]?([A-F])[\)\.\s]", last + " ")
        if m:
            return m.group(1).upper()

    letters: list[str] = []
    for m in _ANSWER_PREFIX_RE.finditer(s):
        letters.append(m.group(1).upper())
    for m in _PAREN_LETTER_RE.finditer(s):
        letters.append(m.group(1).upper())
    for line in s.splitlines():
        m = re.match(r"\s*([A-F])[.)]", line)
        if m:
            letters.append(m.group(1).upper())
            break
    if letters:
        return letters[0]

    for m in _LETTER_LIST_RE.finditer(s):
        letters.extend(re.findall(r"[A-F]", m.group(1)))
    if letters:
        return "".join(sorted(set(letters)))

    return None


def normalize_mcq_gold(gold: str) -> Optional[str]:
    if not gold:
        return None
    s = str(gold).strip()
    cleaned = re.sub(r"[\s,;]+", "", s)
    if 1 <= len(cleaned) <= 6 and all(c.upper() in "ABCDEF" for c in cleaned):
        return "".join(sorted(set(cleaned.upper())))
    return extract_mcq_letters(s)


# ── Number extraction ──────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_thousands(text: str) -> str:
    return str(text).replace(",", "")


def extract_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = _NUMBER_RE.findall(_strip_thousands(text))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def extract_all_numbers(text: str) -> list[float]:
    if not text:
        return []
    out = []
    for m in _NUMBER_RE.findall(_strip_thousands(text)):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


# ── Smart is_correct ─────────────────────────────────────────────────────────


def _numerical_match(gold_num: float, pred_num: float, decimal_places: Optional[int]) -> bool:
    abs_tol = 0.5 * 10 ** (-decimal_places) if decimal_places is not None else 1e-6
    rel_tol = max(abs(gold_num) * 0.01, abs_tol)
    return abs(gold_num - pred_num) <= rel_tol


def smart_is_correct(target: dict, predicted: str) -> bool:
    """Post-hoc evaluation with MCQ letter-set extraction + free_text numeric
    fallback + loosened numerical tolerance."""
    gold = str(target.get("gold_answer", "")).strip()
    atype = target.get("answer_type", "")
    pred = str(predicted).strip()
    if not pred:
        return False

    if atype == "mcq":
        pred_letters = extract_mcq_letters(pred)
        gold_letters = normalize_mcq_gold(gold)
        if pred_letters is None or gold_letters is None:
            return pred.strip().lower() == gold.strip().lower()
        return pred_letters == gold_letters

    if atype == "numerical":
        gold_num = extract_number(gold)
        pred_num = extract_number(pred)
        if gold_num is None or pred_num is None:
            return False
        return _numerical_match(gold_num, pred_num, target.get("decimal_places"))

    if atype == "boolean":
        gl = gold.strip().lower()
        pl = pred.strip().lower()
        gl_norm = "true" if gl in {"yes", "true", "1"} else "false" if gl in {"no", "false", "0"} else gl
        pl_norm = (
            "true"
            if pl.startswith("yes") or pl.startswith("true") or pl == "1"
            else "false"
            if pl.startswith("no") or pl.startswith("false") or pl == "0"
            else pl
        )
        return gl_norm == pl_norm

    gl = gold.strip().lower()
    pl = pred.strip().lower()
    if gl == pl:
        return True

    gold_nums = extract_all_numbers(gold)
    pred_nums = extract_all_numbers(pred)
    if gold_nums and pred_nums:
        for gn in gold_nums:
            for pn in pred_nums:
                if abs(gn) > 1e-9 and abs(pn / gn - 1) < 0.01:
                    return True
                if abs(gn - pn) < 0.01:
                    return True

    if len(gold.split()) <= 4:
        if gl in pl or pl in gl:
            return True

    g_words = set(w for w in re.findall(r"\w+", gl) if len(w) > 2)
    p_words = set(w for w in re.findall(r"\w+", pl) if len(w) > 2)
    _STOP = {
        "the",
        "and",
        "for",
        "with",
        "are",
        "was",
        "were",
        "from",
        "has",
        "have",
        "this",
        "that",
        "will",
        "would",
        "should",
        "could",
    }
    g_words -= _STOP
    p_words -= _STOP
    if g_words and p_words:
        jac = len(g_words & p_words) / len(g_words | p_words)
        if jac >= 0.6:
            return True

    return False


def native_is_correct(ds: str, target: dict, pred: str) -> bool:
    """Headline protocol per dataset (matches paper-native column)."""
    if ds.startswith("finmmr"):
        return finmmr_is_correct(target, pred)
    if ds == "bizbench":
        tgt = dict(target)
        tgt.setdefault("task", "SEC-NUM")
        return bizbench_is_correct(tgt, pred)
    if ds == "finmme":
        return finmme_item_correct(target, pred)
    if ds == "fintmm":
        return squad_em(str(target.get("gold_answer", "")), pred) > 0.5
    return False
