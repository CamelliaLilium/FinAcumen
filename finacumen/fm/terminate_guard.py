"""TerminateGuard — validates terminate final_answer against expected format.

WARN mode: logs warnings for placeholder values and format mismatches,
but does not block termination (agent loop exits normally).
"""
from __future__ import annotations

import re
from typing import Optional

from finacumen.ft.logger import logger

_PLACEHOLDER_PATTERNS: list[str] = [
    r"^data not available$",
    r"^data not provided$",
    r"^insufficient data$",
    r"^cannot determine$",
    r"^I will terminate$",
    r"^terminate$",
    r"^unknown$",
    r"^n/a$",
    r"^none$",
]

_CURRENCY_SYMBOLS: list[str] = [
    "HK$", "A$", "C$", "S$", "R$",
    "$", "\u20ac", "\u00a5", "\uffe5", "\u00a3",
    "\u20b9", "\u20a9",
]


class TerminateGuard:
    """Validates terminate final_answer values (WARN mode only)."""

    @classmethod
    def validate(
        cls,
        final_answer: Optional[str],
        gold_answer: Optional[str] = None,
    ) -> bool:
        """Check final_answer quality. Returns True (always allows terminate in WARN mode)."""
        if not final_answer:
            return True

        s = str(final_answer).strip()
        if not s:
            return True

        lower = s.lower()

        for pattern in _PLACEHOLDER_PATTERNS:
            if re.fullmatch(pattern, lower):
                logger.warning(
                    f"TerminateGuard: final_answer='{s}' is a placeholder value"
                )
                return True

        if gold_answer:
            cls._validate_format(s, str(gold_answer).strip())

        return True

    @classmethod
    def _validate_format(cls, status: str, ga: str) -> None:
        lower_ga = ga.lower()
        lower_s = status.lower()

        if lower_ga in {"true", "false", "yes", "no"}:
            if lower_s not in {"true", "false", "yes", "no"}:
                logger.warning(
                    f"TerminateGuard: expected boolean, got '{status}'"
                )
            return

        if re.fullmatch(r"^[A-H]+$", ga):
            if not re.fullmatch(r"^[A-H]+$", status):
                logger.warning(
                    f"TerminateGuard: expected MCQ letter(s), got '{status}'"
                )
            return

        numeric_prefix = ga.lstrip("-(").lstrip()
        for sym in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True):
            if numeric_prefix.startswith(sym):
                if not status.strip().lstrip("-(").lstrip().startswith(sym):
                    logger.warning(
                        f"TerminateGuard: expected currency prefix '{sym}', got '{status}'"
                    )
                break

        if ga.endswith("%") and not status.endswith("%"):
            logger.warning(
                f"TerminateGuard: expected percentage (with %), got '{status}'"
            )

        cleaned_ga = re.sub(r"^[^\d]*", "", ga)
        cleaned_ga = re.sub(r"[^\d.]", "", cleaned_ga)
        dot_ga = re.search(r"\.(\d+)$", cleaned_ga)
        if dot_ga:
            expected = len(dot_ga.group(1))
            cleaned_s = re.sub(r"^[^\d]*", "", status)
            cleaned_s = re.sub(r"[^\d.]", "", cleaned_s)
            dot_s = re.search(r"\.(\d+)$", cleaned_s)
            if dot_s:
                actual = len(dot_s.group(1))
                if actual < expected:
                    logger.warning(
                        f"TerminateGuard: expected {expected} decimal places, got {actual}"
                    )
