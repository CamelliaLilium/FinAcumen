"""Pruning rules — V2.
R1: Cold entries (use_count==0 after 500+ queries).
R2: Anti-give-up text (regex, write-time block).
"""
from __future__ import annotations

import re

# ── Constants ────────────────────────────────────────────────────────────────

COLD_QUERY_THRESHOLD = 500

# ── R2: Anti-give-up text detector ───────────────────────────────────────────

ANTI_GIVE_UP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdata\s+not\s+available\b",
        r"\bdata\s+not\s+provided\b",
        r"\bdata\s+is\s+missing\b",
        r"\bdata\s+unavailable\b",
        r"\binsufficient\s+data\b",
        r"\bcannot\s+determine\b",
        r"\bno\s+data\s+available\b",
        r"\bunable\s+to\s+find\b",
        r"\breturn\s+.*\b(unavailable|unknown)\b",
        r"\bconclude\s+.*\b(unavailable|unknown|missing)\b",
        r"\bstate\s+.*\b(unavailable|unknown)\b",
    ]
]

PROTECTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bretry\b.*\bbefore\b.*\bconcluding\b",
        r"\bcall\b.*\bagain\b.*\bwith\b",
        r"\badjust\b.*\b(parameter|filter|query)\b",
        r"\bwiden\b.*\b(date\s*range|window)\b",
        r"\bverify\b.*\bdata\b.*\b(before|first)\b",
        r"\bcheck\b.*\b(availability|exists)\b",
        r"\bdo\s+not\b.*\b(give\s*up|conclude|assume)\b",
        r"\bnever\b.*\b(give\s*up|conclude|assume)\b",
        r"\bensure\b.*\bdata\b.*\b(retrieved|fetched|loaded)\b",
        r"\bconfirm\b.*\bdata\b.*\b(available|present)\b",
        r"\btry\b.*\b(alternate|different)\b.*\b(symbol|ticker|name)\b",
        r"\brelaxed\b",
        r"\bproxy\b.*\bmetric\b",
        r"\bfirst\b.*\battempt\b.*\bretry\b",
        r"\bbefore\b.*\bconcluding\b.*\bretry\b",
        r"\bverify\b.*\b(tool|lookup|query)\b.*\b(called|invoked|executed)\b",
        r"\bensure\b.*\b(tool|lookup|query)\b.*\b(called|invoked|executed)\b",
        r"\bconfirm\b.*\b(tool|lookup|query)\b.*\b(called|invoked|executed)\b",
        r"\bdouble.check\b.*\b(tool|lookup|query)\b",
    ]
]


def is_anti_give_up(text: str) -> bool:
    """Check if text encodes a capitulation directive."""
    for p in PROTECTIVE_PATTERNS:
        if p.search(text):
            return False
    for p in ANTI_GIVE_UP_PATTERNS:
        if p.search(text):
            return True
    return False


# ── R1: Cold entries ─────────────────────────────────────────────────────────

def is_cold_entry(entry: dict, total_queries_seen: int) -> bool:
    """True if entry has never been used after sufficient queries."""
    st = entry.get("stats", {})
    return st.get("use_count", 0) == 0 and total_queries_seen >= COLD_QUERY_THRESHOLD
