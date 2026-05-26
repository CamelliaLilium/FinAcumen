"""Core data types for the memory pipeline.

V2: simplified — Entry is a plain dict in bank JSON. Only Trace remains
as a dataclass for passing data between solve and collect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Trace:
    """Agent solving trace — passed from solve to collect for experience extraction."""

    final_cot: str = ""
    chain_deltas: Optional[str] = None
    final_answer: str = ""
    source_variant: str = ""
