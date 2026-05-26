"""Ensure ``finacumen`` is importable when tests run without ``pip install -e``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FINACUMEN_SRC = _REPO / "finacumen"
_s = str(_FINACUMEN_SRC)
if _s not in sys.path:
    sys.path.insert(0, _s)
