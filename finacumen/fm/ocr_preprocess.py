"""OCR pre-processing — extract image text before the agent loop starts.

Inject OCR-extracted markdown into agent context so both vision and non-vision
models get exact numerical values from charts/tables. Does NOT depend on agent
decision-making — runs unconditionally for all targets with images.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from finacumen.ft.tool.ocr import OcrExtract

logger = logging.getLogger(__name__)

_OCR_PREFIX = (
    "[OCR Extracted Data — use these exact values for computation]\n"
    "The following was extracted via OCR from the image(s). Use ONLY these values.\n\n"
)

# Garbage patterns: DeepSeek-OCR hallucinates guardrail instructions instead of
# extracting content. Discard these outputs so they don't confuse the agent.
_GARBAGE_PATTERNS = [
    re.compile(r"^Do not round", re.IGNORECASE),
    re.compile(r"^Do not change", re.IGNORECASE),
    re.compile(r"^Do not use", re.IGNORECASE),
    re.compile(r"^Do not add", re.IGNORECASE),
    re.compile(r"^Do not delete", re.IGNORECASE),
    re.compile(r"^Round to two decimal", re.IGNORECASE),
]

_EMPTY_THRESHOLD = 8  # min characters for a meaningful OCR result


def _is_garbage(text: str) -> bool:
    """Check if OCR output is hallucinated guardrail text rather than real extraction."""
    if len(text.strip()) < _EMPTY_THRESHOLD:
        return True
    for pat in _GARBAGE_PATTERNS:
        if pat.match(text.strip()):
            return True
    return False


async def preprocess_images(image_paths: list[str]) -> str | None:
    """OCR all images and return concatenated markdown.

    Returns None if no valid images or OCR failed entirely.
    """
    if not image_paths:
        return None

    encoded: list[str] = []
    for ip in image_paths[:3]:
        p = Path(ip)
        if p.exists():
            encoded.append(base64.b64encode(p.read_bytes()).decode("ascii"))

    if not encoded:
        return None

    ocr = OcrExtract()
    try:
        result = await ocr.execute(base64_images=encoded)
    except Exception as e:
        logger.warning(f"OCR preprocessing failed: {e}")
        return None

    if not result.get("success"):
        logger.warning(f"OCR preprocessing returned failure: {result.get('error', 'unknown')}")
        return None

    ocr_results = result.get("results")
    if not ocr_results:
        md = result.get("markdown") or result.get("text", "")
        if md.strip() and not _is_garbage(md):
            return _OCR_PREFIX + md.strip()
        return None

    parts: list[str] = []
    for r in ocr_results:
        if not isinstance(r, dict):
            continue
        md = r.get("markdown") or r.get("text", "")
        if md.strip() and not _is_garbage(md):
            parts.append(md.strip())

    if not parts:
        return None

    return _OCR_PREFIX + "\n---\n".join(parts)
