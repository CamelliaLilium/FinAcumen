#!/usr/bin/env python3
"""
utils.py — Shared DSER evaluation helpers (FinAcumen `finacumen.ft.dser_compat`).

Consolidates duplicated helpers from run_dser.py, run_sc_qwen.py,
run_self_consistency.py, run_qwen_ab_test.py, prepare_ab_test_data.py,
and step4_build_index.py into one canonical implementation.

Sections:
  1. I/O Helpers
  2. Answer Parsing & Scoring
  3. Image Encoding
  4. Prompt Constants
  5. Retry Helper
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math  # noqa: F401 — available for callers
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. I/O Helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """Read and return the parsed contents of a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    """Read a newline-delimited JSON file and return a list of parsed objects."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Serialise *obj* to *path* as pretty-printed JSON (UTF-8, no ASCII escaping)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    """Write *rows* to a newline-delimited JSON file.

    Args:
        path: Destination file path.
        rows: Sequence of JSON-serialisable dicts.
        mode: File open mode — ``"w"`` (overwrite) or ``"a"`` (append).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path, key: str = "id") -> set[str]:
    """Return the set of IDs already written to *path* (a JSONL file).

    Silently returns an empty set when *path* does not exist.

    Args:
        path: Path to an existing JSONL results file.
        key:  Field name that stores the ID within each record.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        val = row.get(key)
        if val is not None:
            done.add(str(val))
    return done


# ---------------------------------------------------------------------------
# 2. Answer Parsing & Scoring
# ---------------------------------------------------------------------------


def parse_float(value: str | int | float | None) -> float | None:
    """Convert *value* to float, stripping common currency/percent symbols.

    Handles:
    - Native ``int`` / ``float`` pass-through.
    - Currency symbols: ``$``, ``%``, ``USD``, ``TWD`` (upper and lower case).
    - Comma thousand-separators.
    - Scientific notation (``1.5e3``).

    Returns ``None`` when no number can be extracted.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("USD", "")
        .replace("TWD", "")
        .replace("usd", "")
        .replace("twd", "")
    )
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_final_answer(text: str) -> str:
    """Extract the model's final answer from *text*.

    Checks in order:
    1. ``**Final Answer:**`` — bold markdown form.
    2. ``Final Answer:`` — plain-text form (no bold markers).
    3. ``\\boxed{...}`` — LaTeX boxed notation.
    4. Fallback: the last non-empty line of *text*.

    Trailing periods are stripped from all variants.
    """
    # 1. Bold markdown form
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].rstrip(".")
    # 2. Plain-text form
    m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].rstrip(".")
    # 3. LaTeX boxed
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip().rstrip(".")
    # 4. Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_mcq(value: str) -> str:
    """Normalise an MCQ answer to a sorted string of unique uppercase letters.

    Example: ``"B, A, B"`` → ``"AB"``
    """
    letters = sorted(set(re.findall(r"[A-Za-z]", value.upper())))
    return "".join(letters)


def normalize_boolean(value: str) -> str:
    """Normalise a boolean answer to ``"yes"`` or ``"no"``.

    Recognises: ``yes``, ``true``, ``1`` → ``"yes"``; everything else → ``"no"``.
    """
    v = (value or "").strip().lower()
    if v in {"yes", "true", "1"}:
        return "yes"
    return "no"


def _resolve_numerical_tolerance(target: dict, default_tol: float = 1e-6) -> float:
    """Determine numerical comparison tolerance from target metadata.

    Priority order:
    1. ``target["metadata"]["tolerance"]`` (explicit per-case override)
    2. ``target["decimal_places"]`` → ``0.5 * 10**(-dp)``
       (e.g. dp=2 → tol=0.005, dp=0 → tol=0.5)
    3. *default_tol* fallback (``1e-6``)
    """
    # Check explicit tolerance in metadata
    meta = target.get("metadata")
    if isinstance(meta, dict):
        explicit = meta.get("tolerance")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass

    # Derive from decimal_places (the correct approach for financial data)
    dp = target.get("decimal_places")
    if dp is not None:
        try:
            return 0.5 * 10 ** (-int(dp))
        except (TypeError, ValueError):
            pass

    return default_tol


def is_correct(
    target: dict,
    predicted: str,
    default_tol: float = 1e-6,
) -> bool:
    """Check whether *predicted* matches the gold answer stored in *target*.

    Type-specific comparison rules:

    - ``numerical``: numeric equality within a tolerance derived from
      ``decimal_places`` (e.g. dp=2 → tol=0.005).  Falls back to explicit
      ``metadata.tolerance``, then *default_tol*.
    - ``mcq``: sorted-letter comparison via :func:`normalize_mcq`.
    - ``boolean``: case-insensitive yes/no via :func:`normalize_boolean`.
    - ``free_text``: case-insensitive stripped string comparison.
    """
    gold = str(target.get("gold_answer", ""))
    atype = target.get("answer_type")

    if atype == "numerical":
        tol = _resolve_numerical_tolerance(target, default_tol)
        gold_num = parse_float(gold)
        pred_num = parse_float(predicted)
        return (
            gold_num is not None
            and pred_num is not None
            and abs(gold_num - pred_num) <= tol
        )

    if atype == "mcq":
        return normalize_mcq(gold) == normalize_mcq(predicted)

    if atype == "boolean":
        return normalize_boolean(gold) == normalize_boolean(predicted)

    # free_text (and any unrecognised type)
    return gold.strip().lower() == predicted.strip().lower()


def majority_vote(answers: list[str]) -> str:
    """Return the most common answer after filtering out ERROR entries.

    ``ERROR``-prefixed answers are always excluded before voting.  If all
    answers are errors (or the list is empty) the first element is returned as
    a last resort.
    """
    valid = [a for a in answers if a and not a.startswith("ERROR")]
    if not valid:
        return answers[0] if answers else ""
    return Counter(valid).most_common(1)[0][0]


def majority_vote_numerical(
    answers: list[str],
    decimal_places: int | None = None,
) -> str:
    """Tolerance-aware majority vote for numerical answers.

    Clusters answers within ``0.5 * 10**(-dp)`` tolerance, then picks the
    most popular string from the largest cluster.  Falls back to standard
    string-based :func:`majority_vote` when *decimal_places* is ``None``
    or answers are non-numeric.
    """
    valid = [a for a in answers if a and not a.startswith("ERROR")]
    if not valid:
        return answers[0] if answers else ""

    if decimal_places is None:
        return Counter(valid).most_common(1)[0][0]

    tol = 0.5 * 10 ** (-decimal_places)

    # Parse to floats for clustering
    parsed: list[tuple[str, float]] = []
    unparsed: list[str] = []
    for a in valid:
        f = parse_float(a)
        if f is not None:
            parsed.append((a, f))
        else:
            unparsed.append(a)

    if not parsed:
        return Counter(valid).most_common(1)[0][0]

    # Greedy clustering by tolerance on sorted values
    parsed.sort(key=lambda x: x[1])
    clusters: list[list[str]] = [[parsed[0][0]]]
    centroids: list[float] = [parsed[0][1]]
    for s, f in parsed[1:]:
        if abs(f - centroids[-1]) <= tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])
            centroids.append(f)

    # Pick largest cluster; include unparsed in tie-breaking
    largest = max(clusters, key=len)
    return Counter(largest).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 3. Image Encoding
# ---------------------------------------------------------------------------


def encode_image_data_url(
    path: Path,
    max_side: int = 0,
    quality: int = 85,
) -> str:
    """Encode an image file as a ``data:`` URI string suitable for multimodal APIs.

    Processing pipeline (when PIL is available):
    1. Open image and composite RGBA onto a white RGB background.
    2. If ``max_side > 0`` and the longest side exceeds it, thumbnail-resize.
    3. Re-encode as JPEG with *quality*.
    4. Base64-encode and return as ``data:image/jpeg;base64,<...>``.

    When PIL is not installed **or** ``max_side == 0``, the raw file bytes are
    returned encoded as their native MIME type (PNG/JPEG/WebP).

    Requires: ``Pillow`` (``pip install Pillow``) for resize/recompress path.

    Args:
        path:     Path to the source image file.
        max_side: Maximum pixels on the longest side (0 = no resize).
        quality:  JPEG quality (1–95) used when re-encoding.

    Returns:
        A ``data:<mime>;base64,<data>`` string.
    """
    if _PILImage is None or max_side <= 0:
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    with _PILImage.open(path) as img:
        img = img.convert("RGBA")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        bg = _PILImage.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        buf = io.BytesIO()
        bg.save(buf, format="JPEG", quality=quality, optimize=True)

    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 4. Prompt Constants
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTIONS: dict[str, str] = {
    "numerical": "Output the final answer as a single number.",
    "mcq": "Output only the letter(s) of the correct option(s).",
    "boolean": "Output only Yes or No.",
    "free_text": "Output a concise answer.",
}
"""Per-type answer format instructions injected into prompts."""


# ---------------------------------------------------------------------------
# 5. Retry Helper
# ---------------------------------------------------------------------------


def call_with_retry(
    fn: Callable,
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type, ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential-backoff retries on transient failures.

    Args:
        fn:          The callable to invoke.
        *args:       Positional arguments forwarded to *fn*.
        max_retries: Total number of attempts (1 = no retry).
        base_delay:  Base delay in seconds for the first backoff interval.
                     Actual delay after attempt *k* is ``base_delay * 2^(k-1)``.
        jitter:      When ``True``, adds uniform random jitter of up to 30 %
                     of the computed delay.
        retry_on:    Tuple of exception types that trigger a retry.  All other
                     exceptions propagate immediately.
        **kwargs:    Keyword arguments forwarded to *fn*.

    Returns:
        The return value of the first successful *fn* call.

    Raises:
        The last exception encountered if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except retry_on as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            if jitter:
                delay += delay * random.uniform(0.0, 0.3)
            logger.debug(
                "call_with_retry: attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
