"""
FinanceExtractionSkill

Design goal for the python-first baseline:
- be a narrow, high-confidence structured extractor
- help on OCR-friendly rectangular tables with unique row-column intersections
- help on table-like footnotes where anchors survive OCR as text
- avoid becoming a universal visual rescue path
- avoid cases that require local visual disambiguation across repeated labels,
  totals/subtotals, date-like columns, or blank carry-over rows

Default policy:
1. OCR the current image
2. If OCR looks table-like, try deterministic row/column parsing first
3. For unresolved variables, use a text-only LLM over OCR text as a constrained fallback
4. Save extracted variables into shared python state

It does NOT:
- replace direct chart reading
- do broad visual reasoning for arbitrary charts/legends
- own the full multimodal routing policy
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from finacumen.ft.config import config
from finacumen.ft.flow.extraction_state_bridge import (
    get_current_extraction_context,
    persist_extracted_values_to_shared_python,
    register_variable_semantics,
    validate_extraction_against_current_request,
)
from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.tool.base import BaseTool
from finacumen.ft.tool.ocr import OcrExtract, get_step_images_for_ocr


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _normalize_var_key(text: str) -> str:
    s = _normalize_text(text).replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _looks_like_placeholder_base64(base64_image: str) -> bool:
    s = (base64_image or "").strip()
    if not s:
        return True
    if s.startswith("data:"):
        s = s.split(",", 1)[-1].strip()
    # Tiny 1x1 / placeholder payloads should not override the real step image.
    return len(s) < 512


def _apply_unit_conversion(raw: str) -> Optional[float]:
    text = (raw or "").strip()
    if not text:
        return None
    compact = re.sub(r"\s+", "", text)
    negative = False
    if re.fullmatch(r"\(.*\)", compact):
        negative = True
        compact = compact[1:-1]
    if compact.startswith("-"):
        negative = True
        compact = compact[1:]
    compact = compact.replace("$", "").replace("€", "").replace("¥", "").replace(",", "")
    compact = compact.rstrip(".")

    multiplier = 1.0
    lower = compact.lower()
    if lower.endswith("%"):
        compact = compact[:-1]
    elif "billion" in lower:
        multiplier = 1000.0
        compact = re.sub(r"billion", "", compact, flags=re.IGNORECASE)
    elif "million" in lower:
        multiplier = 1.0
        compact = re.sub(r"million", "", compact, flags=re.IGNORECASE)
    elif "thousand" in lower:
        multiplier = 0.001
        compact = re.sub(r"thousand", "", compact, flags=re.IGNORECASE)
    elif compact.endswith("亿"):
        multiplier = 100000000.0
        compact = compact[:-1]
    elif compact.endswith("万"):
        multiplier = 10000.0
        compact = compact[:-1]

    compact = compact.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return None
    value = float(compact) * multiplier
    return -value if negative else value


_RATE_KEYWORD_RE = re.compile(
    r"\b(rate|margin|yield|percentage|growth|return|ratio|discount|interest|coupon|spread|premium)\b",
    re.IGNORECASE,
)
_CURRENCY_VETO_RE = re.compile(
    r"\b(price|cost|fee|value|amount|per\s*share|average\s*price|revenue|"
    r"income|expense|salary|wage|payment|proceeds|principal|balance|"
    r"deposit|dividend|earning|profit|loss|cash|debt|asset|liability|"
    r"equity|capital|fund|budget|rent|tax)\b",
    re.IGNORECASE,
)
_CURRENCY_SYMBOL_RE = re.compile(r"[$€¥£]")


def _has_currency_veto(raw_text: str, var_name: str, header: str = "") -> bool:
    """Return True if the context has currency/price semantics that VETO percentage normalization."""
    scope = f"{raw_text} {var_name} {header}"
    if _CURRENCY_SYMBOL_RE.search(raw_text):
        return True
    if _CURRENCY_VETO_RE.search(scope):
        return True
    return False


def _count_percent_positive_signals(
    raw_text: str, var_name: str, header: str = ""
) -> int:
    """Count how many positive signals indicate this value is a percentage/rate.

    Returns 0-3. The caller should require >= 2 for confident conversion.
    """
    signals = 0
    if "%" in raw_text:
        signals += 1
    if _RATE_KEYWORD_RE.search(header):
        signals += 1
    if _RATE_KEYWORD_RE.search(var_name.replace("_", " ")):
        signals += 1
    return signals


def _should_normalize_as_percent(
    raw_text: str, var_name: str, header: str = ""
) -> bool:
    """Determine whether a value should be normalized as a percentage (x / 100).

    Requires at least 2 positive signals AND no currency/price veto.
    """
    if _has_currency_veto(raw_text, var_name, header):
        return False
    return _count_percent_positive_signals(raw_text, var_name, header) >= 2


_PERIOD_NOTATION_RE = re.compile(
    r"^\d{2,4}[QqHh][1-4]$"       # 24Q1, 2024Q3, 23H1
    r"|^[QqHh][1-4]\d{2,4}$"      # Q12024, H22023
    r"|^\d{2,4}[年]?[QqHh][1-4]$"  # 24年Q1
    r"|^FY\d{2,4}$"               # FY2024, FY19
    , re.IGNORECASE,
)


def _extract_numeric_from_cell(cell: str) -> Optional[float]:
    cell = (cell or "").strip()
    if not cell:
        return None
    # Reject cells that are period/quarter notations (e.g. 24Q1, 23Q3)
    cell_compact = re.sub(r"[\s\-*/]+", "", cell)
    if _PERIOD_NOTATION_RE.match(cell_compact):
        return None
    candidates = re.findall(
        r"\(\$?\d[\d,]*(?:\.\d+)?%?\)|-?\$?\d[\d,]*(?:\.\d+)?%?",
        cell,
    )
    for cand in candidates:
        val = _apply_unit_conversion(cand)
        if val is not None:
            return val
    return None


def _row_match_score(row_label: str, row_keyword: str) -> int:
    """Score how well *row_label* matches *row_keyword*.

    Returns 0 for no match; higher is better:
      3 — exact normalised match
      2 — one string is a substring of the other (≥3 chars)
      1 — sufficient token-level overlap (word-boundary)
    """
    row_norm = _normalize_text(row_label)
    kw_norm = _normalize_text(row_keyword)
    if not row_norm or not kw_norm:
        return 0
    if kw_norm == row_norm:
        return 3
    # Short labels (1-2 chars like "A", "B"): require the label to appear
    # as a standalone token in the keyword to avoid false substring hits.
    if len(row_norm) <= 2:
        kw_tokens_all = re.split(r"[\s_/,-]+", kw_norm)
        return 3 if row_norm in kw_tokens_all else 0
    # Substring containment (row_norm already guaranteed >= 3 chars)
    if kw_norm in row_norm and len(kw_norm) >= 3:
        return 2
    if row_norm in kw_norm:
        return 2
    # Token overlap with word-boundary matching to prevent partial-word
    # false positives (e.g. "the" matching inside "other").
    kw_tokens = [t for t in re.split(r"[\s_/,-]+", kw_norm) if len(t) >= 2]
    if not kw_tokens:
        return 0
    hit_count = sum(
        1 for token in kw_tokens
        if re.search(r"\b" + re.escape(token) + r"\b", row_norm)
    )
    if hit_count >= max(1, min(2, len(kw_tokens))):
        return 1
    return 0


def _row_matches_keywords(row_label: str, row_keyword: str) -> bool:
    return _row_match_score(row_label, row_keyword) > 0


def _detect_header_row_count(table_rows: List[List[str]]) -> int:
    """Detect how many leading rows are column headers (not data).

    A header row has predominantly non-numeric text or year labels in the
    data columns (all columns except the first, which is the row-label column).
    Stops as soon as a data-like row (majority numeric, non-year) is found.
    """
    if len(table_rows) <= 1:
        return min(len(table_rows), 1)

    header_end = 1
    for i in range(1, min(len(table_rows), 5)):
        row = table_rows[i]
        data_cells = row[1:] if len(row) > 1 else []
        if not data_cells:
            break
        non_empty = [c for c in data_cells if c.strip()]
        if not non_empty:
            first_cell = (row[0] if row else "").strip()
            if first_cell and not first_cell.endswith(":"):
                # Likely a spanning header label (e.g. "Gross Unrealized | | | |").
                header_end = i + 1
                continue
            # Looks like a section label (e.g. "Net sales: | | | |"). Stop.
            break

        year_count = sum(
            1 for c in non_empty if re.fullmatch(r"(?:19|20)\d{2}", c.strip())
        )
        numeric_count = sum(
            1
            for c in non_empty
            if _extract_numeric_from_cell(c) is not None
            and not re.fullmatch(r"(?:19|20)\d{2}", c.strip())
        )
        non_year_total = len(non_empty) - year_count

        if non_year_total > 0 and numeric_count / non_year_total >= 0.5:
            break
        header_end = i + 1

    return header_end


def _build_markdown_table_headers(table_rows: List[List[str]]) -> Tuple[List[str], int]:
    """Build flattened column headers from potentially multi-level header rows.

    For tables with stacked headers (e.g. year on row 1, metric on row 2),
    fills forward across empty cells in each header row, then concatenates
    vertically per column to produce composite headers like ``2010 Fair Value``.
    Single-header tables are returned unchanged.
    """
    if not table_rows:
        return [], 0

    header_count = _detect_header_row_count(table_rows)

    if header_count <= 1:
        return table_rows[0], 1

    header_rows = table_rows[:header_count]
    max_cols = max(len(r) for r in header_rows)

    filled: List[List[str]] = []
    for row in header_rows:
        padded = row + [""] * (max_cols - len(row))
        result = [padded[0]]
        last = ""
        for j in range(1, len(padded)):
            cell = padded[j].strip()
            if cell:
                last = cell
            result.append(last)
        filled.append(result)

    flattened: List[str] = []
    for j in range(max_cols):
        parts: List[str] = []
        seen: set = set()
        for row in filled:
            if j < len(row):
                val = row[j].strip()
                key = val.lower()
                if val and key not in seen:
                    parts.append(val)
                    seen.add(key)
        flattened.append(" ".join(parts))

    return flattened, header_count


def _is_section_header_row(cells: List[str]) -> bool:
    """True if the row is a section/category label with no numeric data."""
    if not cells or not cells[0].strip():
        return False
    data_cells = cells[1:]
    if not data_cells:
        return True
    non_empty = [
        c
        for c in data_cells
        if c.strip() and c.strip() not in ("\u2014", "-", "\u2013", "\u2015", "")
    ]
    numeric = sum(1 for c in non_empty if _extract_numeric_from_cell(c) is not None)
    return numeric == 0


def _detect_header_column_shift(
    headers: List[str], data_rows: List[List[str]]
) -> bool:
    """Detect if the header row is missing a row-label column, causing a
    1-position shift relative to data rows.

    Common OCR artifact::

        Headers: ["2019", "2018", "2017", ""]
        Data:    ["United Kingdom", "99,825", "91,426", "70,163"]

    When this happens, headers[i] aligns with data[i] but semantically
    should align with data[i+1].  Detected when:
    1. The first header cell looks like a year / date (not a row descriptor).
    2. The first cell of most data rows is a non-numeric text label.
    """
    if not headers or not data_rows:
        return False

    first_h = headers[0].strip()
    if not first_h:
        return False

    first_h_lower = first_h.lower()
    is_pure_year = bool(
        re.fullmatch(r"(?:fy\s*)?(?:19|20)\d{2}[a-z]?", first_h.strip(), re.IGNORECASE)
    )
    is_year_prefix = bool(
        re.match(r"(?:fy\s*)?(?:19|20)\d{2}[a-z]?\b", first_h_lower, re.IGNORECASE)
    )
    is_date_phrase = bool(
        re.match(
            r"(?:january|february|march|april|may|june|july|august"
            r"|september|october|november|december|fiscal|fy)\b",
            first_h_lower,
        )
    )

    if not (is_pure_year or is_year_prefix or is_date_phrase):
        return False

    checked = 0
    text_count = 0
    for row in data_rows[:8]:
        if not row or not row[0].strip():
            continue
        checked += 1
        if _extract_numeric_from_cell(row[0]) is None:
            text_count += 1

    return checked >= 1 and text_count >= checked * 0.6


def _score_column_match(col_kw: str, kw_tokens: set, h_norm: str) -> float:
    """Score how well a column header matches the keyword. Higher = better.

    Supports exact match, substring containment, token-subset (for
    multi-level flattened headers like ``2010 Fair Value``), and year match.
    """
    if col_kw == h_norm:
        return 1000

    if col_kw in h_norm:
        return 500 + len(col_kw) / max(len(h_norm), 1) * 100
    if h_norm in col_kw:
        return 400 + len(h_norm) / max(len(col_kw), 1) * 100

    h_tokens = set(t for t in re.split(r"[\s_]+", h_norm) if len(t) >= 2)
    if kw_tokens and kw_tokens.issubset(h_tokens):
        return 300 + len(kw_tokens) / max(len(h_tokens), 1) * 100

    if re.fullmatch(r"(?:19|20)\d{2}[a-z]?", col_kw):
        year = re.sub(r"[^0-9]", "", col_kw)
        if year and re.search(rf"\b{re.escape(year)}[a-z]?\b", h_norm):
            return 50

    return -1


def _resolve_inline_year_group_col(
    col_keyword: str, headers: List[str]
) -> int:
    """Resolve column index when a single header row contains inline year groups.

    Handles tables like::

        2001 | High | Low | 2000 | High | Low

    For col_keyword "2000 high", returns 4 (the "High" after "2000").
    """
    col_kw_norm = _normalize_text(col_keyword)
    m_year = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", col_kw_norm)
    if not m_year:
        return -1
    year = m_year.group(1)
    metric_kw = re.sub(r"\b(?:19|20)\d{2}[a-z]?\b", "", col_kw_norm).strip()
    if not metric_kw:
        return -1

    # Collect year positions in the header
    year_positions: List[tuple] = []
    for i, h in enumerate(headers):
        h_norm = _normalize_text(h)
        if h_norm and re.fullmatch(r"(?:fy\s*)?(?:19|20)\d{2}[a-z]?", h_norm):
            year_positions.append((h_norm, i))

    if len(year_positions) < 2:
        return -1

    # Find the target year's column range
    target_start = None
    target_end = len(headers) - 1
    for idx, (yr, pos) in enumerate(year_positions):
        if yr == year:
            target_start = pos
            if idx + 1 < len(year_positions):
                target_end = year_positions[idx + 1][1] - 1
            break

    if target_start is None:
        return -1

    # Search for the metric within this year's group
    for i in range(target_start, target_end + 1):
        if i < len(headers):
            h_norm = _normalize_text(headers[i])
            if h_norm and (metric_kw == h_norm or metric_kw in h_norm or h_norm in metric_kw):
                return i
    return -1


def _select_col_index_from_grouped_headers(
    col_keyword: str, header_rows: List[List[str]]
) -> int:
    """Resolve year+metric columns from grouped/repeated sub-headers.

    Handles tables like:
      2011 | 2010
      Fair Value | Amortized Cost | Gains | Losses | Fair Value | ...

    where flattened headers may lose year labels for the second metric block.
    """
    if not col_keyword or len(header_rows) < 2:
        return -1

    col_kw_norm = _normalize_text(col_keyword)
    m_year = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", col_kw_norm)
    if not m_year:
        return -1
    year = m_year.group(1)
    metric_kw = re.sub(r"\b(?:19|20)\d{2}[a-z]?\b", "", col_kw_norm).strip()
    if not metric_kw:
        return -1

    # Last header row usually carries per-column metric labels.
    metric_row_norm = [_normalize_text(c) for c in header_rows[-1]]
    metric_hits = [
        i
        for i, cell in enumerate(metric_row_norm)
        if cell and (metric_kw == cell or metric_kw in cell or cell in metric_kw)
    ]
    if len(metric_hits) < 2:
        return -1

    years_order: List[str] = []
    for row in header_rows:
        for cell in row:
            cell_norm = _normalize_text(cell)
            if re.fullmatch(r"(?:19|20)\d{2}[a-z]?", cell_norm) and cell_norm not in years_order:
                years_order.append(cell_norm)

    if len(years_order) < 2 or year not in years_order:
        return -1

    year_idx = years_order.index(year)
    if year_idx >= len(metric_hits):
        return -1
    return metric_hits[year_idx]


def _match_header_to_var_hint(
    headers: List[str], var_name: str
) -> Tuple[int, str]:
    """Infer target column from variable name when no explicit col_keyword.

    Uses compact substring matching to handle concatenated variable names like
    ``milesofpipeline_elba_express`` where ``milesofpipeline`` matches header
    ``Miles of Pipeline``.

    Returns ``(col_index, matched_header)`` or ``(-1, "")``.
    """
    if not headers or not var_name:
        return -1, ""

    var_compact = re.sub(r"[\s_]+", "", var_name.lower())
    var_tokens = set(
        t for t in re.split(r"[\s_]+", var_name.lower()) if len(t) >= 3
    )

    best_idx = -1
    best_score = 0.0

    for i, h in enumerate(headers):
        if i == 0:
            continue
        h_norm = _normalize_text(h)
        if not h_norm:
            continue
        if re.fullmatch(r"(?:fy\s*)?(?:19|20)\d{2}[a-z]?", h_norm.strip()):
            continue
        h_compact = re.sub(r"[\s_]+", "", h_norm)
        if len(h_compact) < 5:
            continue

        score = 0.0
        if h_compact in var_compact:
            score = float(len(h_compact))

        if score == 0:
            h_tokens = set(
                t for t in re.split(r"[\s_]+", h_norm) if len(t) >= 3
            )
            overlap = var_tokens & h_tokens
            if len(overlap) >= 2:
                score = float(len(overlap)) * 3

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx >= 0 and best_score >= 5:
        return best_idx, headers[best_idx]
    return -1, ""


def _parse_markdown_table_for_value(
    text: str, row_keyword: str, col_keyword: str, var_name: str = ""
) -> Tuple[Optional[float], str, str, bool, str]:
    """Return (value, raw_cell, matched_header, used_explicit_col, matched_row_label)."""
    lines = text.splitlines()
    table_rows: List[List[str]] = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if any(c for c in cells if c):
            table_rows.append(cells)

    if not table_rows:
        return None, "", "", False, ""

    headers, data_start = _build_markdown_table_headers(table_rows)
    data_rows = table_rows[data_start:]
    header_rows = table_rows[:data_start] if data_start > 0 else []

    header_shift_applied = _detect_header_column_shift(headers, data_rows)
    if header_shift_applied:
        headers = [""] + headers
        logger.info(
            "[finance_extraction_skill] header column shift detected — "
            "prepended empty row-label column to align with data rows"
        )

    # --- Column matching with scoring (supports flattened multi-level headers) ---
    col_index = -1
    used_explicit_col = False
    matched_header = ""
    if col_keyword:
        # Priority 1: multi-level grouped headers (2+ header rows)
        grouped_idx = _select_col_index_from_grouped_headers(col_keyword, header_rows)
        if grouped_idx != -1 and header_shift_applied:
            grouped_idx += 1
        if grouped_idx != -1 and grouped_idx < len(headers):
            col_index = grouped_idx
            matched_header = headers[grouped_idx]
            used_explicit_col = True
        else:
            # Priority 2: inline year groups in a single header row
            # e.g., ["2001", "High", "Low", "2000", "High", "Low"]
            inline_idx = _resolve_inline_year_group_col(col_keyword, headers)
            if inline_idx != -1 and inline_idx < len(headers):
                col_index = inline_idx
                matched_header = headers[inline_idx]
                used_explicit_col = True
                logger.info(
                    f"[finance_extraction_skill] inline year-group resolved: "
                    f"col_keyword={col_keyword!r} → col {inline_idx} ({matched_header!r})"
                )

        if col_index == -1:
            # Priority 3: general column scoring
            col_kw = _normalize_text(col_keyword)
            kw_tokens = set(t for t in re.split(r"[\s_]+", col_kw) if len(t) >= 2)
            best_score = -1
            for i, h in enumerate(headers):
                h_norm = _normalize_text(h)
                if not h_norm:
                    continue
                score = _score_column_match(col_kw, kw_tokens, h_norm)
                if score >= 1000:
                    col_index, matched_header, used_explicit_col = i, h, True
                    best_score = score
                    break
                if score > best_score:
                    best_score = score
                    col_index, matched_header, used_explicit_col = i, h, True

            if best_score < 0:
                col_index, matched_header, used_explicit_col = -1, "", False

    # --- Row matching: collect all candidates, pick best score ---
    # Tuple: (match_score, section_relevance, cells)
    _RowHit = Tuple[int, int, List[str]]
    candidates: List[_RowHit] = []
    section_header = ""
    for cells in data_rows:
        if not cells:
            continue

        row_label = cells[0].strip()

        if _is_section_header_row(cells):
            section_header = row_label
            continue

        score = _row_match_score(row_label, row_keyword)
        if not score and section_header:
            section_norm = _normalize_text(section_header)
            kw_norm = _normalize_text(row_keyword)
            section_tokens = set(re.split(r"[\s_/,\-]+", section_norm))
            remaining = [
                t
                for t in re.split(r"[\s_/,\-]+", kw_norm)
                if t not in section_tokens and len(t) >= 2
            ]
            if remaining:
                score = _row_match_score(row_label, " ".join(remaining))

        if score:
            sec_score = _row_match_score(section_header, row_keyword) if section_header else 0
            candidates.append((score, sec_score, cells))

    # Sort by descending match quality; section relevance breaks ties.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # When no explicit col_keyword, try to infer column from variable name
    inferred_col_index = -1
    inferred_col_header = ""
    if not col_keyword and var_name and col_index == -1:
        inferred_col_index, inferred_col_header = _match_header_to_var_hint(
            headers, var_name
        )
        if inferred_col_index >= 0:
            logger.info(
                "[finance_extraction_skill] inferred column from var_name %r: "
                "col %d (%r)",
                var_name,
                inferred_col_index,
                inferred_col_header,
            )

    for _, _, cells in candidates:
        row_label = cells[0].strip()
        if col_index != -1 and col_index < len(cells):
            num = _extract_numeric_from_cell(cells[col_index])
            if num is not None:
                return num, cells[col_index], matched_header, used_explicit_col, row_label
        elif inferred_col_index != -1 and inferred_col_index < len(cells):
            num = _extract_numeric_from_cell(cells[inferred_col_index])
            if num is not None:
                return (
                    num,
                    cells[inferred_col_index],
                    inferred_col_header,
                    True,
                    row_label,
                )
        elif not col_keyword:
            for cell in cells[1:]:
                num = _extract_numeric_from_cell(cell)
                if num is not None:
                    return num, cell, "", False, row_label
    return None, "", "", False, ""


def _is_column_header_consistent(col_keyword: str, matched_header: str) -> bool:
    col_kw = _normalize_text(col_keyword)
    header = _normalize_text(matched_header)
    if not col_kw:
        return True
    if not header:
        return False
    if col_kw == header or col_kw in header or header in col_kw:
        return True
    col_year = re.sub(r"[^0-9]", "", col_kw)
    head_year = re.sub(r"[^0-9]", "", header)
    if col_year and head_year and col_year == head_year:
        return True
    kw_tokens = set(t for t in re.split(r"[\s_]+", col_kw) if len(t) >= 2)
    h_tokens = set(t for t in re.split(r"[\s_]+", header) if len(t) >= 2)
    if kw_tokens and kw_tokens.issubset(h_tokens):
        return True
    return False


_SUMMARY_ROW_PATTERNS = re.compile(
    r"^(?:total|subtotal|sub-total|net total|grand total|合计|小计|"
    r"ending balance|beginning balance|thereafter|总计|总额|"
    r"total[\s_].*|.*[\s_]total)$",
    re.IGNORECASE,
)


def _is_summary_row_label(label: str) -> bool:
    """Check if a row label looks like a summary/total row."""
    norm = _normalize_text(label).strip("- *")
    return bool(_SUMMARY_ROW_PATTERNS.match(norm))


def _detect_duplicate_value_anomaly(
    extracted: Dict[str, float],
) -> List[str]:
    """Return variable names involved in suspicious duplicate values.

    Two different variables with the exact same extracted value is a strong
    signal for row/column mis-alignment.  Only flags when there are 2+
    variables to compare.
    """
    if len(extracted) < 2:
        return []
    val_to_vars: Dict[float, List[str]] = {}
    for var, val in extracted.items():
        val_to_vars.setdefault(val, []).append(var)
    suspicious: List[str] = []
    for val, vars_list in val_to_vars.items():
        if len(vars_list) >= 2:
            suspicious.extend(vars_list)
    return suspicious


def _validate_row_match_quality(
    var_name: str,
    row_keyword: str,
    matched_row_label: str,
) -> Optional[str]:
    """Return a warning string if the matched row label is semantically
    inconsistent with the expected row keyword.

    Uses token-level analysis to catch cases like expecting "other assets"
    but matching to "prepaid expenses and other current assets".
    """
    if not matched_row_label or not row_keyword:
        return None

    rk_norm = _normalize_text(row_keyword)
    ml_norm = _normalize_text(matched_row_label).strip("- *()[]")

    if rk_norm == ml_norm:
        return None

    _FILLER = frozenset({
        "the", "a", "an", "of", "and", "or", "in", "for", "to", "by",
        "at", "on", "is", "its", "as", "net", "per",
    })

    kw_tokens = set(
        t for t in re.split(r"[\s_/,\-()]+", rk_norm) if len(t) >= 2 and t not in _FILLER
    )
    ml_tokens = set(
        t for t in re.split(r"[\s_/,\-()]+", ml_norm) if len(t) >= 2 and t not in _FILLER
    )

    if not kw_tokens or not ml_tokens:
        return None

    extra_tokens = ml_tokens - kw_tokens
    missing_tokens = kw_tokens - ml_tokens

    # The matched label has substantial content not in the keyword.
    # This suggests a different (more specific or different) row.
    if len(extra_tokens) >= 2 and len(extra_tokens) > len(kw_tokens):
        return (
            f"row_keyword='{row_keyword}' matched to '{matched_row_label}' "
            f"which has significant extra content: [{', '.join(sorted(extra_tokens))}]"
        )

    # The keyword has tokens not present in the matched label at all,
    # indicating the match was partial / potentially wrong row.
    if missing_tokens and len(missing_tokens) >= len(kw_tokens) * 0.5:
        return (
            f"row_keyword='{row_keyword}' matched to '{matched_row_label}' "
            f"but keyword tokens [{', '.join(sorted(missing_tokens))}] are missing"
        )

    return None


_COLUMN_HINT_RE = re.compile(
    r"\b(column|col|actual|constant|fair value|amortized cost|"
    r"gain(?:s)?|loss(?:es)?|high|low|ending balance|beginning balance)\b",
    re.IGNORECASE,
)


def _query_implies_explicit_column(semantic_query: str) -> bool:
    text = (semantic_query or "").strip()
    if not text:
        return False
    if re.search(r"(?:column|col)\s*[:'\" ]", text, re.IGNORECASE):
        return True
    return bool(_COLUMN_HINT_RE.search(text))


def _parse_variable_semantics(var_name: str, semantic_query: str = "") -> Tuple[str, str]:
    text = (semantic_query or var_name or "").strip()
    if not text:
        return "", ""

    # Structured suffix pattern for grouped table columns:
    #   <entity>_<metric>_<year>  -> row=<entity>, col="<year> <metric>"
    # Example:
    #   corporate_notes_bonds_fair_value_2010
    #   -> row="corporate notes bonds", col="2010 fair value"
    # This stabilizes multi-level headers where year repeats across sub-columns.
    var_lower = (var_name or "").strip().lower()
    m_var_year = re.search(r"^(.*?)[_ ]((?:19|20)\d{2}[a-z]?)$", var_lower)
    if m_var_year:
        stem = m_var_year.group(1).strip(" _")
        year = m_var_year.group(2).strip()
        metric_suffixes = [
            ("net_income_effect", "net income effect"),
            ("amortized_cost", "amortized cost"),
            ("fair_value", "fair value"),
            ("gain_on_swaps", "gain on swaps"),
            ("loss_on_swaps", "loss on swaps"),
            ("gain_on_note", "gain on note"),
            ("loss_on_note", "loss on note"),
            ("gains", "gains"),
            ("losses", "losses"),
            ("actual", "actual"),
            ("constant", "constant"),
        ]
        for suffix, metric_label in metric_suffixes:
            marker = "_" + suffix
            # Check metric as suffix: <entity>_<metric>_<year>
            if stem.endswith(marker):
                row_part = stem[: -len(marker)].strip(" _")
                if row_part:
                    row = row_part.replace("_", " ").strip()
                    col = f"{year} {metric_label}".strip()
                    if row and col:
                        return row, col
            # Check metric as prefix: <metric>_<entity>_<year>
            prefix = suffix + "_"
            if stem.startswith(prefix):
                entity_part = stem[len(prefix):].strip(" _")
                if entity_part:
                    row = entity_part.replace("_", " ").strip()
                    col = f"{year} {metric_label}".strip()
                    if row and col:
                        return row, col

    # Common structured naming pattern: metric_for_entity -> row=entity, col=metric.
    var_name_norm = (var_name or "").strip()
    m_for = re.fullmatch(
        r"([a-z][a-z0-9_]*)_for_([a-z][a-z0-9_]*)",
        var_name_norm,
        re.IGNORECASE,
    )
    if m_for:
        metric = m_for.group(1).replace("_", " ").strip()
        entity = m_for.group(2).replace("_", " ").strip()
        if metric and entity:
            return entity, metric

    csv_parts = [
        re.sub(r"\s+", " ", part).strip().strip("'\"")
        for part in re.split(r"[，,、]+", text)
        if re.sub(r"\s+", " ", part).strip().strip("'\"")
    ]
    if len(csv_parts) >= 3:
        return " ".join(csv_parts[:-1]).strip(), csv_parts[-1].strip()

    text_lower = text.lower().replace("_", " ")

    m_rc = re.search(
        r"row\s*:\s*([^,|]+?)\s*,\s*(?:column|col)\s*:\s*([^,|\n]+)",
        text_lower,
        re.IGNORECASE,
    )
    if m_rc:
        return m_rc.group(1).strip(), m_rc.group(2).strip()

    # row 'X' (and|,) column 'Y' — explicit quoted anchors from step text
    # Supports both "column 'Y'" and "'Y' column" orderings.
    m_row_col = re.search(
        r"""row\s+(?:labeled\s+)?['"]([^'"]+)['"]\s*(?:,?\s*and\s+)?"""
        r"""(?:column\s+['"]([^'"]+)['"]|['"]([^'"]+)['"]\s+column)""",
        text,
        re.IGNORECASE,
    )
    if m_row_col:
        row = m_row_col.group(1).strip()
        col = (m_row_col.group(2) or m_row_col.group(3)).strip()
        return row, col

    # column 'Y' alone (row inferred from for/of 'X' or var_name)
    # Matches both "column 'Y'" and "'Y' column".
    m_col_quoted = re.search(
        r"""(?:column\s+['"]([^'"]+)['"]|['"]([^'"]+)['"]\s+column)""",
        text,
        re.IGNORECASE,
    )
    if m_col_quoted:
        col = (m_col_quoted.group(1) or m_col_quoted.group(2)).strip()
        m_for_row = re.search(
            r"""(?:for|of)\s+['"]([^'"]+)['"]""",
            text,
            re.IGNORECASE,
        )
        if m_for_row:
            row = m_for_row.group(1).strip()
        else:
            row = re.sub(
                r"_?(total|sum|net|\d{4}|f\d{2}|fy\d{2,4})$", "",
                (var_name or "").lower(),
            ).replace("_", " ").strip()
            if not row:
                row = text_lower
        return row, col

    m_as_of = re.search(
        r"(.+?)\s+as\s+of\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_as_of:
        metric = re.sub(r"\s+", " ", m_as_of.group(1)).strip(" ,.")
        period = re.sub(r"\s+", " ", m_as_of.group(2)).strip(" ,.")
        if metric and period:
            return metric, period

    m_for_entity_year = re.search(
        r"(.+?)\s+for\s+(.+?)\s+in\s+((?:19|20)\d{2}[a-z]?)",
        text_lower,
        re.IGNORECASE,
    )
    if m_for_entity_year:
        return m_for_entity_year.group(2).strip(), m_for_entity_year.group(3).strip()

    years = re.findall(r"(?:19|20)\d{2}[a-z]?", text_lower)
    if years:
        # Prefer the year from var_name to avoid multi-year pollution when
        # a bundled semantic query mentions years for other variables.
        var_years = re.findall(
            r"(?:19|20)\d{2}[a-z]?",
            (var_name or "").lower().replace("_", " "),
        )
        if var_years and var_years[0] in years:
            year = var_years[0]
        else:
            year = years[-1]
        metric = re.sub(r"(?:19|20)\d{2}[a-z]?", "", text_lower).strip(" ,.")
        _COMPOUND_VALUE_TERMS = {
            "fair value", "book value", "market value", "net value",
            "gross value", "present value", "future value", "par value",
            "face value", "carrying value", "notional value",
        }
        has_compound = any(cv in metric for cv in _COMPOUND_VALUE_TERMS)
        if not has_compound:
            metric = re.sub(r"\b(value|amount|figure|number)\b", "", metric)
        metric = metric.strip(" ,.")
        if metric:
            return metric, year

    # "the METRIC for ENTITY (from/in ...)" — non-year entity extraction
    m_metric_entity = re.search(
        r"(?:the\s+)?(.+?)\s+for\s+(.+?)(?:\s+(?:from|in)\s+.+)?\s*[,.]?\s*$",
        text.strip(),
        re.IGNORECASE,
    )
    if m_metric_entity:
        metric = m_metric_entity.group(1).strip()
        entity = m_metric_entity.group(2).strip()
        if (
            metric
            and entity
            and len(metric.split()) <= 5
            and len(entity.split()) <= 5
            and not re.fullmatch(r"(?:19|20)\d{2}[a-z]?", entity)
            and len(entity) >= 2
        ):
            return entity, metric

    # Fiscal year abbreviations: f19 -> 2019, fy18 -> 2018
    fy_matches = re.findall(r"\bf(?:y)?(\d{2,4})\b", text_lower)
    if fy_matches:
        short = fy_matches[-1]
        if len(short) == 2:
            year = ("20" if int(short) < 50 else "19") + short
        else:
            year = short
        metric = re.sub(r"\bf(?:y)?\d{2,4}\b", "", text_lower).strip(" ,.")
        metric = re.sub(r"\b(value|amount|figure|number)\b", "", metric).strip(" ,.")
        if metric:
            return metric, year

    # "_total" / "_sum" suffix in var_name as column hint
    if var_name:
        m_total_suffix = re.search(r"^(.+?)_(total|sum)$", var_name.lower())
        if m_total_suffix:
            row = m_total_suffix.group(1).replace("_", " ").strip()
            col = m_total_suffix.group(2)
            if row:
                return row, col

    return text, ""


def _parse_decoupled_extraction(step_text: str) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    # Capture comma-separated var list after save_as (e.g. "save_as v1, v2")
    pattern = (
        r"extract\s+(.+?)\s+save_as\s+"
        r"([a-z_][a-z0-9_]*(?:\s*,\s*[a-z_][a-z0-9_]*)*)"
    )
    for match in re.finditer(pattern, step_text or "", re.IGNORECASE):
        semantic_clause = match.group(1).strip()
        var_list_str = match.group(2).strip()
        variables = [v.strip() for v in var_list_str.split(",") if v.strip()]
        if not variables:
            continue

        if len(variables) == 1:
            sq = re.sub(r"[\"']", "", semantic_clause).strip()
            sq = re.sub(r"\s+", " ", sq).strip().strip(",.")
            if sq:
                result.append((sq, variables[0]))
        else:
            # Multiple vars: try to split semantic clause per variable.
            # Look for sub-clauses like "var_name from row 'X' and column 'Y'"
            for var in variables:
                sub_pat = (
                    re.escape(var) + r"\s+(.+?)(?=\s*,\s*[a-z_][a-z0-9_]*\s|$)"
                )
                sub_m = re.search(sub_pat, semantic_clause, re.IGNORECASE)
                if sub_m:
                    sub_sq = sub_m.group(1).strip().strip(",.")
                    sub_sq = re.sub(r"[\"']", "", sub_sq).strip()
                    sub_sq = re.sub(r"\s+", " ", sub_sq).strip().strip(",.")
                    if sub_sq:
                        result.append((f"{var} {sub_sq}", var))
                        continue
                # Fallback: use var name alone (let _parse_variable_semantics
                # handle it from the variable name).
                result.append((var, var))
    return result


_OUTPUT_VAR_STOPWORDS = frozenset({
    "numerical", "variable", "text", "conclusion", "value", "result",
    "output", "number", "string", "the", "a", "an", "for", "with",
    "from", "and", "or", "as", "to", "in", "of", "by", "is", "it",
})


def _parse_output_as_variables(step_text: str) -> List[str]:
    text = step_text or ""
    matches = re.findall(r"output\s+as\s+([a-z_][a-z0-9_,\s]*)", text, re.IGNORECASE)
    variables: List[str] = []
    for match in matches:
        for part in re.split(r"[,\s]+", match):
            part = part.strip()
            if (
                re.fullmatch(r"[a-z_][a-z0-9_]*", part)
                and part not in _OUTPUT_VAR_STOPWORDS
                and "_" in part
            ):
                variables.append(part)
    return list(dict.fromkeys(variables))


def _get_semantic_queries(step_text: str) -> Dict[str, str]:
    return {var: semantic for semantic, var in _parse_decoupled_extraction(step_text)}


def _parse_text_assignments(
    text: str, variables: List[str]
) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    values: Dict[str, Optional[float]] = {v: None for v in variables}
    raw_values: Dict[str, str] = {v: "" for v in variables}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        key_part, val_part = line.split("=", 1)
        key_norm = _normalize_var_key(key_part)
        for var in variables:
            if values[var] is not None:
                continue
            if key_norm != _normalize_var_key(var):
                continue
            parsed = _apply_unit_conversion(val_part)
            if parsed is not None:
                values[var] = parsed
                raw_values[var] = val_part.strip()
    return values, raw_values


def _ocr_looks_table_like(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False
    delimited = sum(1 for line in lines if "|" in line or "\t" in line)
    numeric_dense = sum(
        1 for line in lines[:40] if len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", line)) >= 3
    )
    year_dense = sum(
        1 for line in lines[:40] if len(re.findall(r"\b(?:19|20)\d{2}\b", line)) >= 2
    )
    return delimited >= 2 or numeric_dense >= 3 or year_dense >= 2


def _count_table_data_rows(ocr_text: str) -> int:
    rows = 0
    for line in (ocr_text or "").splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped or "---" in stripped:
            continue
        rows += 1
    return max(0, rows - 1)


def _extract_markdown_table_rows(text: str) -> List[List[str]]:
    table_rows: List[List[str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "|" not in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if any(c for c in cells if c):
            table_rows.append(cells)
    return table_rows


def _build_targeted_table_excerpt(
    ocr_text: str,
    var_name: str,
    semantic_query: str,
    row_keyword: str,
    col_keyword: str,
    max_rows: int = 8,
) -> str:
    """Build a small OCR excerpt around likely rows/headers for one variable."""
    table_rows = _extract_markdown_table_rows(ocr_text)
    if not table_rows:
        return (ocr_text or "")[:1800]

    headers, data_start = _build_markdown_table_headers(table_rows)
    data_rows = table_rows[data_start:]
    header_rows = table_rows[:data_start] if data_start > 0 else table_rows[:1]

    query = semantic_query or var_name
    scored: List[Tuple[int, List[str], str]] = []
    current_section = ""
    for cells in data_rows:
        if not cells:
            continue
        row_label = cells[0].strip()
        if _is_section_header_row(cells):
            current_section = row_label
            continue
        score = max(
            _row_match_score(row_label, row_keyword),
            _row_match_score(row_label, query),
        )
        if current_section:
            score = max(score, _row_match_score(current_section, query))
        if score > 0:
            scored.append((score, cells, current_section))

    selected_lines: List[str] = []
    for header in header_rows[:2]:
        selected_lines.append("| " + " | ".join(header) + " |")

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        for _, cells, section in scored[:max_rows]:
            if section and section not in seen:
                selected_lines.append("| " + " | ".join([section] + [""] * (len(cells) - 1)) + " |")
                seen.add(section)
            row_sig = "|".join(cells)
            if row_sig not in seen:
                selected_lines.append("| " + " | ".join(cells) + " |")
                seen.add(row_sig)
    else:
        for cells in data_rows[:max_rows]:
            selected_lines.append("| " + " | ".join(cells) + " |")

    header_hint = ", ".join(h for h in headers[1:] if h.strip())[:300]
    return (
        f"Variable: {var_name}\n"
        f"Semantic query: {semantic_query or var_name}\n"
        f"Current row hint: {row_keyword or '(empty)'}\n"
        f"Current col hint: {col_keyword or '(empty)'}\n"
        f"Candidate headers: {header_hint}\n"
        "Local table excerpt:\n"
        + "\n".join(selected_lines[: max_rows + 3])
    )


def _parse_anchor_hint_response(
    text: str, variables: List[str]
) -> Dict[str, Dict[str, str]]:
    """Parse JSON-like model output for semantic anchor hints."""
    raw = (text or "").strip()
    if not raw:
        return {}

    candidates = [raw]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    brace = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(1).strip())

    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except Exception:
            continue

    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("anchors"), dict):
        payload = payload["anchors"]
    elif isinstance(payload.get("hints"), dict):
        payload = payload["hints"]

    parsed: Dict[str, Dict[str, str]] = {}
    for var in variables:
        entry = payload.get(var)
        if not isinstance(entry, dict):
            continue
        parsed[var] = {
            "row_hint": str(entry.get("row_hint") or entry.get("row") or "").strip(),
            "col_hint": str(entry.get("col_hint") or entry.get("col") or "").strip(),
            "section_hint": str(entry.get("section_hint") or entry.get("section") or "").strip(),
            "reason": str(entry.get("reason") or "").strip(),
        }
    return parsed


def _collect_row_candidates(
    ocr_text: str, row_keyword: str
) -> List[Tuple[int, str]]:
    """Collect all row labels matching *row_keyword* with their scores.

    Returns list of (score, row_label) sorted descending by score.
    """
    table_rows = _extract_markdown_table_rows(ocr_text)
    if not table_rows:
        return []

    _, data_start = _build_markdown_table_headers(table_rows)
    data_rows = table_rows[data_start:]
    candidates: List[Tuple[int, str]] = []
    section_header = ""
    for cells in data_rows:
        if not cells:
            continue
        row_label = cells[0].strip()
        if _is_section_header_row(cells):
            section_header = row_label
            continue
        score = _row_match_score(row_label, row_keyword)
        if not score and section_header:
            section_norm = _normalize_text(section_header)
            kw_norm = _normalize_text(row_keyword)
            section_tokens = set(re.split(r"[\s_/,\-]+", section_norm))
            remaining = [
                t
                for t in re.split(r"[\s_/,\-]+", kw_norm)
                if t not in section_tokens and len(t) >= 2
            ]
            if remaining:
                score = _row_match_score(row_label, " ".join(remaining))
        if score:
            candidates.append((score, row_label))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def _detect_column_ambiguity(
    col_keyword: str, ocr_text: str,
) -> bool:
    """Check if col_keyword is a bare year and multiple columns contain it."""
    if not col_keyword or not ocr_text:
        return False
    col_kw = _normalize_text(col_keyword)
    if not re.fullmatch(r"(?:19|20)\d{2}[a-z]?", col_kw):
        return False
    table_rows = _extract_markdown_table_rows(ocr_text)
    if not table_rows:
        return False
    headers, _ = _build_markdown_table_headers(table_rows)
    year_hits = sum(
        1 for h in headers
        if h.strip() and re.search(rf"\b{re.escape(col_kw)}\b", _normalize_text(h))
    )
    return year_hits >= 2


def _detect_anchor_assist_reasons(
    var_name: str,
    semantic_query: str,
    row_keyword: str,
    col_keyword: str,
    extracted: Dict[str, float],
    matched_rows: Dict[str, str],
    provenance: Dict[str, str],
    ocr_text: str = "",
    matched_headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    reasons: List[str] = []
    if var_name not in extracted:
        reasons.append("no_structural_match")
    elif provenance.get(var_name, "").startswith("ocr_structure"):
        warning = _validate_row_match_quality(
            var_name,
            row_keyword,
            matched_rows.get(var_name, ""),
        )
        if warning:
            reasons.append("row_match_warning")

    if _query_implies_explicit_column(semantic_query) and not col_keyword:
        reasons.append("column_semantics_unparsed")

    if ocr_text and row_keyword:
        candidates = _collect_row_candidates(ocr_text, row_keyword)
        if len(candidates) >= 2:
            top_score = candidates[0][0]
            same_score_count = sum(1 for s, _ in candidates if s == top_score)
            if same_score_count >= 2:
                reasons.append("multiple_row_candidates")

    if ocr_text and _detect_column_ambiguity(col_keyword, ocr_text):
        mh = (matched_headers or {}).get(var_name, "")
        var_tokens = set(
            t for t in re.split(r"[\s_]+", var_name.lower())
            if len(t) >= 3 and not re.fullmatch(r"(?:19|20)\d{2}[a-z]?", t)
        )
        if mh and var_tokens:
            mh_norm = _normalize_text(mh)
            overlap = sum(1 for t in var_tokens if t in mh_norm)
            if overlap == 0:
                reasons.append("column_ambiguity")
    return reasons


def _detect_source_entity(
    ocr_text: str,
    extracted: Dict[str, float],
    raw_values: Dict[str, str],
) -> Optional[str]:
    """Identify which table row/entity the extracted values most likely came from."""
    if not extracted or not raw_values:
        return None
    lines = ocr_text.splitlines()
    table_rows: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or "|" not in stripped or "---" in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if any(c for c in cells if c):
            table_rows.append(cells)
    if len(table_rows) < 2:
        return None
    best_row, best_hits = -1, 0
    for i, cells in enumerate(table_rows[1:], 1):
        row_text = "|".join(cells)
        hits = sum(1 for r in raw_values.values() if r.strip() and r.strip() in row_text)
        if hits > best_hits:
            best_hits = hits
            best_row = i
    if best_row < 1 or best_hits == 0:
        return None
    for cell in table_rows[best_row][:4]:
        cell_s = cell.strip()
        if cell_s and not re.fullmatch(r"[-+]?\d[\d,.%]*", cell_s):
            return cell_s
    return None


class FinanceExtractionSkill(BaseTool):
    name: str = "finance_extraction_skill"
    description: str = (
        "High-confidence structured extractor for OCR-friendly rectangular tables with "
        "unique row/column intersections and table-like footnotes. Avoid repeated labels, "
        "ambiguous totals, date-like columns, and visually grouped rows. Not a general "
        "chart-reading tool."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of variable names to extract, e.g. ['revenue_2023', 'cost_2023']. The image is provided automatically; do NOT pass image data.",
            },
        },
        "required": ["variables"],
    }

    def __init__(self, **data):
        super().__init__(**data)
        self._llm = None

    def _get_text_llm(self) -> Optional[LLM]:
        if self._llm is not None:
            return self._llm
        try:
            self._llm = LLM(config_name="vision")
        except Exception as e:
            logger.debug(f"[finance_extraction_skill] No vision LLM: {e}")
            self._llm = None
        return self._llm

    async def _extract_missing_with_text_llm(
        self,
        ocr_text: str,
        missing_vars: List[str],
        semantic_queries: Dict[str, str],
        effective_question: str = "",
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
        llm = self._get_text_llm()
        if llm is None or not missing_vars:
            return ({v: None for v in missing_vars}, {v: "" for v in missing_vars})

        targets = []
        for var in missing_vars:
            query = semantic_queries.get(var, var)
            targets.append(f"- {var}: {query}")

        question_hint = ""
        if effective_question:
            question_hint = (
                f"\n- The user's question is: {effective_question}"
                "\n- Use this to identify the RELEVANT entity/company/row. "
                "ALL values MUST come from that SAME entity's row."
            )

        prompt = f"""You are extracting values only from OCR text of a financial figure.

Return only exact variable assignments, one per line:
variable_name = value

Rules:
- Use ONLY values explicitly grounded in the OCR text below.
- Prefer exact row/column intersections or exact table-like footnote values.
- If a variable cannot be grounded confidently, omit it.
- Do NOT compute derived values.
- SIGN CONVENTION: In financial tables, values in parentheses like (12,776) or ($5.2) represent NEGATIVE numbers. You MUST output them with a minus sign: variable = -12776, variable = -5.2.
- ENTITY CONSISTENCY: When the table has multiple entities/companies/rows, ALL extracted values MUST come from the SAME entity/row. Never mix values from different rows.{question_hint}

Targets:
{chr(10).join(targets)}

OCR TEXT:
---
{ocr_text[:12000]}
---"""

        try:
            response = await llm.ask(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            llm_text = (response or "").strip()
            logger.info(
                f"[finance_extraction_skill] text-only OCR extraction preview: {llm_text[:300]}..."
            )
            return _parse_text_assignments(llm_text, missing_vars)
        except Exception as e:
            logger.warning(f"[finance_extraction_skill] OCR text extraction failed: {e}")
            return ({v: None for v in missing_vars}, {v: "" for v in missing_vars})

    async def _resolve_anchor_hints_with_text_llm(
        self,
        ocr_text: str,
        assist_reasons: Dict[str, List[str]],
        semantic_queries: Dict[str, str],
        row_keywords: Dict[str, str],
        col_keywords: Dict[str, str],
        effective_question: str = "",
    ) -> Dict[str, Dict[str, str]]:
        llm = self._get_text_llm()
        if llm is None or not assist_reasons:
            return {}

        targets: List[str] = []
        for var, reasons in assist_reasons.items():
            query = semantic_queries.get(var, var)
            excerpt = _build_targeted_table_excerpt(
                ocr_text,
                var,
                query,
                row_keywords.get(var, ""),
                col_keywords.get(var, ""),
            )
            targets.append(
                f"## {var}\n"
                f"- reasons: {', '.join(reasons)}\n"
                f"{excerpt}"
            )

        question_hint = ""
        if effective_question:
            question_hint = f"\nUser question: {effective_question}"

        prompt = f"""You help a deterministic finance table extraction pipeline resolve ambiguous row/column anchors.

Your job is NOT to return values. Your job is ONLY to identify the EXACT row label and column header from the OCR table that corresponds to each target variable.

Return ONLY JSON with this shape:
{{
  "variable_name": {{
    "row_hint": "exact row label from OCR text",
    "col_hint": "exact column/header from OCR text",
    "section_hint": "section/category header if relevant, or empty",
    "reason": "very short explanation"
  }}
}}

Critical rules:
- Use ONLY labels/phrases that appear VERBATIM in the OCR text.
- Do NOT compute or derive values.
- When multiple rows have similar names (e.g. "Basic earnings per share" as a shares count row vs "Basic earnings per share (cents per share)" as an EPS row), choose the one that matches the SEMANTIC MEANING of the variable.
- Understand financial concept mappings: EBIT often appears as "Income before income taxes", "Operating income", or "Earnings before interest and taxes". EPS appears as "earnings per share (cents)" not shares count.
- When a variable name contains a metric suffix (like "_eps", "_ratio", "_rate"), match to the row that represents that metric's VALUE, not a related label.
- If you are not confident, return empty strings.{question_hint}

Targets:
{chr(10).join(targets)}
"""

        try:
            response = await llm.ask(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            llm_text = (response or "").strip()
            logger.info(
                "[finance_extraction_skill] semantic anchor assist preview: %s...",
                llm_text[:300],
            )
            return _parse_anchor_hint_response(llm_text, list(assist_reasons.keys()))
        except Exception as e:
            logger.warning(
                "[finance_extraction_skill] semantic anchor assist failed: %s",
                e,
            )
            return {}

    async def _understand_extraction_intent(
        self,
        ocr_text: str,
        variables: List[str],
        semantic_queries: Dict[str, str],
        effective_question: str = "",
    ) -> Dict[str, Tuple[str, str]]:
        """FIRST LAYER: Semantic understanding of extraction intent.

        Called for EVERY skill invocation, for ALL variables, as the PRIMARY
        source of (row_hint, col_hint). The model sees the actual table
        headers and row labels, and maps each variable to its exact cell
        location using financial domain knowledge.

        The output feeds directly into the deterministic structural matching
        layer as seeds — the model does not extract values, only anchors.

        Returns {var: (row_hint, col_hint)}.
        """
        llm = self._get_text_llm()
        if llm is None:
            return {}

        table_rows = _extract_markdown_table_rows(ocr_text)
        if not table_rows:
            return {}

        headers, data_start = _build_markdown_table_headers(table_rows)
        data_rows = table_rows[data_start:]

        row_labels: List[str] = []
        for cells in data_rows[:30]:
            if cells and cells[0].strip() and not _is_section_header_row(cells):
                row_labels.append(cells[0].strip())

        header_str = " | ".join(h for h in headers if h.strip())[:300]
        labels_str = ", ".join(row_labels[:20])
        vars_str = "\n".join(
            f"- {v}: {semantic_queries.get(v, v)}" for v in variables
        )
        q_hint = f"\nQ: {effective_question}" if effective_question else ""

        prompt = f"""Map variables to table cells.
Columns: {header_str}
Rows: {labels_str}
Variables:
{vars_str}{q_hint}

Return JSON: {{"var_name": {{"row": "exact row label", "col": "exact column header"}}}}
Rules: row/col MUST be verbatim from above lists. Understand financial concepts: EBIT=Income before taxes/Operating income/息税前利润; EPS=earnings per share/每股收益; Net profit=Net income/净利润/归母净利润; Revenue=Net sales/营业收入. If unsure, empty string."""

        try:
            response = await llm.ask(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            raw = (response or "").strip()
            logger.info(
                "[finance_extraction_skill] intent understanding: %s...",
                raw[:200],
            )
            parsed = _parse_anchor_hint_response(raw, variables)
            result: Dict[str, Tuple[str, str]] = {}
            for var, hints in parsed.items():
                row_h = hints.get("row_hint", "")
                col_h = hints.get("col_hint", "")
                if row_h or col_h:
                    result[var] = (row_h, col_h)
            return result
        except Exception as e:
            logger.warning(
                "[finance_extraction_skill] intent understanding failed: %s", e
            )
            return {}

    async def execute(
        self,
        variables: Optional[List[str]] = None,
        base64_image: Optional[str] = None,
        use_context_image: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        agent_vars = variables or kwargs.get("variables") or []
        base64_image = base64_image or kwargs.get("base64_image")
        use_context_image = kwargs.get("use_context_image", use_context_image)

        if use_context_image and (
            not base64_image or _looks_like_placeholder_base64(base64_image)
        ):
            ctx_images = get_step_images_for_ocr() or []
            if ctx_images:
                if base64_image and _looks_like_placeholder_base64(base64_image):
                    logger.info(
                        "[finance_extraction_skill] replacing placeholder/invalid base64_image with current step image"
                    )
                base64_image = ctx_images[0]

        # Normalize variables: LLM may pass [{"name": "x", "type": "number"}] instead of ["x"]
        normalized: List[str] = []
        for v in agent_vars:
            if isinstance(v, dict):
                normalized.append(str(v.get("name", v.get("variable", ""))))
            else:
                normalized.append(str(v))
        normalized = [v for v in normalized if v]
        variables = list(dict.fromkeys(normalized)) if normalized else []

        if not variables:
            ctx_for_vars = get_current_extraction_context()
            inferred_step = ctx_for_vars.get("step_text", "")
            if inferred_step:
                inferred = _parse_decoupled_extraction(inferred_step)
                variables = [var for _, var in inferred]
                if not variables:
                    variables = _parse_output_as_variables(inferred_step)
                if variables:
                    logger.info(
                        f"[finance_extraction_skill] inferred {len(variables)} variables "
                        f"from step context: {variables}"
                    )

        if not variables:
            return {
                "success": False,
                "observation": (
                    "[ERROR] finance_extraction_skill requires explicit 'variables'. "
                    "Provide a list like variables=[\"revenue_2024\", \"pe_ratio\"]."
                ),
            }
        if not base64_image:
            return {
                "success": False,
                "observation": "[ERROR] No image provided for finance_extraction_skill.",
            }

        step_ctx = get_current_extraction_context()
        step_text = step_ctx.get("step_text", "")
        primary_metric_anchor = step_ctx.get("effective_question", "")
        semantic_queries = _get_semantic_queries(step_text) if step_text else {}

        ocr_tool = OcrExtract()
        ocr_result = await ocr_tool.execute(base64_image=base64_image)
        if not ocr_result.get("success"):
            return {
                "success": False,
                "observation": f"[OCR_ERROR] {ocr_result.get('error', 'OCR failed')}",
            }

        ocr_text = (ocr_result.get("markdown") or ocr_result.get("text") or "").strip()
        if not _ocr_looks_table_like(ocr_text):
            return {
                "success": True,
                "values": {},
                "raw_values": {},
                "provenance": {},
                "ocr_text": ocr_text,
                "observation": (
                    "[STRUCTURE_NOT_CONFIDENT] OCR does not look like a dense structured table. "
                    "Use direct visual reading instead of finance_extraction_skill."
                ),
            }

        data_row_count = _count_table_data_rows(ocr_text)
        extracted: Dict[str, float] = {}
        raw_values: Dict[str, str] = {}
        provenance: Dict[str, str] = {}
        matched_rows: Dict[str, str] = {}
        matched_headers: Dict[str, str] = {}
        row_keywords: Dict[str, str] = {}
        col_keywords: Dict[str, str] = {}

        # ── LAYER 1: Semantic intent understanding (model, primary) ──
        intent = await self._understand_extraction_intent(
            ocr_text, variables, semantic_queries,
            effective_question=primary_metric_anchor,
        )

        # ── LAYER 2: Deterministic structural matching ──
        for var in variables:
            if var in intent and intent[var][0]:
                row_kw, col_kw = intent[var]
                prov_tag = "ocr_structure"
            else:
                query = semantic_queries.get(var, var)
                row_kw, col_kw = _parse_variable_semantics(var, query)
                prov_tag = "ocr_structure"

            row_keywords[var] = row_kw
            col_keywords[var] = col_kw

            num, raw, matched_header, used_explicit_col, matched_row = (
                _parse_markdown_table_for_value(ocr_text, row_kw, col_kw, var_name=var)
            )
            if num is not None:
                if col_kw and (not used_explicit_col or not _is_column_header_consistent(
                    col_kw, matched_header
                )):
                    logger.info(
                        "[finance_extraction_skill] column mismatch for %s: "
                        "expected=%r, matched=%r — skipping",
                        var, col_kw, matched_header,
                    )
                    continue
                extracted[var] = num
                raw_values[var] = raw
                provenance[var] = prov_tag
                matched_rows[var] = matched_row
                matched_headers[var] = matched_header

        missing = [var for var in variables if var not in extracted]
        if missing:
            llm_vals, llm_raw = await self._extract_missing_with_text_llm(
                ocr_text, missing, semantic_queries,
                effective_question=primary_metric_anchor,
            )
            for var in missing:
                if llm_vals.get(var) is not None:
                    extracted[var] = float(llm_vals[var])
                    raw_values[var] = llm_raw.get(var, "")
                    provenance[var] = "ocr_text_llm"

        confidence: Dict[str, str] = {
            var: (
                "high"
                if provenance.get(var) == "ocr_structure"
                else "medium"
            )
            for var in extracted
        }
        risk_flags: Dict[str, List[str]] = {var: [] for var in extracted}

        has_structural_hit = any(
            src.startswith("ocr_structure") for src in provenance.values()
        )
        if extracted and not has_structural_hit and data_row_count >= 2:
            for var in extracted:
                confidence[var] = "medium"

        for var in list(extracted.keys()):
            if provenance.get(var) == "ocr_text_llm" and data_row_count >= 2 and not matched_rows.get(var):
                risk_flags[var].append("row_not_grounded")
            if _query_implies_explicit_column(semantic_queries.get(var, var)) and not col_keywords.get(var):
                risk_flags[var].append("column_semantics_unparsed")

        # --- P1 Variable-level validations ---
        validation_warnings: List[str] = []

        # (a) Duplicate value anomaly detection
        dup_vars = _detect_duplicate_value_anomaly(extracted)
        if dup_vars:
            dup_detail = ", ".join(
                f"{v}={extracted[v]}" for v in dup_vars
            )
            logger.warning(
                "[finance_extraction_skill] DUPLICATE VALUE ANOMALY: %s",
                dup_detail,
            )
            for v in dup_vars:
                confidence[v] = "low"
                risk_flags.setdefault(v, []).append("duplicate_value_anomaly")
            validation_warnings.append(
                f"[DUPLICATE_VALUE_WARNING] Variables with identical values "
                f"(likely row/column mis-alignment): {dup_detail}. "
                f"Verify each variable against the image before trusting."
            )

        # (b) Summary row caution
        for var in list(extracted.keys()):
            row_label = matched_rows.get(var, "")
            if row_label and _is_summary_row_label(row_label):
                logger.info(
                    "[finance_extraction_skill] summary row match for %s: '%s'",
                    var,
                    row_label,
                )
                if confidence.get(var) == "high":
                    confidence[var] = "medium"
                risk_flags.setdefault(var, []).append("summary_row_caution")
                validation_warnings.append(
                    f"[SUMMARY_ROW_CAUTION] '{var}' matched summary row "
                    f"'{row_label}'. Verify this is the intended row, not a "
                    f"subtotal or aggregate that includes unrelated items."
                )

        # (c) Variable-level row match quality
        for var in list(extracted.keys()):
            if not provenance.get(var, "").startswith("ocr_structure"):
                continue
            row_kw = row_keywords.get(var, "")
            row_label = matched_rows.get(var, "")
            warning = _validate_row_match_quality(var, row_kw, row_label)
            if warning:
                logger.warning(
                    "[finance_extraction_skill] row match quality issue for %s: %s",
                    var,
                    warning,
                )
                if confidence.get(var) == "high":
                    confidence[var] = "medium"
                risk_flags.setdefault(var, []).append("row_match_warning")
                validation_warnings.append(
                    f"[ROW_MATCH_WARNING] {var}: {warning}. "
                    f"Verify the extracted value against the image."
                )

        # (d) Semantic type classification for downstream agents
        variable_semantics: Dict[str, str] = {}
        for var in list(extracted.keys()):
            raw = raw_values.get(var, "")
            query = semantic_queries.get(var, var)
            row_kw, col_kw = _parse_variable_semantics(var, query)
            header_ctx = col_kw or ""
            if _has_currency_veto(raw, var, header_ctx):
                variable_semantics[var] = "currency/price"
            elif _count_percent_positive_signals(raw, var, header_ctx) >= 2:
                variable_semantics[var] = "rate/percentage"
            elif _count_percent_positive_signals(raw, var, header_ctx) == 1:
                variable_semantics[var] = "possibly_rate (single signal)"
            else:
                variable_semantics[var] = "general"

        source_entity = _detect_source_entity(ocr_text, extracted, raw_values)

        global_risk_flags: List[str] = []
        provenance_kinds = {src for src in provenance.values() if src}
        if len(extracted) >= 2 and len(provenance_kinds) >= 2:
            global_risk_flags.append("mixed_provenance")
        if len(extracted) >= 2 and any(
            "row_not_grounded" in risk_flags.get(var, []) for var in extracted
        ):
            global_risk_flags.append("mixed_grounding")
        if any(
            "column_semantics_unparsed" in risk_flags.get(var, []) for var in extracted
        ):
            global_risk_flags.append("column_semantics_ambiguity")

        blocking_reasons: List[str] = []
        for var in extracted:
            var_flags = risk_flags.get(var, [])
            var_conf = confidence.get(var, "")
            if var_conf == "low" and "row_match_warning" in var_flags:
                blocking_reasons.append(
                    f"variable '{var}' has low confidence AND row-match ambiguity — not trustworthy"
                )
            elif var_conf == "low" and "row_not_grounded" in var_flags:
                blocking_reasons.append(
                    f"variable '{var}' has low confidence AND ungrounded row — not trustworthy"
                )
        if data_row_count >= 2 and "column_semantics_ambiguity" in global_risk_flags:
            blocking_reasons.append(
                "the request specifies column semantics, but they could not be parsed into a stable column anchor"
            )

        extraction_risk = {
            "global_flags": global_risk_flags,
            "blocking_reasons": blocking_reasons,
            "per_variable": {
                var: {
                    "provenance": provenance.get(var, ""),
                    "confidence": confidence.get(var, ""),
                    "matched_row": matched_rows.get(var, ""),
                    "matched_header": matched_headers.get(var, ""),
                    "row_keyword": row_keywords.get(var, ""),
                    "column_keyword": col_keywords.get(var, ""),
                    "risk_flags": risk_flags.get(var, []),
                }
                for var in extracted
            },
        }

        gate_err = await validate_extraction_against_current_request(
            extracted=extracted,
            extraction_risk=extraction_risk,
        )
        if gate_err:
            return {
                "success": False,
                "values": extracted,
                "raw_values": raw_values,
                "provenance": provenance,
                "confidence": confidence,
                "matched_rows": matched_rows,
                "matched_headers": matched_headers,
                "source_entity": source_entity,
                "extraction_risk": extraction_risk,
                "intent_hints": {v: {"row": r, "col": c} for v, (r, c) in intent.items()},
                "ocr_text": ocr_text,
                "observation": gate_err,
            }

        auto_save_status = ""
        if extracted:
            auto_save_status = await persist_extracted_values_to_shared_python(extracted)
            if variable_semantics:
                register_variable_semantics(
                    {
                        var: {
                            "semantic_type": variable_semantics[var],
                            "provenance": provenance.get(var, ""),
                            "confidence": confidence.get(var, ""),
                            "matched_row": matched_rows.get(var, ""),
                            "matched_header": matched_headers.get(var, ""),
                            "row_keyword": row_keywords.get(var, ""),
                            "column_keyword": col_keywords.get(var, ""),
                            "source_entity": source_entity or "",
                            "risk_flags": risk_flags.get(var, []),
                        }
                        for var in variable_semantics
                    }
                )

        missing_after = [var for var in variables if var not in extracted]
        observation_lines = []
        if extracted:
            high_conf = [f"{k}={v}" for k, v in extracted.items() if confidence.get(k) == "high"]
            med_conf = [f"{k}={v}" for k, v in extracted.items() if confidence.get(k) == "medium"]
            low_conf = [f"{k}={v}" for k, v in extracted.items() if confidence.get(k) == "low"]
            if high_conf:
                observation_lines.append(
                    "Extracted (structural match, high confidence): " + ", ".join(high_conf)
                )
            if med_conf:
                observation_lines.append(
                    "Extracted (medium confidence — verify against image): "
                    + ", ".join(med_conf)
                )
            if low_conf:
                observation_lines.append(
                    "Extracted (LOW confidence — likely extraction error, "
                    "MUST verify against image or re-extract with direct read): "
                    + ", ".join(low_conf)
                )
        if source_entity:
            observation_lines.append(f"Source entity/row: {source_entity}")
        semantic_notes = [
            f"{k}: {v}" for k, v in variable_semantics.items() if v != "general"
        ]
        if semantic_notes:
            observation_lines.append(
                "Variable semantics: " + "; ".join(semantic_notes)
            )
        for w in validation_warnings:
            observation_lines.append(w)
        if global_risk_flags:
            observation_lines.append(
                "Structured risk flags: " + ", ".join(global_risk_flags)
            )
        if missing_after:
            observation_lines.append("Missing: " + ", ".join(missing_after))
        if auto_save_status:
            observation_lines.append(auto_save_status.strip())

        return {
            "success": True,
            "values": extracted,
            "raw_values": raw_values,
            "provenance": provenance,
            "confidence": confidence,
            "matched_rows": matched_rows,
            "matched_headers": matched_headers,
            "source_entity": source_entity,
            "variable_semantics": variable_semantics,
            "extraction_risk": extraction_risk,
            "intent_hints": {v: {"row": r, "col": c} for v, (r, c) in intent.items()},
            "ocr_text": ocr_text,
            "observation": "\n".join(observation_lines).strip(),
        }
