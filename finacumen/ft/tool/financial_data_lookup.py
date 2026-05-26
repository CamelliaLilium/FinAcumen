"""Offline lookup over the FinTMMBench provided knowledge base.

FinTMMBench (arXiv 2503.05185) ships three JSON-formatted offline data sources
under ``datasets/FinTMMBench/data/``:

- ``StockPrice.json`` (124,130 rows) — daily open/close/high/low/volume per
  (Symbol, Date) for ~50 US tickers, year 2022 (calendar inferred from sample).
- ``News.json`` (3,143 rows) — news article text tagged with Company, Date,
  Symbol, type. Used for sentiment/event questions.
- ``FinancialTable.json`` (35,038 rows) — quarterly financial indicators
  (grossProfit, revenue, margins, etc.) per (Symbol, Date, indicator_name).

Most fintmm test/train questions reference one of these three sources.
v0/v1/v2 ``agent-only`` had no way to access them and answered ~0-10% on
fintmm. This tool exposes the three sources to the agent as a single tool
with a ``source`` enum parameter; the agent picks the source and filters by
symbol / date / indicator to retrieve rows.

Design choices:
- Lazy load: the 3 JSON files (~85 MB combined) are loaded on the first
  call, then cached at module level. A pilot with 50 fintmm questions hits
  the tool ~50-150 times so amortization is fine.
- Indexed by Symbol: filtering by Symbol is the dominant query pattern;
  we build a Symbol -> [row_idx] index once.
- Date matching is tolerant: accepts YYYY-MM-DD, single-day queries against
  StockPrice (Interval=Date) and FinancialTable (Interval=Quarter); also
  accepts a "YYYY-MM-DD..YYYY-MM-DD" range form.
- Output is a JSON-serialized matches list, capped at top-N rows so the
  agent's context budget stays under control. Empty matches return a
  helpful ``hint`` listing available symbols (top 10 by row count) so the
  agent can recover from a typo'd ticker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from finacumen.ft.tool.base import BaseTool, ToolResult

# Resolve datasets dir relative to this file's location within the repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "datasets" / "train" / "FinTMMBench" / "data"

_DEFAULT_TOP_N = 10
_HINT_TOP_SYMBOLS = 10

# Module-level caches; populated on first access.
_STOCK_PRICE: Optional[list[dict]] = None
_NEWS: Optional[list[dict]] = None
_FINANCIAL_TABLE: Optional[list[dict]] = None
_INDEX_BY_SOURCE: dict[str, dict[str, list[int]]] = {}
_TOP_SYMBOLS_CACHE: dict[str, list[str]] = {}


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"FinTMMBench data file missing: {path}. The tool requires "
            f"FinTMMBench's offline data under datasets/FinTMMBench/data/."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_loaded(source: str) -> list[dict]:
    """Lazy-load the source JSON, build the Symbol index on first call."""
    global _STOCK_PRICE, _NEWS, _FINANCIAL_TABLE
    if source == "stock_price":
        if _STOCK_PRICE is None:
            _STOCK_PRICE = _load_json(_DATA_DIR / "StockPrice.json")
            _build_index("stock_price", _STOCK_PRICE)
        return _STOCK_PRICE
    if source == "news":
        if _NEWS is None:
            _NEWS = _load_json(_DATA_DIR / "News.json")
            _build_index("news", _NEWS)
        return _NEWS
    if source == "financial_table":
        if _FINANCIAL_TABLE is None:
            _FINANCIAL_TABLE = _load_json(_DATA_DIR / "FinancialTable.json")
            _build_index("financial_table", _FINANCIAL_TABLE)
        return _FINANCIAL_TABLE
    raise ValueError(
        f"Unknown source {source!r}. Must be one of: stock_price, news, financial_table."
    )


def _build_index(source: str, rows: list[dict]) -> None:
    by_symbol: dict[str, list[int]] = {}
    sym_count: dict[str, int] = {}
    for i, row in enumerate(rows):
        sym = row.get("Symbol")
        # News.Symbol is a Python list (json-decoded as list); StockPrice /
        # FinancialTable.Symbol is a plain string ticker. Some old dumps may
        # have a stringified list "['AMD']" — handle both.
        symbols: list[str] = []
        if isinstance(sym, list):
            symbols = [str(s).strip().upper() for s in sym if str(s).strip()]
        elif isinstance(sym, str) and sym.startswith("[") and sym.endswith("]"):
            try:
                parsed = json.loads(sym.replace("'", '"'))
                if isinstance(parsed, list):
                    symbols = [str(s).strip().upper() for s in parsed if str(s).strip()]
            except json.JSONDecodeError:
                pass
        elif isinstance(sym, str) and sym.strip():
            symbols = [sym.strip().upper()]
        for s in symbols:
            by_symbol.setdefault(s, []).append(i)
            sym_count[s] = sym_count.get(s, 0) + 1
    _INDEX_BY_SOURCE[source] = by_symbol
    # Top-N symbols by row count for "hint" recovery.
    _TOP_SYMBOLS_CACHE[source] = [
        s for s, _ in sorted(sym_count.items(), key=lambda kv: -kv[1])[:_HINT_TOP_SYMBOLS]
    ]


_QUARTER_END = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}


def _normalize_quarter(s: str) -> str:
    """Map quarter-style date hints to FinancialTable's YYYY-MM-DD form.

    FinancialTable.json uses calendar-quarter-end dates only — Q1 → 03-31,
    Q2 → 06-30, etc. Models routinely pass ``2022-Q2`` / ``Q2 2022`` /
    ``2022Q2``; without normalization the date filter never matches and
    the model gives up. This helper covers the common shapes; fall through
    leaves the string unchanged so YYYY-MM-DD inputs still work.
    """
    import re

    s_norm = s.strip().upper().replace(" ", "")
    # 2022-Q2 / 2022Q2
    m = re.fullmatch(r"(\d{4})[-]?(Q[1-4])", s_norm)
    if m:
        return f"{m.group(1)}-{_QUARTER_END[m.group(2)]}"
    # Q22022 / Q2-2022
    m = re.fullmatch(r"(Q[1-4])[-]?(\d{4})", s_norm)
    if m:
        return f"{m.group(2)}-{_QUARTER_END[m.group(1)]}"
    return s


def _parse_date_filter(raw: str) -> tuple[str, Optional[str]]:
    """Normalize date input.

    Accepts:
      - 'YYYY-MM-DD' or 'YYYY-MM-DD..YYYY-MM-DD'
      - 'YYYY-Q1' / '2022Q1' / 'Q1 2022' (auto-mapped to quarter-end YYYY-MM-DD)
      - free-text dates parsed via dateutil if available

    Returns ``(start, end_or_none)`` both as YYYY-MM-DD strings (or quarter
    string for FinancialTable). end_or_none is None for single-day queries.
    """
    s = (raw or "").strip()
    if ".." in s:
        a, b = s.split("..", 1)
        return _normalize_quarter(a.strip()), _normalize_quarter(b.strip())
    return _normalize_quarter(s), None


def _map_to_quarter_end(date_str: str) -> str | None:
    """Map a YYYY-MM-DD date to the most recent quarter-end date.
    
    Returns None if the string doesn't match YYYY-MM-DD format.
    """
    try:
        from datetime import date as dt
        y, m, d = map(int, date_str.strip().split("-"))
        dt(y, m, d)  # validate
    except (ValueError, OverflowError):
        return None
    if m <= 3:
        return f"{y}-03-31"
    elif m <= 6:
        return f"{y}-06-30"
    elif m <= 9:
        return f"{y}-09-30"
    else:
        return f"{y}-12-31"


def _row_matches_date(row: dict, start: str, end: Optional[str]) -> bool:
    row_date = str(row.get("Date") or "").strip()
    if not row_date:
        return False
    if end is None:
        if row_date == start or row_date.startswith(start):
            return True
        # Quarter-end fallback: try mapping start to nearest quarter-end
        q_start = _map_to_quarter_end(start)
        if q_start and q_start != start:
            if row_date == q_start or row_date.startswith(q_start):
                return True
        return False
    return start <= row_date <= end


class FinancialDataLookup(BaseTool):
    name: str = "financial_data_lookup"
    description: str = (
        "Query offline FinTMMBench knowledge base. Use when the question "
        "asks about a company's stock price on a specific date "
        "(StockPrice), news / sentiment / events for a company over a "
        "period (News), or a quarterly financial indicator like revenue / "
        "gross profit / margin (FinancialTable). Returns a JSON list of "
        "matching rows. The agent should call python_execute afterward to "
        "compute on the returned values. Do NOT use for questions that "
        "have all data inline in context (table extraction, chart reading)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["stock_price", "news", "financial_table"],
                "description": (
                    "Which offline source. stock_price = daily OHLCV; news "
                    "= article text tagged by company; financial_table = "
                    "quarterly indicators (revenue, profit, margin, etc.)."
                ),
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Stock ticker symbol (e.g. CMCSA, MNST, AAPL). "
                    "Case-insensitive. If unknown, leave blank to scan all."
                ),
            },
            "date": {
                "type": "string",
                "description": (
                    "Single date 'YYYY-MM-DD' or range "
                    "'YYYY-MM-DD..YYYY-MM-DD'. Quarterly hints "
                    "'YYYY-Q[1-4]' / 'Q[1-4] YYYY' are auto-mapped to "
                    "calendar-quarter-end (e.g. 2022-Q2 → 2022-06-30, the "
                    "form FinancialTable uses). Optional."
                ),
            },
            "indicator": {
                "type": "string",
                "description": (
                    "For stock_price: one of openPrice / closePrice / "
                    "highPrice / lowPrice / volume. For financial_table: "
                    "indicator name like grossProfit / revenue / netIncome. "
                    "Optional — if omitted, returns all indicators for the "
                    "matching (symbol, date)."
                ),
            },
            "top_n": {
                "type": "integer",
                "description": (
                    f"Cap rows returned. Default {_DEFAULT_TOP_N}; raise "
                    f"to 50 for date-range queries."
                ),
            },
        },
        "required": ["source"],
    }

    async def execute(
        self,
        source: str,
        symbol: str = "",
        date: str = "",
        indicator: str = "",
        top_n: int = _DEFAULT_TOP_N,
        **_,
    ) -> ToolResult:
        """Filter the chosen source by symbol / date / indicator, return matches."""
        try:
            rows = _ensure_loaded(source)
        except (ValueError, FileNotFoundError) as e:
            return ToolResult(error=str(e))

        sym_norm = (symbol or "").strip().upper()
        index = _INDEX_BY_SOURCE.get(source, {})

        # Restrict candidate row indices via Symbol index when symbol given.
        if sym_norm:
            candidate_idx = index.get(sym_norm, [])
            if not candidate_idx:
                hint = (
                    f"No rows for symbol {sym_norm!r} in {source}. "
                    f"Top symbols available in {source}: "
                    f"{', '.join(_TOP_SYMBOLS_CACHE.get(source, [])[:_HINT_TOP_SYMBOLS])}"
                )
                return ToolResult(
                    output=json.dumps(
                        {"matches": [], "hint": hint, "n_total_rows": len(rows)},
                        ensure_ascii=False,
                    )
                )
            candidates = (rows[i] for i in candidate_idx)
        else:
            candidates = iter(rows)

        # Apply date filter.
        if date:
            try:
                start, end = _parse_date_filter(date)
            except Exception as e:
                return ToolResult(error=f"date parse failed: {e}")
            candidates = (r for r in candidates if _row_matches_date(r, start, end))

        # Apply indicator filter (stock_price + financial_table only).
        ind_norm = (indicator or "").strip()
        if ind_norm and source in ("stock_price", "financial_table"):
            candidates = (
                r for r in candidates
                if str(r.get("indicator_name", "")).strip() == ind_norm
            )

        # Materialize + cap.
        matches: list[dict] = []
        cap = max(1, int(top_n or _DEFAULT_TOP_N))
        for r in candidates:
            matches.append(r)
            if len(matches) >= cap:
                break

        if not matches:
            hint_parts = [f"No rows matched in {source}"]
            if sym_norm:
                hint_parts.append(f"symbol={sym_norm}")
            if date:
                hint_parts.append(f"date={date}")
            if ind_norm:
                hint_parts.append(f"indicator={ind_norm}")
            return ToolResult(
                output=json.dumps(
                    {
                        "matches": [],
                        "hint": "; ".join(hint_parts) + ". Check spelling, date format YYYY-MM-DD, or relax filters.",
                        "n_total_rows": len(rows),
                    },
                    ensure_ascii=False,
                )
            )

        return ToolResult(
            output=json.dumps(
                {
                    "matches": matches,
                    "n_returned": len(matches),
                    "n_total_rows": len(rows),
                    "capped": len(matches) >= cap,
                },
                ensure_ascii=False,
            )
        )
