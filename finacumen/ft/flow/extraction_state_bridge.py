import contextvars
import inspect
from typing import Any, Dict, Optional

from finacumen.ft.logger import logger
from finacumen.ft.tool.ocr import get_step_context

_shared_python_execute_ctx: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "shared_python_execute", default=None
)

_variable_semantic_registry: contextvars.ContextVar[Dict[str, Dict[str, Any]]] = contextvars.ContextVar(
    "variable_semantic_registry", default={}
)


def set_shared_python_execute(instance: Optional[Any]) -> None:
    _shared_python_execute_ctx.set(instance)


def get_shared_python_execute() -> Optional[Any]:
    return _shared_python_execute_ctx.get()


def register_variable_semantics(semantics: Dict[str, Any]) -> None:
    """Register semantic metadata for extracted variables.

    Args:
        semantics: mapping of variable_name -> semantic_type or metadata dict.
                   The metadata dict may include keys like semantic_type,
                   provenance, confidence, matched_row, matched_header, etc.
    """
    if not semantics:
        return
    registry = _variable_semantic_registry.get()
    if not registry:
        registry = {}
    for name, payload in semantics.items():
        existing = dict(registry.get(name) or {})
        if isinstance(payload, dict):
            incoming = dict(payload)
            semantic_type = incoming.get("semantic_type")
            if semantic_type is not None:
                incoming["semantic_type"] = str(semantic_type)
            existing.update(incoming)
            registry[name] = existing
        else:
            existing["semantic_type"] = str(payload)
            registry[name] = existing
    _variable_semantic_registry.set(registry)
    logger.info(
        f"[extraction_state_bridge] registered variable semantics: "
        f"{[(name, (payload.get('semantic_type') if isinstance(payload, dict) else payload)) for name, payload in semantics.items()]}"
    )


def get_variable_semantics() -> Dict[str, Dict[str, Any]]:
    """Retrieve all registered variable semantic metadata."""
    return _variable_semantic_registry.get() or {}


def get_variable_semantic_type(var_name: str) -> Optional[str]:
    """Get the semantic type of a specific variable, or None if unregistered."""
    registry = _variable_semantic_registry.get() or {}
    entry = registry.get(var_name)
    return entry.get("semantic_type") if entry else None


def reset_variable_semantic_registry() -> None:
    """Clear all registered variable semantics (call at flow start)."""
    _variable_semantic_registry.set({})


def get_current_extraction_context() -> Dict[str, str]:
    step_ctx = get_step_context() or {}
    return {
        "step_text": (step_ctx.get("step_text") or "").strip(),
        "user_request": (step_ctx.get("user_request") or "").strip(),
        "effective_question": (step_ctx.get("effective_question") or "").strip(),
    }


async def validate_extraction_against_current_request(**kwargs: Any) -> Optional[str]:
    """Validation gate — reject only when high-risk flags are present.

    The first-layer semantic understanding model already handles variable-to-row
    mapping. This gate only rejects on clear structural risk signals
    (blocking_reasons), not on token-level source-entity mismatches which
    are unreliable for cross-language / financial-synonym scenarios.
    """
    extracted: Dict = kwargs.get("extracted") or {}
    if not extracted:
        return None

    extraction_risk: Dict[str, Any] = kwargs.get("extraction_risk") or {}
    blocking_reasons = extraction_risk.get("blocking_reasons") or []
    if blocking_reasons:
        reasons_text = "; ".join(str(r).strip() for r in blocking_reasons if str(r).strip())
        if reasons_text:
            return (
                "[VALIDATION_GATE] Structured extraction risk requires direct visual verification. "
                f"Reasons: {reasons_text}. Reject this extraction and fall back to direct visual reading."
            )

    return None


async def persist_extracted_values_to_shared_python(extracted: Dict[str, float]) -> str:
    python_executor = get_shared_python_execute()
    if not python_executor or not extracted:
        return (
            "\n[SYSTEM WARNING] No shared Python executor found. "
            "You MUST call python_execute manually to store these variables."
        )

    try:
        assign_code = "\n".join(f"{k} = {v}" for k, v in extracted.items())
        if hasattr(python_executor, "run"):
            if inspect.iscoroutinefunction(python_executor.run):
                await python_executor.run(assign_code)
            else:
                python_executor.run(assign_code)
        elif hasattr(python_executor, "execute"):
            exec_fn = python_executor.execute
            if inspect.iscoroutinefunction(exec_fn):
                await exec_fn(code=assign_code)
            else:
                exec_fn(code=assign_code)
        logger.info(
            f"[extraction_state_bridge] saved extracted variables to shared python: {list(extracted.keys())}"
        )
        return (
            "\n[SYSTEM ACTION] Extracted variables have been automatically saved to the Python environment."
        )
    except Exception as e:
        logger.error(f"[extraction_state_bridge] failed to save variables: {e}")
        return (
            f"\n[SYSTEM WARNING] Auto-save failed ({e}). "
            f"You MUST call python_execute manually to store: {', '.join(extracted.keys())}"
        )
