"""Build a ``schema.Trace`` from the ft-only base's solve-time state.

finacumen variant wraps ``ft-only`` (via ``_FTOnlyWithTraceCapture``),
so this adapter accepts only that captured variant name.
"""
from __future__ import annotations

from typing import Any

from finacumen.fm.schema import Trace

FINAL_COT_CHAR_CAP = 3200


def build_trace(base_variant: Any, result: dict) -> Trace:
    """Build a Trace from the captured ft-only base's _last_agent.

    Raises ValueError if the base is not the trace-capture subclass.
    """
    name = getattr(base_variant, "name", "") or ""
    if name != "__ft_only_with_trace__":
        raise ValueError(
            f"trace_adapter only supports the ft-only trace-capture wrapper; "
            f"got {name!r}. finacumen must not wrap any other base."
        )
    voted = str(result.get("final_answer", ""))
    agent = getattr(base_variant, "_last_agent", None)
    final = _final_cot_from_agent_memory(agent, voted)
    return Trace(
        final_cot=final,
        chain_deltas=None,
        final_answer=voted,
        source_variant=name,
    )


# -------- agent-only helpers ------------------------------------------------

def _final_cot_from_agent_memory(agent: Any, voted: str) -> str:
    """Trim agent.memory.messages into a compact trace:
    - last assistant content (the one before terminate or the terminate call text)
    - at most 2 preceding python_execute outputs for evidence
    Fallback to voted if the agent is None or has no usable messages.
    """
    if agent is None or not getattr(agent, "memory", None):
        return voted
    messages = list(getattr(agent.memory, "messages", []) or [])
    last_assistant = ""
    tool_outputs: list[str] = []
    for msg in reversed(messages):
        role = getattr(msg, "role", "") or ""
        content = getattr(msg, "content", "") or ""
        if not last_assistant and role == "assistant" and content:
            last_assistant = str(content)
        if role == "tool" and content:
            tool_outputs.append(str(content))
            if len(tool_outputs) >= 2:
                break
    parts: list[str] = []
    if last_assistant:
        parts.append("[Assistant final reasoning]\n" + last_assistant.strip())
    for i, out in enumerate(reversed(tool_outputs), start=1):
        parts.append(f"[Tool output #{i}]\n" + out.strip())
    text = "\n\n".join(parts) if parts else voted
    return _truncate(text, FINAL_COT_CHAR_CAP)


# -------- generic --------------------------------------------------------

def _truncate(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[: cap - 3] + "..."
