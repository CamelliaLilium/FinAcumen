"""LLM adapter registry — pick the right adapter for a given model id.

Adapters encapsulate per-model quirks (function-call support, multimodal
content shape, api_key resolution) so variants/agents stay model-agnostic.
"""
from __future__ import annotations

from .base import LLMAdapter
from .default import DefaultAdapter
from .qwen3_vl import Qwen3VLAdapter
from .glm_flash import GLMFlashAdapter
from .glm_thinking import GLMThinkingAdapter

# Order matters: specific adapters first, DefaultAdapter last (catch-all).
ADAPTERS: list[LLMAdapter] = [
    Qwen3VLAdapter(),
    GLMFlashAdapter(),
    GLMThinkingAdapter(),
    DefaultAdapter(),
]


def get_adapter(model_id: str) -> LLMAdapter:
    """Return the first adapter whose `matches()` accepts the model id.

    `DefaultAdapter` is the catch-all (always matches), so this never raises.
    """
    model_id = (model_id or "").strip()
    for ad in ADAPTERS:
        if ad.matches(model_id):
            return ad
    return ADAPTERS[-1]
