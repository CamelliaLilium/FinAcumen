"""GLM-4.1V-9B-Thinking — AIHubMix OpenAI-compatible gateway.

Critical quirks (observed via empirical testing 2026-04-29):
  - Does NOT support OpenAI function-calling tools API
    (returns 400 'Function call is not supported for this model').
    → supports_function_calling = False; falls back to prompt-based dispatch.
  - **Rejects temperature=0.0** with 400 'parameter invalid' (Thinking models
    require sampling). Use T ≥ 0.3; we recommend 0.6 for stability.
    → min_temperature = 0.3; callers should use the adapter helper to clamp.
  - Output includes <|begin_of_box|>...<|end_of_box|> thinking markers
    (model-specific reasoning trace; the variant should strip them when
    extracting the final_answer).
"""
from __future__ import annotations

import os

from .default import DefaultAdapter


class GLMThinkingAdapter(DefaultAdapter):
    name = "glm-4.1v-thinking"

    def matches(self, model_id: str) -> bool:
        m = model_id.lower()
        return "glm-4.1v" in m or "glm-4.1v-9b-thinking" in m

    @property
    def supports_function_calling(self) -> bool:
        return False  # AIHubMix gateway returns 400 on tools=...

    # GLM Thinking models reject T=0; clamp to >= 0.3 in clamp_temperature.
    min_temperature: float = 0.3
    supports_images: bool = True
    use_max_completion_tokens: bool = False

    def resolve_api_key(self, settings_api_key: str) -> str:
        """Prefer settings.api_key when sk-prefixed; fall back to OPENAI_API_KEY env (AIHubMix uses OpenAI-compatible auth)."""
        cfg = settings_api_key or ""
        if cfg.startswith("sk-"):
            return cfg
        return os.environ.get("OPENAI_API_KEY", "") or cfg

    def prepare_content(self, text: str, images=None):
        """When no image, return plain string (avoid 400 'parameter invalid' on empty multimodal list)."""
        if not images:
            return text  # plain string for text-only queries
        return super().prepare_content(text, images)

    def prepare_tools(self, tools):
        # GLM-4.1V-Thinking via AIHubMix rejects tools=. Always omit; caller
        # should fall back to inject_tool_prompt + parse_tool_calls.
        return None
