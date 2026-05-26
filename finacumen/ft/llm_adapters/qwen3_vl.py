"""Qwen3-VL-8B-Instruct — DashScope OpenAI-compatible endpoint.

Quirks vs DefaultAdapter:
  - Allows QWEN_API_KEY env override of placeholder config (legacy)
  - Otherwise pure OpenAI behavior (multimodal list + tools)
"""
from __future__ import annotations

import os

from .default import DefaultAdapter


class Qwen3VLAdapter(DefaultAdapter):
    name = "qwen3-vl"
    # DashScope-specific knobs: disable thinking, set generous budget.
    # GLM and most other gateways reject these — only Qwen3 series accepts them.
    extra_body = {"enable_thinking": False, "thinking_budget": 81920}

    def matches(self, model_id: str) -> bool:
        m = model_id.lower()
        return "qwen3-vl" in m or "qwen3-vl-8b" in m

    def resolve_api_key(self, settings_api_key: str) -> str:
        """Prefer settings.api_key when sk-prefixed; fall back to QWEN_API_KEY env."""
        cfg = settings_api_key or ""
        if cfg.startswith("sk-"):
            return cfg  # config wins for real keys (avoids stale env override)
        return os.environ.get("QWEN_API_KEY", "") or cfg
