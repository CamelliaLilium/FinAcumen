"""GLM-4.6V-Flash — Zhipu AI official endpoint (OpenAI-compatible).

Free model with native multimodal + function calling support.
Endpoint: https://open.bigmodel.cn/api/paas/v4/
Auth: Bearer token (Zhipu ID.secret format, works with OpenAI SDK).
"""
from __future__ import annotations

from .default import DefaultAdapter


class GLMFlashAdapter(DefaultAdapter):
    name = "glm-flash"

    def matches(self, model_id: str) -> bool:
        m = model_id.lower()
        return "glm-4" in m and "flash" in m

    def resolve_api_key(self, settings_api_key: str) -> str:
        cfg = settings_api_key or ""
        if cfg and not cfg.startswith("YOUR_") and not cfg.endswith("_FROM_ENV"):
            return cfg
        return cfg
