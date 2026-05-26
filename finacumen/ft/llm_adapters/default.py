"""DefaultAdapter — OpenAI-compatible behavior assumed.

Used as a catch-all for unknown models that speak the OpenAI API. Concrete
adapters subclass this and override only the bits that differ.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional


class DefaultAdapter:
    """OpenAI-compatible default. Function calling + multimodal both supported."""

    name = "default"
    min_temperature: float = 0.0  # most OpenAI-compatible models accept T=0
    # Model-specific OpenAI extra_body params. Default: empty (no extras).
    # Override per-adapter for endpoint-specific knobs.
    extra_body: dict = {}
    supports_images: bool = True
    use_max_completion_tokens: bool = False

    def matches(self, model_id: str) -> bool:
        # Last-resort fallback — registry order ensures we only land here when
        # no other adapter claims the model.
        return True

    @property
    def supports_function_calling(self) -> bool:
        return True

    def clamp_temperature(self, temperature: float) -> float:
        """Clamp T below the model's min; useful guard for callers."""
        return max(temperature, self.min_temperature)

    # ---- api_key resolution ------------------------------------------------

    def resolve_api_key(self, settings_api_key: str) -> str:
        """Prefer settings.api_key when it's a real key; fall back to OPENAI_API_KEY env."""
        cfg = settings_api_key or ""
        # "Real key" heuristic: starts with 'sk-' or any non-placeholder string.
        is_real = cfg.startswith("sk-") or (
            cfg and not cfg.startswith("YOUR_") and not cfg.endswith("_FROM_ENV")
        )
        if is_real:
            return cfg
        return os.environ.get("OPENAI_API_KEY", "") or cfg

    # ---- content + tools ---------------------------------------------------

    def prepare_content(self, text: str, images=None):
        """Always emit OpenAI multimodal list. Pure-text models override to return str."""
        if not images:
            return [{"type": "text", "text": text}]
        content = [{"type": "text", "text": text}]
        for img in images:
            mime = "image/jpeg"
            try:
                import base64
                head = base64.b64decode(img[:8] + "==")[:4]
                if head[:4] == b"\x89PNG":
                    mime = "image/png"
                elif head[:2] == b"\xff\xd8":
                    mime = "image/jpeg"
                elif head[:4] == b"RIFF" or head[:4] == b"GIF8":
                    mime = "image/gif"
            except Exception:
                pass
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img if img.startswith("http") or img.startswith("data:")
                    else f"data:{mime};base64,{img}",
                    "detail": "auto",
                },
            })
        return content

    def prepare_tools(self, tools: Optional[list[dict]]) -> Optional[list[dict]]:
        return tools  # passthrough

    # ---- prompt-based fallback (for non-function-call models) -------------

    _TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(?P<name>[A-Za-z_][\w]*)\s*\|\s*(?P<args>\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )

    def inject_tool_prompt(self, system: str, tools: list[dict]) -> str:
        if not tools:
            return system
        descs = []
        for t in tools:
            fn = t.get("function") or {}
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            descs.append(f"- {name}: {desc}\n    parameters schema: {json.dumps(params)}")
        spec = "\n".join(descs)
        return (
            system + "\n\n"
            "When a tool call is needed, emit EXACTLY one block per call on its "
            "own line:\n"
            "  <tool_call>tool_name|{\"arg1\": value1, ...}</tool_call>\n"
            "After the block, stop output (the runtime will execute the tool "
            "and resume).\n\n"
            "Available tools:\n" + spec
        )

    def parse_tool_calls(self, content: str) -> list[dict]:
        if not content:
            return []
        out = []
        for i, m in enumerate(self._TOOL_CALL_RE.finditer(content)):
            out.append({
                "id": f"call_prompt_{i}",
                "type": "function",
                "function": {
                    "name": m.group("name"),
                    "arguments": m.group("args"),
                },
            })
        return out
