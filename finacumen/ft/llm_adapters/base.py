"""LLMAdapter Protocol — uniform interface for per-model behavior.

Each adapter answers four questions for one (or a family of) model id:
  1. How to build the API client (api_key/base_url/headers)
  2. Does this model support OpenAI tools API?
  3. How to format user content (string vs multimodal-list)
  4. If tools unsupported, how to inject + parse tool calls via prompt

Adapters are stateless singletons; their methods are called by `app.llm.LLM`.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable, Any


@runtime_checkable
class LLMAdapter(Protocol):
    """Per-model behavior contract."""

    name: str

    def matches(self, model_id: str) -> bool:
        """Return True if this adapter handles the given model id."""
        ...

    @property
    def supports_function_calling(self) -> bool:
        """Whether OpenAI tools/tool_choice API works for this model."""
        ...

    def resolve_api_key(self, settings_api_key: str) -> str:
        """Pick the right api_key (config vs env) for this model.

        Default behavior: trust the config key when it's a real `sk-` key;
        otherwise fall back to a model-specific env var.
        """
        ...

    def prepare_content(self, text: str, images: Optional[list[str]] = None) -> Any:
        """Build the user-content payload (string or list-of-blocks).

        Returns either:
          - str (for text-only models that reject multimodal lists)
          - list[dict] (OpenAI multimodal format)
        """
        ...

    def prepare_tools(self, tools: Optional[list[dict]]) -> Optional[list[dict]]:
        """Sanitize/filter the tools list for this model.

        Return None to indicate "do not pass tools to this model" — the
        caller should then use prompt-based dispatch via inject_tool_prompt
        and parse_tool_calls.
        """
        ...

    def inject_tool_prompt(self, system: str, tools: list[dict]) -> str:
        """For non-function-call models: inject tool spec into system prompt.

        Returns a system message string instructing the model to emit
        `<tool_call>name|json_args</tool_call>` blocks; caller parses them
        with `parse_tool_calls`.
        """
        ...

    def parse_tool_calls(self, content: str) -> list[dict]:
        """Parse `<tool_call>name|{...}</tool_call>` blocks from text output.

        Returns OpenAI-compatible tool_calls list:
          [{"id": "call_0", "type": "function",
            "function": {"name": "...", "arguments": "{...}"}}, ...]
        """
        ...

    min_temperature: float
    """Minimum temperature this model accepts. Most adapters allow 0.0;
    Thinking-style models (GLM-4.1V) reject T=0 with 400. Callers should
    `max(temperature, adapter.min_temperature)` before submitting."""

    supports_images: bool
    """Whether this model can process image content in multimodal format.
    When True, LLM.format_messages keeps base64_image fields and converts
    them to OpenAI image_url blocks. When False, image fields are stripped."""

    use_max_completion_tokens: bool
    """Use max_completion_tokens param instead of max_tokens (reasoning models like o1/o3).
    Most models use max_tokens; set True only for OpenAI reasoning models."""
