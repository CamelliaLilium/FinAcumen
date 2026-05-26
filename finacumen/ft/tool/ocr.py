"""
OCR tool for multimodal financial documents.

This tool is intentionally narrow:
- read image(s)
- return cleaned OCR text / markdown
- expose current-step image context for multimodal execution

It does NOT perform metric selection, table arbitration, or downstream reasoning.
"""

import asyncio
import contextvars
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from finacumen.ft.config import config
from finacumen.ft.logger import logger
from finacumen.ft.tool.base import BaseTool

_MAX_OCR_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_MAX_OCR_CONCURRENCY = 3
_OCR_SEMAPHORE = asyncio.Semaphore(_MAX_OCR_CONCURRENCY)
_SHARED_OCR_CLIENT: Optional[AsyncOpenAI] = None

_step_images_ctx: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "ocr_step_images", default=None
)
_step_context_ctx: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "ocr_step_context", default=None
)


def set_step_images_for_ocr(images: Optional[List[str]]) -> None:
    _step_images_ctx.set(images)


def get_step_images_for_ocr() -> Optional[List[str]]:
    return _step_images_ctx.get()


def set_step_context(ctx: Optional[Dict[str, str]]) -> None:
    _step_context_ctx.set(ctx)


def get_step_context() -> Optional[Dict[str, str]]:
    return _step_context_ctx.get()


def _ensure_data_url(base64_image: str) -> str:
    s = (base64_image or "").strip()
    if not s:
        return ""
    if s.startswith("data:"):
        return s
    mime = "image/jpeg" if s.startswith("/9j/") else "image/png"
    return f"data:{mime};base64,{s}"


def _html_table_to_markdown(html: str) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html or "", re.IGNORECASE | re.DOTALL)
    if not rows:
        return html

    markdown_lines: List[str] = []
    expected_cols = None
    for row_idx, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)
        cleaned = []
        for cell in cells:
            cell = re.sub(r"<[^>]+>", "", cell or "")
            cell = re.sub(r"\s+", " ", cell).strip()
            cleaned.append(cell)
        if not cleaned:
            continue
        if expected_cols is None:
            expected_cols = len(cleaned)
        elif expected_cols and len(cleaned) != expected_cols:
            if len(cleaned) < expected_cols:
                cleaned.extend([""] * (expected_cols - len(cleaned)))
            else:
                cleaned = cleaned[:expected_cols]

        markdown_lines.append("| " + " | ".join(cleaned) + " |")
        if row_idx == 0:
            markdown_lines.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
    return "\n".join(markdown_lines) if markdown_lines else html


def _replace_html_tables_with_markdown(text: str) -> str:
    if not text or "<table" not in text.lower():
        return text
    return re.sub(
        r"<table.*?>.*?</table>",
        lambda m: "\n" + _html_table_to_markdown(m.group(0)) + "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


class OcrExtract(BaseTool):
    name: str = "ocr_extract"
    description: str = (
        "Extract OCR text/markdown from image(s). Supports base64_image, base64_images, "
        "or use_context_image=true for the current multimodal step."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "base64_image": {"type": "string"},
            "base64_images": {"type": "array", "items": {"type": "string"}},
            "use_context_image": {"type": "boolean"},
        },
        "required": [],
    }

    _OCR_PROMPT = (
        "Extract ALL text and table data from this image. Output as clean markdown. "
        "Preserve all numbers and labels exactly. "
        "For charts: describe axis labels, data ranges, and visible values in markdown."
    )

    _OCR_SYSTEM_PROMPT = ""

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def _get_client(cls) -> AsyncOpenAI:
        global _SHARED_OCR_CLIENT
        if _SHARED_OCR_CLIENT is not None:
            return _SHARED_OCR_CLIENT
        ocr_cfg = getattr(config, "ocr_config", None)
        if not ocr_cfg:
            raise RuntimeError(
                "OCR config not found. Add [ocr] section in config.toml with model, base_url, api_key."
            )
        _SHARED_OCR_CLIENT = AsyncOpenAI(
            api_key=ocr_cfg.api_key,
            base_url=ocr_cfg.base_url,
        )
        return _SHARED_OCR_CLIENT

    async def _ocr_single(self, base64_image: str) -> Dict[str, Any]:
        client = self._get_client()
        ocr_cfg = getattr(config, "ocr_config", None)
        url = _ensure_data_url(base64_image)

        last_err: Optional[Exception] = None
        for attempt in range(_MAX_OCR_RETRIES):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._OCR_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ]
                if self._OCR_SYSTEM_PROMPT:
                    messages.insert(0, {"role": "system", "content": self._OCR_SYSTEM_PROMPT})
                response = await client.chat.completions.create(
                    model=ocr_cfg.model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0,
                )
                raw = (response.choices[0].message.content or "").strip()
                cleaned = re.sub(r"^```[\w-]*\n?", "", raw)
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()
                # Strip proprietary formatting tokens from DeepSeek-OCR
                cleaned = re.sub(r"<\|[^|>]+\|>", "", cleaned)
                # Collapse excessive blank lines
                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
                markdown = _replace_html_tables_with_markdown(cleaned)
                logger.info(f"[ocr_extract] OCR preview: {(markdown or '')[:240]}...")
                return {
                    "success": True,
                    "text": cleaned,
                    "markdown": markdown,
                    "regions": [],
                }
            except Exception as e:
                last_err = e
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"[ocr_extract] attempt {attempt + 1}/{_MAX_OCR_RETRIES} failed: {e}; "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        raise last_err

    async def execute(
        self,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
        use_context_image: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        base64_image = base64_image or kwargs.get("base64_image")
        base64_images = base64_images or kwargs.get("base64_images")
        use_context_image = kwargs.get("use_context_image", use_context_image)

        if use_context_image and not base64_image and not base64_images:
            ctx_images = get_step_images_for_ocr() or []
            if len(ctx_images) == 1:
                base64_image = ctx_images[0]
            elif len(ctx_images) > 1:
                base64_images = ctx_images

        try:
            async with _OCR_SEMAPHORE:
                if base64_images:
                    results = []
                    for idx, image in enumerate(base64_images):
                        item = await self._ocr_single(image)
                        results.append(
                            {
                                "index": idx,
                                "text": item.get("text", ""),
                                "markdown": item.get("markdown", ""),
                                "regions": item.get("regions", []),
                            }
                        )
                    return {"success": True, "results": results}

                if base64_image:
                    return await self._ocr_single(base64_image)

                return {
                    "success": False,
                    "error": "No image provided. Use base64_image, base64_images, or use_context_image=true.",
                }
        except Exception as e:
            logger.warning(f"[ocr_extract] failed: {e}")
            return {"success": False, "error": str(e)}
