"""Embedding facade — unified API over DashScope and NV-Embed-v2 backends.

No fallback — failures propagate as exceptions.
"""
from __future__ import annotations

import asyncio

import numpy as np

from finacumen.ft.config import config


_INSTRUCTION_TEMPLATE = (
    "Instruct: Given prior financial analysis experiences, select relevant "
    "past entries that could help answer this question.\n"
    "Query: {question}\n"
    "Context: {context_snippet}"
)


def embedding_dimension() -> int:
    ec = config.embedding_config
    if ec is None:
        raise RuntimeError("[embedding] section missing from config.toml")
    return ec.resolved_dimensions()


async def embed_text(text: str) -> np.ndarray:
    """Embed text → (D,) float32 numpy array."""
    if not text or not text.strip():
        raise ValueError("embed_text requires non-empty text")
    ec = config.embedding_config
    if ec is None:
        raise RuntimeError("[embedding] section missing from config.toml")
    expected_dim = ec.resolved_dimensions()

    try:
        if ec.provider == "dashscope":
            from finacumen.embeddings.dashscope import embed_text_sync
            result = await asyncio.to_thread(
                embed_text_sync,
                api_key=ec.api_key, model=ec.model,
                text=text, expected_dim=expected_dim,
            )
        else:
            from finacumen.embeddings.nv_embed_v2 import embed_text_nv_openai_compatible
            result = await embed_text_nv_openai_compatible(ec, text)
    except Exception as e:
        raise RuntimeError(
            f"embed_text failed: provider={ec.provider} model={ec.model}: {e}"
        ) from e

    if result.shape[-1] != expected_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {result.shape[-1]}, expected {expected_dim}"
        )
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine: zero-norm vector")
    return float(np.dot(a, b) / (na * nb))
