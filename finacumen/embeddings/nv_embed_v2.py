"""NV-Embed-v2 (and compatible) via OpenAI-style ``/v1/embeddings``.

Sibling to ``dashscope.py`` under ``finacumen.embeddings``.
"""

from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI

from finacumen.ft.config import EmbeddingSettings


async def embed_text_nv_openai_compatible(ec: EmbeddingSettings, text: str) -> np.ndarray:
    if not ec.base_url:
        raise RuntimeError("nv_embed_v2 embedding requires embedding.base_url")
    client = AsyncOpenAI(api_key=ec.api_key, base_url=ec.base_url)
    expected_dim = ec.resolved_dimensions()
    kw: dict = {"model": ec.model, "input": text}
    if ec.dimensions is not None:
        kw["dimensions"] = ec.dimensions
    resp = await client.embeddings.create(**kw)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    if vec.size != expected_dim:
        raise RuntimeError(f"embedding dim mismatch: expected {expected_dim}, got {vec.size}")
    return vec
