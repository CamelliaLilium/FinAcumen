"""DashScope semantic embeddings via official ``dashscope.TextEmbedding`` (sync).

Sibling to ``nv_embed_v2.py`` under ``finacumen.embeddings``.
"""

from __future__ import annotations

import numpy as np


def embed_text_sync(*, api_key: str, model: str, text: str, expected_dim: int) -> np.ndarray:
    import dashscope
    from dashscope import TextEmbedding

    dashscope.api_key = api_key
    rsp = TextEmbedding.call(model=model, input=text)
    if rsp.status_code != 200:
        raise RuntimeError(
            f"dashscope.TextEmbedding failed: code={getattr(rsp, 'code', '')} "
            f"message={getattr(rsp, 'message', rsp)}"
        )
    out = rsp.output
    if not out or "embeddings" not in out or not out["embeddings"]:
        raise RuntimeError("dashscope embedding response missing output['embeddings']")
    vec = np.asarray(out["embeddings"][0]["embedding"], dtype=np.float32)
    if vec.size != expected_dim:
        raise RuntimeError(f"embedding dim mismatch: expected {expected_dim}, got {vec.size}")
    return vec
