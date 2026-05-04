"""Similarity scoring between two text outputs.

Two backends:
- embedding_similarity: sentence-transformers cosine. Cheap, deterministic, runs locally.
- judge_similarity: LLM-as-judge. Slower, more expensive, but better at semantic nuance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embedding_similarity(a: str, b: str) -> float:
    """Cosine similarity between sentence-transformer embeddings."""
    import numpy as np
    model = _get_model()
    embs = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def judge_similarity(a: str, b: str, judge_model: str = "gpt-4o-mini") -> float:
    """LLM-as-judge similarity. Returns float in [0, 1]."""
    import litellm

    prompt = (
        "Score the semantic similarity of the two responses below on a scale "
        "from 0.0 (totally different meaning) to 1.0 (same meaning). "
        "Reply with ONLY the number.\n\n"
        f"Response A:\n{a}\n\nResponse B:\n{b}\n\nScore:"
    )
    resp = litellm.completion(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        score = float(text.split()[0])
    except (ValueError, IndexError):
        score = 0.0
    return max(0.0, min(1.0, score))
