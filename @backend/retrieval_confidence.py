"""
Retrieval confidence: cosine similarity thresholds and RAG vs fallback routing.

Below the RAG threshold, use unsourced fallback (no IRS sources) so weak matches
do not fail strict grounding verification. Legacy env RETRIEVAL_HIGH_MIN_SIMILARITY
is still honored if RETRIEVAL_RAG_MIN_SIMILARITY is unset.
"""
import os
from typing import Any, Dict, List, Tuple


def _rag_similarity_threshold() -> float:
    """
    Minimum max cosine similarity (0–1) to run strict RAG (retrieve → context → verify).

    Default 0.52 avoids borderline matches (~0.43–0.50) that look related but are not
    actually grounded in the retrieved IRS text.
    """
    raw = (
        os.getenv("RETRIEVAL_RAG_MIN_SIMILARITY")
        or os.getenv("RETRIEVAL_HIGH_MIN_SIMILARITY")
        or "0.52"
    ).strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.52
    return max(0.0, min(1.0, v))


def assess_retrieval_confidence(chunks: List[Dict[str, Any]]) -> Tuple[float, bool]:
    """
    Returns (confidence_0_to_100, sufficient_for_strict_rag).

    confidence: best chunk's cosine similarity scaled to 0–100.
    sufficient_for_strict_rag: True iff best similarity >= RAG gate (see _rag_similarity_threshold).
    """
    if not chunks:
        return 0.0, False
    sims = [float(c.get("similarity", 0.0)) for c in chunks]
    max_sim = max(sims)
    confidence_pct = min(max_sim * 100.0, 100.0)
    sufficient = max_sim >= _rag_similarity_threshold()
    return round(confidence_pct, 1), sufficient
