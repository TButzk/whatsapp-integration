"""High-level retriever: given a customer question, return the best pre-written answer."""

from __future__ import annotations

import logging

from local_ai_backend.config import Settings
from local_ai_backend.rag.embedder import Embedder
from local_ai_backend.rag.qa_store import QAHit, QAStore

logger = logging.getLogger("local-ai-backend.rag")


def find_best_answer(
    question: str,
    store: QAStore,
    embedder: Embedder,
    settings: Settings,
) -> QAHit | None:
    """Embed *question* and return the best matching Q&A hit.

    Returns ``None`` when:
    - the store is empty
    - embedding fails
    - best score is below ``rag_min_score_hint``
    """
    try:
        vec = embedder.embed(question)
    except Exception as exc:
        logger.warning("rag_embed_failed question=%.50r error=%s", question, exc)
        return None

    hits = store.search(vec, top_k=3)
    if not hits:
        return None

    best = hits[0]
    logger.debug(
        "rag_search best_score=%.3f qa_id=%s question=%.60r",
        best.score,
        best.qa_id,
        question,
    )

    if best.score < settings.rag_min_score_hint:
        return None

    return best
