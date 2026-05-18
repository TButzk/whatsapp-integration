"""Chroma-backed store for pre-written Q&A pairs.

Each Q&A pair can have multiple text variants (the canonical question +
optional aliases).  Every variant is stored as a separate Chroma document
all pointing to the same answer via metadata, so retrieval is robust even
when the customer phrases things differently.

Metadata layout per Chroma document
------------------------------------
{
    "qa_id":    "frete-01",
    "question": "Qual o prazo de entrega?",   # canonical question
    "answer":   "O prazo é de 3 a 7 dias...",
    "category": "frete",
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from local_ai_backend.config import Settings

logger = logging.getLogger("local-ai-backend.rag")

# Lazy Chroma import so the module can be imported even when chromadb is not
# installed (RAG disabled path).  Error is raised only at instantiation time.
try:
    import chromadb  # type: ignore[import-untyped]

    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CHROMA_AVAILABLE = False


@dataclass
class QAHit:
    qa_id: str
    question: str  # canonical question stored with this pair
    answer: str
    category: str
    score: float  # cosine similarity [0, 1]; higher = better match


@dataclass
class _StoredPair:
    qa_id: str
    question: str
    answer: str
    category: str


class QAStore:
    """Persistent Chroma collection of Q&A pairs with pre-computed embeddings."""

    def __init__(self, settings: Settings) -> None:
        if not _CHROMA_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed.  Run: pip install chromadb"
            )
        self._collection_name = settings.rag_collection_name
        client = chromadb.PersistentClient(path=settings.rag_vectorstore_path)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "qa_store_ready collection=%s path=%s",
            self._collection_name,
            settings.rag_vectorstore_path,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_pair(
        self,
        qa_id: str,
        question: str,
        answer: str,
        category: str,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert all text variants (question + aliases) for one Q&A pair."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        ids = [f"{qa_id}:v{i}" for i in range(len(texts))]
        metadatas = [
            {
                "qa_id": qa_id,
                "question": question,
                "answer": answer,
                "category": category,
            }
            for _ in texts
        ]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("upserted qa_id=%s variants=%d", qa_id, len(texts))

    def delete_pair(self, qa_id: str) -> None:
        """Remove all variants for a given qa_id."""
        results = self._collection.get(where={"qa_id": qa_id}, include=[])
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[QAHit]:
        """Return top_k hits sorted by descending cosine similarity."""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["metadatas", "distances", "documents"],
        )

        hits: list[QAHit] = []
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        for dist, meta in zip(distances, metadatas):
            # Chroma cosine space: distance = 1 - cosine_similarity
            score = max(0.0, 1.0 - dist)
            hits.append(
                QAHit(
                    qa_id=meta["qa_id"],
                    question=meta["question"],
                    answer=meta["answer"],
                    category=meta["category"],
                    score=score,
                )
            )

        # Deduplicate by qa_id (same answer may appear via different aliases)
        seen: set[str] = set()
        unique: list[QAHit] = []
        for hit in hits:
            if hit.qa_id not in seen:
                seen.add(hit.qa_id)
                unique.append(hit)

        return unique

    def list_all(self) -> list[_StoredPair]:
        """Return one entry per distinct qa_id."""
        results = self._collection.get(include=["metadatas"])
        seen: set[str] = set()
        pairs: list[_StoredPair] = []
        for meta in results["metadatas"]:
            qid = meta["qa_id"]
            if qid not in seen:
                seen.add(qid)
                pairs.append(
                    _StoredPair(
                        qa_id=qid,
                        question=meta["question"],
                        answer=meta["answer"],
                        category=meta["category"],
                    )
                )
        return pairs

    @property
    def count(self) -> int:
        return self._collection.count()
