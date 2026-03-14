"""Fase 3: RAG (Retrieval-Augmented Generation) de documentos da empresa.

Pipeline leve de ingestão e busca semântica usando Chroma local e
embeddings do Ollama. Todos os erros são silenciados com fallback para
lista vazia — o bridge nunca deve travar por causa do RAG.

Uso rápido:
    # Ingestão (execute uma vez ou quando os documentos mudarem)
    python -c "from rag import ingest_documents; ingest_documents()"

    # Busca (usada automaticamente pelo orquestrador)
    from rag import search_documents
    chunks = search_documents("política de troca")
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("auto-reply-bridge.rag")

RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_DOCS_PATH = os.getenv("RAG_DOCS_PATH", "./docs")
RAG_VECTOR_DB_PATH = os.getenv("RAG_VECTOR_DB_PATH", "./data/vectorstore")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2500"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# Module-level singletons so Chroma is initialised once per process.
_chroma_client = None
_chroma_collection = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_chroma():
    """Return the Chroma collection, initialising it on first call."""
    global _chroma_client, _chroma_collection  # noqa: PLW0603

    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb  # optional heavy dependency

        db_path = Path(RAG_VECTOR_DB_PATH).resolve()
        db_path.mkdir(parents=True, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(path=str(db_path))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection loaded: %d chunk(s)", _chroma_collection.count()
        )
        return _chroma_collection
    except ImportError:
        logger.warning("chromadb not installed; RAG disabled")
        return None
    except Exception as exc:
        logger.warning("Failed to initialise Chroma: %s", exc)
        return None


def _get_embedding(text: str) -> Optional[list[float]]:
    """Return an embedding vector for *text* via the Ollama embeddings API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": RAG_EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError("Empty embedding returned")
        return embedding
    except Exception as exc:
        logger.warning("Failed to get embedding: %s", exc)
        return None


def _chunk_text(
    text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP
) -> list[str]:
    """Split *text* into overlapping chunks of approximately *size* chars."""
    if len(text) <= size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer natural break points
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, start + overlap, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def _load_document(file_path: Path) -> Optional[str]:
    """Read a supported document file and return its plain-text content."""
    suffix = file_path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix in (".html", ".htm"):
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            # Strip HTML tags and collapse whitespace
            text = re.sub(r"<[^>]+>", " ", raw)
            return re.sub(r"\s+", " ", text).strip()

        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF – optional

                doc = fitz.open(str(file_path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text
            except ImportError:
                logger.debug("PyMuPDF not installed; skipping PDF %s", file_path.name)
                return None
    except Exception as exc:
        logger.warning("Failed to load document %s: %s", file_path.name, exc)
        return None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_documents(docs_path: Optional[str] = None) -> int:
    """Ingest supported documents from *docs_path* into the vector store.

    Returns the number of chunks successfully stored.
    """
    if not RAG_ENABLED:
        logger.info("RAG disabled; skipping ingest")
        return 0

    collection = _get_chroma()
    if collection is None:
        return 0

    docs_dir = Path(docs_path or RAG_DOCS_PATH).resolve()
    if not docs_dir.exists():
        logger.warning("Docs path does not exist: %s", docs_dir)
        return 0

    ingested = 0
    supported = {".txt", ".md", ".html", ".htm", ".pdf"}

    for file_path in sorted(docs_dir.rglob("*")):
        if file_path.suffix.lower() not in supported:
            continue

        content = _load_document(file_path)
        if not content:
            continue

        chunks = _chunk_text(content)
        file_id = hashlib.md5(str(file_path).encode()).hexdigest()  # noqa: S324

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{i}"
            embedding = _get_embedding(chunk)
            if embedding is None:
                logger.warning(
                    "Could not embed chunk %d of %s; skipping", i, file_path.name
                )
                continue
            try:
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": file_path.name, "chunk": i}],
                )
                ingested += 1
            except Exception as exc:
                logger.warning("Failed to upsert chunk %s: %s", chunk_id, exc)

    logger.info("Ingested %d chunk(s) into RAG vector store", ingested)
    return ingested


def search_documents(query: str, top_k: Optional[int] = None) -> list[str]:
    """Return the top-k document chunks most relevant to *query*.

    Returns an empty list on any failure (graceful fallback).
    """
    if not RAG_ENABLED:
        return []

    collection = _get_chroma()
    if collection is None:
        return []

    if collection.count() == 0:
        return []

    k = min(top_k or RAG_TOP_K, collection.count())
    embedding = _get_embedding(query)
    if embedding is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents"],
        )
        docs: list[str] = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception as exc:
        logger.warning("RAG search failed: %s", exc)
        return []


def format_rag_context(chunks: list[str]) -> str:
    """Concatenate chunks into a context block, respecting RAG_MAX_CONTEXT_CHARS."""
    if not chunks:
        return ""

    selected: list[str] = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + len(chunk) > RAG_MAX_CONTEXT_CHARS:
            break
        selected.append(chunk)
        total_chars += len(chunk)

    if not selected:
        return ""

    return "Informações da empresa:\n" + "\n---\n".join(selected)
