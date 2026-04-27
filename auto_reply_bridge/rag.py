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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import requests

logger = logging.getLogger("auto-reply-bridge.rag")

_MODULE_DIR = Path(__file__).resolve().parent


def _read_bounded_int_env(
    var_name: str, default: int, min_value: int, max_value: int
) -> int:
    value = os.getenv(var_name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default=%s", var_name, value, default)
        return default

    if parsed < min_value:
        logger.warning("%s=%s below minimum %s; clamping", var_name, parsed, min_value)
        return min_value
    if parsed > max_value:
        logger.warning("%s=%s above maximum %s; clamping", var_name, parsed, max_value)
        return max_value
    return parsed


def _read_optional_int_env(var_name: str) -> Optional[int]:
    value = os.getenv(var_name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; ignoring", var_name, value)
        return None


def _resolve_path_setting(value: str, default_relative: str) -> str:
    """Resolve a configured path, treating relative paths as module-relative."""
    raw = (value or "").strip()
    if not raw:
        return str((_MODULE_DIR / default_relative).resolve())

    path = Path(raw)
    if path.is_absolute():
        return str(path)

    return str((_MODULE_DIR / path).resolve())

RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_DOCS_PATH = _resolve_path_setting(os.getenv("RAG_DOCS_PATH", "docs"), "docs")
RAG_VECTOR_DB_PATH = _resolve_path_setting(
    os.getenv("RAG_VECTOR_DB_PATH", "data/vectorstore"), "data/vectorstore"
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2500"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
OLLAMA_TIMEOUT_MAX_SECONDS = _read_bounded_int_env(
    "OLLAMA_TIMEOUT_MAX_SECONDS", default=300, min_value=30, max_value=1800
)
OLLAMA_MAIN_TIMEOUT = _read_bounded_int_env(
    "OLLAMA_MAIN_TIMEOUT", default=90, min_value=5, max_value=OLLAMA_TIMEOUT_MAX_SECONDS
)
RAG_OLLAMA_REQUEST_TIMEOUT = _read_bounded_int_env(
    "RAG_OLLAMA_REQUEST_TIMEOUT",
    default=OLLAMA_MAIN_TIMEOUT,
    min_value=5,
    max_value=OLLAMA_TIMEOUT_MAX_SECONDS,
)
RAG_OLLAMA_TAGS_TIMEOUT = _read_bounded_int_env(
    "RAG_OLLAMA_TAGS_TIMEOUT", default=10, min_value=2, max_value=120
)
RAG_OLLAMA_KEEP_ALIVE = os.getenv(
    "RAG_OLLAMA_KEEP_ALIVE", os.getenv("OLLAMA_KEEP_ALIVE", "5m")
).strip()
RAG_OLLAMA_NUM_GPU = _read_optional_int_env("RAG_OLLAMA_NUM_GPU")
if RAG_OLLAMA_NUM_GPU is None:
    RAG_OLLAMA_NUM_GPU = _read_optional_int_env("OLLAMA_NUM_GPU")
RAG_EMBED_RECHECK_SECONDS = int(os.getenv("RAG_EMBED_RECHECK_SECONDS", "30"))
_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
RAG_RECALL_K = int(os.getenv("RAG_RECALL_K", "12"))
RAG_EXPANDED_CONTEXT_CHARS = int(os.getenv("RAG_EXPANDED_CONTEXT_CHARS", "2200"))
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "").strip()
RAG_RERANK_TIMEOUT = int(os.getenv("RAG_RERANK_TIMEOUT", "60"))
RAG_RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "6"))

# ---------------------------------------------------------------------------
# Melhoria 2 — threshold mínimo de relevância.
# Resultados abaixo de RAG_MIN_SCORE (score composto) são descartados antes
# de serem enviados ao modelo, evitando contexto fraco ou irrelevante.
# RAG_MAX_DISTANCE filtra pela distância coseno bruta (opcional).
# ---------------------------------------------------------------------------

def _read_float_env(var_name: str, default: float) -> float:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default=%s", var_name, raw, default)
        return default

RAG_MIN_SCORE = _read_float_env("RAG_MIN_SCORE", 0.4)
RAG_MAX_DISTANCE = _read_float_env("RAG_MAX_DISTANCE", 0.0)  # 0.0 = desabilitado

logger.info(
    "RAG Ollama settings | base_url=%s keep_alive=%s timeout=%ss tags_timeout=%ss num_gpu=%s min_score=%s max_distance=%s",
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_KEEP_ALIVE,
    RAG_OLLAMA_REQUEST_TIMEOUT,
    RAG_OLLAMA_TAGS_TIMEOUT,
    RAG_OLLAMA_NUM_GPU if RAG_OLLAMA_NUM_GPU is not None else "default",
    RAG_MIN_SCORE,
    RAG_MAX_DISTANCE if RAG_MAX_DISTANCE > 0 else "off",
)

# Module-level singletons so Chroma is initialised once per process.
_chroma_client = None
_chroma_collection = None
_embed_model_checked = False
_embed_model_available = False
_embedding_disabled_reason: Optional[str] = None
_embeddings_disabled_until = 0.0


@dataclass(slots=True)
class RAGSearchResult:
    """Structured RAG result used by the recall and refinement pipeline."""

    candidate_id: str
    text: str
    source: str
    source_path: str
    doc_id: str
    score: float
    distance: Optional[float]
    start_offset: int
    end_offset: int
    evidence_chunks: list[str] = field(default_factory=list)


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


def _disable_embeddings(reason: str) -> None:
    """Disable embeddings for the current process and log a single warning."""
    global _embedding_disabled_reason, _embeddings_disabled_until  # noqa: PLW0603

    _embeddings_disabled_until = time.time() + max(RAG_EMBED_RECHECK_SECONDS, 0)

    if _embedding_disabled_reason is None:
        _embedding_disabled_reason = reason
        logger.warning(reason)


def _enable_embeddings() -> None:
    """Clear any temporary embedding disable state."""
    global _embedding_disabled_reason, _embeddings_disabled_until  # noqa: PLW0603

    _embedding_disabled_reason = None
    _embeddings_disabled_until = 0.0


def _check_embed_model_available() -> bool:
    """Check once whether the configured embedding model exists in Ollama."""
    global _embed_model_checked, _embed_model_available  # noqa: PLW0603

    if _embed_model_checked and _embed_model_available:
        return _embed_model_available

    if _embedding_disabled_reason is not None and time.time() < _embeddings_disabled_until:
        return False

    _embed_model_checked = True

    try:
        response = requests.get(
            f"{RAG_OLLAMA_BASE_URL}/api/tags", timeout=RAG_OLLAMA_TAGS_TIMEOUT
        )
        response.raise_for_status()

        models = (response.json().get("models") or [])
        names: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            for key in ("name", "model"):
                value = (model.get(key) or "").strip()
                if value:
                    names.add(value)

        # Accept exact match (including tag) or default-tag form "name:latest".
        _embed_model_available = (
            RAG_EMBED_MODEL in names
            or f"{RAG_EMBED_MODEL}:latest" in names
        )

        if _embed_model_available:
            _enable_embeddings()
        else:
            _disable_embeddings(
                "Embedding model not installed in Ollama: "
                f"{RAG_EMBED_MODEL}. Run: ollama pull {RAG_EMBED_MODEL}"
            )

        return _embed_model_available
    except Exception as exc:
        logger.warning("Could not verify embedding model availability: %s", exc)
        # If model discovery fails, keep old behavior and try the request anyway.
        _embed_model_available = True
        _enable_embeddings()
        return True


def _extract_embedding(payload: dict[str, Any]) -> Optional[list[float]]:
    """Extract embedding from Ollama response across API variants."""
    direct = payload.get("embedding")
    if isinstance(direct, list) and direct:
        return direct

    many = payload.get("embeddings")
    if isinstance(many, list) and many:
        first = many[0]
        if isinstance(first, list) and first:
            return first
        if isinstance(first, (int, float)):
            return many  # Some APIs may return a single vector directly.

    return None


def _get_embedding(text: str) -> Optional[list[float]]:
    """Return an embedding vector for *text* via the Ollama embeddings API."""
    if not _check_embed_model_available():
        return None

    last_error: Optional[Exception] = None

    # Try both API shapes to support different Ollama versions.
    embed_payload: dict[str, Any] = {
        "model": RAG_EMBED_MODEL,
        "keep_alive": RAG_OLLAMA_KEEP_ALIVE,
    }
    if RAG_OLLAMA_NUM_GPU is not None:
        embed_payload["options"] = {"num_gpu": RAG_OLLAMA_NUM_GPU}

    attempts = [
        ("/api/embed", {**embed_payload, "input": text}),
        ("/api/embeddings", {**embed_payload, "prompt": text}),
    ]

    saw_404 = False
    try:
        for path, body in attempts:
            try:
                response = requests.post(
                    f"{RAG_OLLAMA_BASE_URL}{path}",
                    json=body,
                    timeout=RAG_OLLAMA_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                embedding = _extract_embedding(response.json())
                if not embedding:
                    raise ValueError("Empty embedding returned")
                return embedding
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status == 404:
                    saw_404 = True
                    continue
                break
            except Exception as exc:
                last_error = exc
                break
    except Exception as exc:
        last_error = exc

    if saw_404:
        global _embed_model_available  # noqa: PLW0603
        _embed_model_available = False
        _disable_embeddings(
            "Ollama embedding endpoint is unavailable (404). "
            "Embeddings were disabled for this process."
        )
        return None

    if last_error is not None:
        logger.warning("Failed to get embedding: %s", last_error)

        return None


def _normalise_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) >= 2}


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = _normalise_tokens(query)
    if not query_tokens:
        return 0.0

    text_tokens = _normalise_tokens(text)
    if not text_tokens:
        return 0.0

    overlap = query_tokens & text_tokens
    if not overlap:
        return 0.0

    return len(overlap) / len(query_tokens)


def _find_left_boundary(text: str, lower_bound: int, anchor: int) -> int:
    for sep in ("\n\n", "\n", ". ", " "):
        pos = text.rfind(sep, lower_bound, anchor)
        if pos >= lower_bound:
            return pos + len(sep)
    return lower_bound


def _find_right_boundary(text: str, anchor: int, upper_bound: int) -> int:
    for sep in ("\n\n", "\n", ". ", " "):
        pos = text.find(sep, anchor, upper_bound)
        if pos != -1:
            return pos
    return upper_bound


def _expand_span(text: str, start_offset: int, end_offset: int) -> tuple[str, int, int]:
    """Expand a matched span into a broader context window with soft boundaries."""
    if not text:
        return "", 0, 0

    start = max(0, min(start_offset, len(text)))
    end = max(start, min(end_offset, len(text)))
    if len(text) <= RAG_EXPANDED_CONTEXT_CHARS:
        return text.strip(), 0, len(text)

    budget = max(RAG_EXPANDED_CONTEXT_CHARS, end - start)
    half_budget = budget // 2
    lower_bound = max(0, start - half_budget)
    upper_bound = min(len(text), end + half_budget)

    expanded_start = _find_left_boundary(text, lower_bound, start)
    expanded_end = _find_right_boundary(text, end, upper_bound)

    if expanded_end - expanded_start > budget:
        expanded_end = min(len(text), expanded_start + budget)

    if expanded_end <= expanded_start:
        expanded_start = lower_bound
        expanded_end = upper_bound

    return text[expanded_start:expanded_end].strip(), expanded_start, expanded_end


def _call_rerank_model(query: str, candidates: list[RAGSearchResult]) -> list[str]:
    """Optionally rerank candidates with a dedicated model.

    The model must return candidate IDs in best-to-worst order.
    """
    if not RAG_RERANK_MODEL or len(candidates) < 2:
        return []

    prompt_lines = [
        "Voce esta reranqueando trechos recuperados para responder uma pergunta.",
        "Retorne apenas os IDs dos candidatos mais relevantes em ordem, separados por virgula.",
        "Se nenhum candidato for relevante, retorne NONE.",
        "",
        f"Pergunta: {query}",
        "",
        "Candidatos:",
    ]
    for candidate in candidates[:RAG_RERANK_CANDIDATES]:
        prompt_lines.extend(
            [
                f"{candidate.candidate_id} | fonte={candidate.source}",
                candidate.text,
                "",
            ]
        )

    try:
        payload: dict[str, Any] = {
            "model": RAG_RERANK_MODEL,
            "stream": False,
            "keep_alive": RAG_OLLAMA_KEEP_ALIVE,
            "messages": [
                {
                    "role": "user",
                    "content": "\n".join(prompt_lines),
                }
            ],
        }
        if RAG_OLLAMA_NUM_GPU is not None:
            payload["options"] = {"num_gpu": RAG_OLLAMA_NUM_GPU}

        response = requests.post(
            f"{RAG_OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=RAG_RERANK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        content = ((payload.get("message") or {}).get("content") or "").strip()
        if not content or content.upper() == "NONE":
            return []

        ranked_ids = re.findall(r"C\d+", content.upper())
        return ranked_ids
    except Exception as exc:
        logger.warning("RAG rerank model failed: %s", exc)
        return []


def _chunk_text(
    text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP
) -> list[str]:
    """Split *text* into overlapping chunks of approximately *size* chars."""
    return [chunk for chunk, _start, _end in _chunk_text_with_offsets(text, size, overlap)]


# ---------------------------------------------------------------------------
# Melhoria 10 — regex para detectar títulos / seções em documentos
# empresariais.  Usado por _chunk_text_with_offsets para preferir quebrar
# ANTES de um título, preservando blocos semânticos coesos.
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+"           # Markdown headings: # Título, ## Sub, etc.
    r"|[A-ZÀ-Ú][A-ZÀ-Ú0-9 ,]{4,}$"  # Linhas ALL-CAPS (ex.: "POLÍTICA DE TROCAS")
    r"|\*\*[^*]+\*\*\s*$"  # **Bold title** em linha própria
    r"|\d+[\.\)]\s+"       # Listas numeradas: 1. Item, 2) Item
    r")",
    re.MULTILINE,
)


def _chunk_text_with_offsets(
    text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping chunks and preserve original offsets.

    Melhoria 10: detecta títulos (Markdown #, linhas ALL-CAPS, **bold**)
    e prefere quebrar ANTES deles, mantendo seções coesas nos chunks.
    """
    if not text:
        return []

    if size <= 0:
        return [(text.strip(), 0, len(text))] if text.strip() else []

    overlap = min(max(overlap, 0), max(size - 1, 0))
    if len(text) <= size:
        cleaned = text.strip()
        return [(cleaned, 0, len(text))] if cleaned else []

    # Pré-calcular posições de títulos para preferir quebra antes deles
    heading_positions: set[int] = set()
    for m in _HEADING_RE.finditer(text):
        heading_positions.add(m.start())

    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Melhoria 10: tentar quebrar ANTES de um título dentro da
            # janela [start+overlap, end], priorizando títulos sobre
            # separadores genéricos para manter seções coesas.
            best_heading_break = -1
            for hp in heading_positions:
                if start + overlap < hp <= end:
                    # Pegar o título mais próximo do fim do chunk
                    if hp > best_heading_break:
                        best_heading_break = hp

            if best_heading_break > start:
                end = best_heading_break
            else:
                # Fallback original: quebrar em separadores naturais
                for sep in ("\n\n", "\n", ". ", " "):
                    pos = text.rfind(sep, start + overlap, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
        raw_chunk = text[start:end]
        chunk = raw_chunk.strip()
        if chunk:
            left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            right_trim = len(raw_chunk.rstrip())
            chunk_start = start + left_trim
            chunk_end = start + right_trim
            chunks.append((chunk, chunk_start, chunk_end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _get_docs_root(docs_path: Optional[str] = None) -> Path:
    return Path(docs_path or RAG_DOCS_PATH).resolve()


def _resolve_source_path(source_path: str, docs_path: Optional[str] = None) -> Optional[Path]:
    if not source_path:
        return None

    base_path = _get_docs_root(docs_path)
    candidate = (base_path / source_path).resolve()
    try:
        candidate.relative_to(base_path)
    except ValueError:
        return None
    return candidate


def _infer_section_title(text: str, chunk_start: int) -> str:
    """Tenta inferir o título da seção que contém o chunk.

    Melhoria 10: olha para trás a partir de chunk_start buscando a linha
    de título mais próxima (Markdown #, ALL-CAPS, **bold**).
    Retorna string vazia se não encontrar.
    """
    # Buscar no trecho anterior ao chunk (até 500 chars atrás)
    search_start = max(0, chunk_start - 500)
    context = text[search_start:chunk_start]
    # Procurar a ÚLTIMA linha de título antes do chunk
    last_title = ""
    for m in _HEADING_RE.finditer(context):
        line_end = context.find("\n", m.start())
        if line_end == -1:
            line_end = len(context)
        candidate = context[m.start():line_end].strip()
        # Limpar marcadores Markdown
        candidate = re.sub(r"^#+\s*", "", candidate)
        candidate = re.sub(r"^\*\*|\*\*$", "", candidate)
        candidate = re.sub(r"^\d+[\.\)]\s*", "", candidate)
        if candidate:
            last_title = candidate
    return last_title


def _query_collection(
    collection: Any,
    embedding: list[float],
    n_results: int,
) -> dict[str, Any]:
    include_fields = cast(Any, ["documents", "metadatas", "distances"])
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=include_fields,
            where={"entry_type": "chunk"},
        )
        docs = (results.get("documents") or [[]])[0]
        if docs:
            return results
    except Exception as exc:
        logger.debug("Chunk-scoped query failed, falling back to legacy query: %s", exc)

    return collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=include_fields,
    )


def _rank_results(query: str, candidates: list[RAGSearchResult]) -> list[RAGSearchResult]:
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    reranked_ids = _call_rerank_model(query, ranked)
    if not reranked_ids:
        return ranked

    by_id = {candidate.candidate_id: candidate for candidate in ranked}
    reranked: list[RAGSearchResult] = []
    seen: set[str] = set()
    for candidate_id in reranked_ids:
        candidate = by_id.get(candidate_id)
        if candidate is None or candidate_id in seen:
            continue
        reranked.append(candidate)
        seen.add(candidate_id)

    reranked.extend(candidate for candidate in ranked if candidate.candidate_id not in seen)
    return reranked


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
                import fitz  # type: ignore[import-not-found]  # PyMuPDF - optional

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

    docs_dir = _get_docs_root(docs_path)
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

        file_id = hashlib.sha256(str(file_path).encode()).hexdigest()
        relative_source = file_path.relative_to(docs_dir).as_posix()

        try:
            collection.delete(where={"source_path": relative_source})
        except Exception:
            try:
                collection.delete(where={"source": file_path.name})
            except Exception:
                pass

        chunks = _chunk_text_with_offsets(content)

        for i, (chunk, start_offset, end_offset) in enumerate(chunks):
            chunk_id = f"{file_id}_{i}"
            embedding = _get_embedding(chunk)
            if embedding is None:
                if _embedding_disabled_reason is not None:
                    logger.info("Stopping ingest early: %s", _embedding_disabled_reason)
                    logger.info("Ingested %d chunk(s) into RAG vector store", ingested)
                    return ingested
                logger.warning(
                    "Could not embed chunk %d of %s; skipping", i, file_path.name
                )
                continue

            # Melhoria 10: inferir título da seção para metadado extra
            section_title = _infer_section_title(content, start_offset)

            try:
                meta: dict[str, Any] = {
                    "source": file_path.name,
                    "source_path": relative_source,
                    "doc_id": file_id,
                    "entry_type": "chunk",
                    "chunk": i,
                    "ordinal": i,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }
                if section_title:
                    meta["section_title"] = section_title

                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[meta],
                )
                ingested += 1
            except Exception as exc:
                logger.warning("Failed to upsert chunk %s: %s", chunk_id, exc)

    logger.info("Ingested %d chunk(s) into RAG vector store", ingested)
    return ingested


def search_documents_detailed(
    query: str,
    top_k: Optional[int] = None,
    docs_path: Optional[str] = None,
) -> list[RAGSearchResult]:
    """Return structured retrieval results after recall, expansion and reranking."""
    if not RAG_ENABLED:
        return []

    collection = _get_chroma()
    if collection is None:
        return []

    if collection.count() == 0:
        return []

    k = min(top_k or RAG_TOP_K, collection.count())
    recall_k = min(max(k, RAG_RECALL_K), collection.count())
    embedding = _get_embedding(query)
    if embedding is None:
        return []

    try:
        results = _query_collection(collection, embedding, recall_k)
        raw_docs = results.get("documents") or []
        raw_metas = results.get("metadatas") or []
        raw_distances = results.get("distances") or []
        first_docs = raw_docs[0] if raw_docs else []
        first_metas = raw_metas[0] if raw_metas else []
        first_distances = raw_distances[0] if raw_distances else []
        docs = first_docs if isinstance(first_docs, list) else []
        metas = first_metas if isinstance(first_metas, list) else []
        distances = first_distances if isinstance(first_distances, list) else []

        if not docs:
            return []

        source_cache: dict[str, str] = {}
        merged: dict[tuple[str, int, int], RAGSearchResult] = {}

        for index, (doc, meta) in enumerate(zip(docs, metas)):
            if not isinstance(doc, str) or not doc:
                continue

            source = meta.get("source", "") if isinstance(meta, dict) else ""
            source_path = meta.get("source_path", "") if isinstance(meta, dict) else ""
            doc_id = meta.get("doc_id", "") if isinstance(meta, dict) else ""
            start_offset = meta.get("start_offset", 0) if isinstance(meta, dict) else 0
            end_offset = meta.get("end_offset", len(doc)) if isinstance(meta, dict) else len(doc)

            expanded_text = doc
            expanded_start = 0
            expanded_end = len(doc)

            resolved_path = _resolve_source_path(source_path, docs_path)
            if resolved_path is not None:
                cache_key = resolved_path.as_posix()
                if cache_key not in source_cache:
                    loaded = _load_document(resolved_path)
                    source_cache[cache_key] = loaded or ""
                full_text = source_cache.get(cache_key, "")
                if full_text:
                    expanded_text, expanded_start, expanded_end = _expand_span(
                        full_text,
                        int(start_offset) if isinstance(start_offset, int) else 0,
                        int(end_offset) if isinstance(end_offset, int) else len(doc),
                    )

            distance = distances[index] if index < len(distances) else None
            vector_score = 0.0
            if isinstance(distance, (int, float)):
                vector_score = 1.0 / (1.0 + max(float(distance), 0.0))
            lexical_score = _lexical_overlap_score(query, expanded_text or doc)
            total_score = lexical_score * 1.5 + vector_score

            merged_key = (source_path or source, expanded_start, expanded_end)
            result = merged.get(merged_key)
            if result is None:
                candidate_id = f"C{len(merged) + 1}"
                merged[merged_key] = RAGSearchResult(
                    candidate_id=candidate_id,
                    text=expanded_text or doc,
                    source=source or source_path or "documento",
                    source_path=source_path,
                    doc_id=doc_id,
                    score=total_score,
                    distance=float(distance) if isinstance(distance, (int, float)) else None,
                    start_offset=expanded_start,
                    end_offset=expanded_end,
                    evidence_chunks=[doc],
                )
                continue

            result.score = max(result.score, total_score)
            if doc not in result.evidence_chunks:
                result.evidence_chunks.append(doc)

        ranked = _rank_results(query, list(merged.values()))

        # ---- Melhoria 2: filtro por threshold de relevância ----
        # Descarta resultados com score composto abaixo do mínimo
        # e/ou distância vetorial acima do máximo, evitando enviar
        # contexto fraco ao modelo.
        pre_filter_count = len(ranked)
        if RAG_MIN_SCORE > 0:
            ranked = [r for r in ranked if r.score >= RAG_MIN_SCORE]
        if RAG_MAX_DISTANCE > 0:
            ranked = [
                r for r in ranked
                if r.distance is None or r.distance <= RAG_MAX_DISTANCE
            ]
        filtered_out = pre_filter_count - len(ranked)
        if filtered_out > 0:
            logger.info(
                "RAG threshold filter | removed=%d kept=%d min_score=%.2f max_dist=%s",
                filtered_out, len(ranked), RAG_MIN_SCORE,
                RAG_MAX_DISTANCE if RAG_MAX_DISTANCE > 0 else "off",
            )

        return ranked[:k]
    except Exception as exc:
        logger.warning("RAG search failed: %s", exc)
        return []


def search_documents(query: str, top_k: Optional[int] = None) -> list[str]:
    """Return the top-k retrieved contexts as plain strings for compatibility."""
    results = search_documents_detailed(query, top_k=top_k)
    return [result.text for result in results if result.text]


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


# ---------------------------------------------------------------------------
# Melhoria 1 — contexto estruturado com evidências rastreáveis.
# Transforma list[RAGSearchResult] em bloco de texto que preserva fonte,
# score e caminho para auditoria no prompt final.
# ---------------------------------------------------------------------------


def format_rag_evidence_context(
    results: list[RAGSearchResult],
    max_chars: int = 0,
) -> str:
    """Formata evidências RAG em bloco estruturado para o prompt do LLM.

    Cada evidência inclui fonte, caminho, score e trecho — permitindo
    rastreabilidade e grounding no prompt final.

    Args:
        results: lista de RAGSearchResult retornada por search_documents_detailed.
        max_chars: limite de caracteres (0 = usar RAG_MAX_CONTEXT_CHARS).

    Returns:
        String formatada com blocos [EVIDÊNCIA N] ou string vazia se sem resultados.
    """
    if not results:
        return ""

    limit = max_chars or RAG_MAX_CONTEXT_CHARS
    blocks: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        block_lines = [
            f"[EVIDÊNCIA {i}]",
            f"fonte: {result.source}",
        ]
        if result.source_path and result.source_path != result.source:
            block_lines.append(f"caminho: {result.source_path}")
        block_lines.append(f"score: {result.score:.2f}")
        if result.distance is not None:
            block_lines.append(f"distância: {result.distance:.4f}")
        block_lines.append("trecho:")
        block_lines.append(result.text)

        block = "\n".join(block_lines)

        if total_chars + len(block) > limit and blocks:
            break
        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return ""

    return "\n\n".join(blocks)
