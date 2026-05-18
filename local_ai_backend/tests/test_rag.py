"""Tests for the RAG / Q&A semantic retrieval layer."""

from unittest.mock import Mock

import httpx
import pytest

from local_ai_backend.config import Settings
from local_ai_backend.main import create_app
from local_ai_backend.queueing import InMemoryMessagePublisher


# ---------------------------------------------------------------------------
# Endpoints — RAG disabled
# ---------------------------------------------------------------------------


def _plain_client():
    from fastapi.testclient import TestClient

    settings = Settings(
        APP_NAME="test-rag-disabled",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="q",
        DATABASE_URL="postgresql+psycopg://p:p@db/app",
        OLLAMA_BASE_URL="http://ollama.internal",
    )
    return TestClient(create_app(settings, publisher=InMemoryMessagePublisher()))


def test_import_qa_returns_400_when_rag_disabled() -> None:
    client = _plain_client()
    response = client.post(
        "/admin/knowledge/import-qa",
        json={"pairs": [{"id": "t-01", "question": "Test?", "answer": "Yes.", "category": "test"}]},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "rag_not_enabled"


def test_list_qa_pairs_returns_400_when_rag_disabled() -> None:
    client = _plain_client()
    response = client.get("/admin/knowledge/qa-pairs")
    assert response.status_code == 400
    assert response.json()["detail"] == "rag_not_enabled"


# ---------------------------------------------------------------------------
# Embedder — mock Ollama transport
# ---------------------------------------------------------------------------


def test_embedder_calls_ollama_embed_endpoint() -> None:
    from local_ai_backend.rag.embedder import Embedder

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(status_code=200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    transport = httpx.MockTransport(handler)
    embedder = Embedder(
        settings=Settings(OLLAMA_BASE_URL="http://ollama.test"),
        http_client=httpx.Client(transport=transport),
    )

    vec = embedder.embed("prazo de entrega")

    assert vec == [0.1, 0.2, 0.3]


def test_embedder_handles_flat_embedding_response() -> None:
    from local_ai_backend.rag.embedder import Embedder

    def handler(request: httpx.Request) -> httpx.Response:
        # Some Ollama builds return a flat list
        return httpx.Response(status_code=200, json={"embeddings": [[0.5, 0.6]]})

    transport = httpx.MockTransport(handler)
    embedder = Embedder(
        settings=Settings(OLLAMA_BASE_URL="http://ollama.test"),
        http_client=httpx.Client(transport=transport),
    )

    vec = embedder.embed("teste")
    assert vec == [0.5, 0.6]


# ---------------------------------------------------------------------------
# Retriever — mock store + embedder
# ---------------------------------------------------------------------------


def test_find_best_answer_returns_none_when_score_too_low() -> None:
    from local_ai_backend.rag.qa_store import QAHit
    from local_ai_backend.rag.retriever import find_best_answer

    mock_embedder = Mock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_store = Mock()
    mock_store.search.return_value = [
        QAHit(qa_id="x", question="Q", answer="A", category="c", score=0.40)
    ]

    result = find_best_answer("alguma coisa", mock_store, mock_embedder, Settings(RAG_MIN_SCORE_HINT=0.60))

    assert result is None


def test_find_best_answer_returns_hit_when_score_above_hint() -> None:
    from local_ai_backend.rag.qa_store import QAHit
    from local_ai_backend.rag.retriever import find_best_answer

    mock_embedder = Mock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_store = Mock()
    mock_store.search.return_value = [
        QAHit(qa_id="frete-01", question="Qual o prazo?", answer="3 a 7 dias.", category="frete", score=0.78)
    ]

    result = find_best_answer(
        "prazo de entrega",
        mock_store,
        mock_embedder,
        Settings(RAG_MIN_SCORE_HINT=0.60, RAG_MIN_SCORE_DIRECT=0.85),
    )

    assert result is not None
    assert result.qa_id == "frete-01"


def test_find_best_answer_returns_none_when_embed_fails() -> None:
    from local_ai_backend.rag.retriever import find_best_answer

    mock_embedder = Mock()
    mock_embedder.embed.side_effect = RuntimeError("ollama down")
    mock_store = Mock()

    result = find_best_answer("qualquer coisa", mock_store, mock_embedder, Settings())

    assert result is None
    mock_store.search.assert_not_called()


# ---------------------------------------------------------------------------
# QAStore — Chroma EphemeralClient (in-memory, no disk)
# ---------------------------------------------------------------------------

chromadb = pytest.importorskip("chromadb")


def _make_store(collection_name: str = "test_qa"):
    """Return a QAStore backed by an in-memory Chroma client."""
    from local_ai_backend.rag.qa_store import QAStore

    settings = Settings(RAG_COLLECTION_NAME=collection_name, RAG_VECTORSTORE_PATH="ignored")

    # Monkey-patch chromadb.PersistentClient to use EphemeralClient so no disk I/O
    import local_ai_backend.rag.qa_store as _mod
    orig = _mod.chromadb.PersistentClient

    def _ephemeral(path):  # noqa: ARG001
        return chromadb.EphemeralClient()

    _mod.chromadb.PersistentClient = _ephemeral
    try:
        store = QAStore(settings)
    finally:
        _mod.chromadb.PersistentClient = orig
    return store


def _fake_vec(n: int = 3) -> list[float]:
    return [float(n) / 10, float(n + 1) / 10, float(n + 2) / 10]


def test_qa_store_upsert_and_count() -> None:
    store = _make_store("col_upsert")
    store.upsert_pair(
        qa_id="frete-01",
        question="Qual o prazo de entrega?",
        answer="3 a 7 dias úteis.",
        category="frete",
        texts=["Qual o prazo de entrega?", "quanto tempo demora"],
        embeddings=[_fake_vec(1), _fake_vec(2)],
    )
    assert store.count == 2  # two variants


def test_qa_store_list_all_deduplicates_by_qa_id() -> None:
    store = _make_store("col_list")
    store.upsert_pair(
        qa_id="frete-01",
        question="Prazo?",
        answer="7 dias.",
        category="frete",
        texts=["Prazo?", "quanto demora"],
        embeddings=[_fake_vec(1), _fake_vec(2)],
    )
    pairs = store.list_all()
    assert len(pairs) == 1
    assert pairs[0].qa_id == "frete-01"


def test_qa_store_search_returns_hit() -> None:
    store = _make_store("col_search")
    vec = _fake_vec(1)
    store.upsert_pair(
        qa_id="frete-01",
        question="Prazo?",
        answer="7 dias.",
        category="frete",
        texts=["Prazo?"],
        embeddings=[vec],
    )
    hits = store.search(vec, top_k=1)
    assert len(hits) == 1
    assert hits[0].qa_id == "frete-01"
    assert hits[0].score >= 0.99  # querying with the exact same vector


def test_qa_store_delete_pair() -> None:
    store = _make_store("col_delete")
    store.upsert_pair(
        qa_id="del-01",
        question="Q?",
        answer="A.",
        category="c",
        texts=["Q?", "alias"],
        embeddings=[_fake_vec(1), _fake_vec(2)],
    )
    assert store.count == 2
    store.delete_pair("del-01")
    assert store.count == 0


# ---------------------------------------------------------------------------
# Wiring: RAG in _generate_auto_reply (mocked rag_store + rag_embedder)
# ---------------------------------------------------------------------------


def _build_rag_app(rag_hit=None):
    from fastapi.testclient import TestClient
    from local_ai_backend.rag.qa_store import QAHit  # noqa: F401

    settings = Settings(
        APP_NAME="test-rag-wired",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="q",
        DATABASE_URL="postgresql+psycopg://p:p@db/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
        RAG_ENABLED=True,
        RAG_MIN_SCORE_DIRECT=0.85,
        RAG_MIN_SCORE_HINT=0.60,
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())

    mock_embedder = Mock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_store = Mock()
    mock_store.search.return_value = [rag_hit] if rag_hit is not None else []

    app.state.rag_embedder = mock_embedder
    app.state.rag_store = mock_store
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.llm_client.chat = Mock(
        return_value=type("_R", (), {"content": "Resposta LLM"})()
    )
    return app, TestClient(app)


_WEBHOOK_PAYLOAD = {
    "event": "message_created",
    "account": {"id": 1},
    "message_type": "incoming",
    "private": False,
    "conversation": {"id": 700, "status": "open"},
}


def test_rag_direct_hit_bypasses_llm() -> None:
    """score >= rag_min_score_direct → LLM called once for validation (SIM) then pre-written answer sent."""
    from local_ai_backend.rag.qa_store import QAHit

    hit = QAHit(qa_id="frete-01", question="Prazo?", answer="Entrega em 3 a 7 dias úteis.", category="frete", score=0.92)
    app, client = _build_rag_app(rag_hit=hit)

    # Pre-seed: conversation is waiting for a free-text question (skips menu)
    state = app.state.conv_state_store.get("700")
    state.flow_state = "PRODUTO_DUVIDA"
    app.state.conv_state_store.save(state)

    # Validation must return "SIM" so the direct RAG answer is accepted; no generation call follows
    app.state.llm_client.chat = Mock(
        side_effect=[type("_R", (), {"content": "SIM"})()]
    )

    client.post("/webhook/chatwoot", json={**_WEBHOOK_PAYLOAD, "content": "qual o prazo de entrega"})

    app.state.llm_client.chat.assert_called_once()  # validation only; no LLM generation
    app.state.chatwoot_client.send_conversation_reply.assert_called_once()
    sent = app.state.chatwoot_client.send_conversation_reply.call_args.kwargs.get(
        "content",
        app.state.chatwoot_client.send_conversation_reply.call_args.args[2]
        if len(app.state.chatwoot_client.send_conversation_reply.call_args.args) > 2
        else "",
    )
    assert "3 a 7 dias" in sent


def test_rag_hint_passes_context_to_llm() -> None:
    """score in [hint, direct) → LLM reformats with pre-written answer as context."""
    from local_ai_backend.rag.qa_store import QAHit

    hit = QAHit(qa_id="frete-02", question="Prazo?", answer="Entrega em 5 dias.", category="frete", score=0.72)
    app, client = _build_rag_app(rag_hit=hit)

    # Pre-seed: conversation is waiting for a free-text question (skips menu)
    state = app.state.conv_state_store.get("700")
    state.flow_state = "PRODUTO_DUVIDA"
    app.state.conv_state_store.save(state)

    client.post("/webhook/chatwoot", json={**_WEBHOOK_PAYLOAD, "content": "qual o prazo de entrega"})

    app.state.llm_client.chat.assert_called_once()
    system_prompt = app.state.llm_client.chat.call_args.kwargs.get("system_prompt", "")
    assert "Entrega em 5 dias" in system_prompt


def test_rag_no_hit_falls_through_to_normal_flow() -> None:
    """No RAG hit → intent/policy/LLM flow runs; RAG embed was at least called."""
    app, client = _build_rag_app(rag_hit=None)

    # Pre-seed: conversation is waiting for a free-text question (skips menu)
    state = app.state.conv_state_store.get("700")
    state.flow_state = "PRODUTO_DUVIDA"
    app.state.conv_state_store.save(state)

    client.post("/webhook/chatwoot", json={**_WEBHOOK_PAYLOAD, "content": "oi como vai"})

    app.state.rag_embedder.embed.assert_called_once()
