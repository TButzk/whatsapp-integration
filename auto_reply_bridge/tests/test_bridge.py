"""Tests for the auto-reply bridge — all phases.

Run from the auto_reply_bridge directory:
    python -m pytest tests/ -v
"""

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Ensure the bridge package is importable when pytest is run from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Disable RAG / history by default so tests run without external services
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("CHAT_HISTORY_ENABLED", "false")
os.environ.setdefault("CHATWOOT_WEBHOOK_SECRET", "")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(
    content: str = "Olá",
    msg_type: str = "incoming",
    private: bool = False,
    sender_type: str = "contact",
    conversation_status: str = "open",
    account_id: int = 1,
    conversation_id: int = 42,
) -> dict:
    return {
        "event": "message_created",
        "message_type": msg_type,
        "private": private,
        "content": content,
        "id": 100,
        "account": {"id": account_id},
        "conversation": {"id": conversation_id, "status": conversation_status, "channel": "whatsapp"},
        "contact": {"id": 5, "name": "Fulano"},
        "sender": {"type": sender_type},
    }


# ===========================================================================
# Phase 1 — chat_history
# ===========================================================================

class TestChatHistory:
    """Tests for chat_history module."""

    def test_filter_and_format_basic(self):
        from chat_history import _filter_and_format

        messages = [
            {"message_type": "incoming", "private": False, "content": "Oi", "created_at": 1},
            {"message_type": "outgoing", "private": False, "content": "Olá!", "created_at": 2, "sender": {"type": "agent"}},
        ]
        result = _filter_and_format(messages)
        assert len(result) == 2
        assert result[0]["role"] == "Cliente"
        assert result[1]["role"] == "Atendente"

    def test_filter_accepts_numeric_message_types(self):
        from chat_history import _filter_and_format

        messages = [
            {"message_type": 0, "private": False, "content": "Oi", "created_at": 1, "sender_type": "contact"},
            {"message_type": 1, "private": False, "content": "Olá!", "created_at": 2, "sender_type": "user"},
        ]
        result = _filter_and_format(messages)
        assert len(result) == 2
        assert result[0]["role"] == "Cliente"
        assert result[1]["role"] == "Atendente"

    def test_filter_skips_private_messages(self):
        from chat_history import _filter_and_format

        messages = [
            {"message_type": "incoming", "private": True, "content": "nota privada", "created_at": 1},
            {"message_type": "incoming", "private": False, "content": "mensagem real", "created_at": 2},
        ]
        result = _filter_and_format(messages)
        assert len(result) == 1
        assert result[0]["content"] == "mensagem real"

    def test_filter_skips_empty_content(self):
        from chat_history import _filter_and_format

        messages = [
            {"message_type": "incoming", "private": False, "content": "   ", "created_at": 1},
            {"message_type": "incoming", "private": False, "content": "ok", "created_at": 2},
        ]
        result = _filter_and_format(messages)
        assert len(result) == 1

    def test_filter_deduplicates_content(self):
        from chat_history import _filter_and_format

        messages = [
            {"message_type": "incoming", "private": False, "content": "oi", "created_at": 1},
            {"message_type": "incoming", "private": False, "content": "oi", "created_at": 2},
        ]
        result = _filter_and_format(messages)
        assert len(result) == 1

    def test_max_messages_limit(self, monkeypatch):
        import chat_history
        monkeypatch.setattr(chat_history, "CHAT_HISTORY_MAX_MESSAGES", 2)

        messages = [
            {"message_type": "incoming", "private": False, "content": f"msg{i}", "created_at": i}
            for i in range(5)
        ]
        result = chat_history._filter_and_format(messages)
        assert len(result) <= 2

    def test_format_history_context_empty(self):
        from chat_history import format_history_context
        assert format_history_context([]) == ""

    def test_format_history_context_nonempty(self):
        from chat_history import format_history_context
        history = [
            {"role": "Cliente", "content": "Qual o prazo?"},
            {"role": "Bot", "content": "3 a 5 dias."},
        ]
        ctx = format_history_context(history)
        assert "Cliente:" in ctx
        assert "Bot:" in ctx
        assert "Qual o prazo?" in ctx

    def test_multi_turn_follow_up_context(self):
        """A follow-up question 'e o prazo?' should appear after previous context."""
        from chat_history import _filter_and_format, format_history_context

        messages = [
            {"message_type": "incoming", "private": False, "content": "Meu pedido chegou?", "created_at": 1},
            {"message_type": "outgoing", "private": False, "content": "Seu pedido está a caminho.", "created_at": 2, "sender": {"type": "agent"}},
            {"message_type": "incoming", "private": False, "content": "e o prazo?", "created_at": 3},
        ]
        history = _filter_and_format(messages)
        ctx = format_history_context(history)
        assert "e o prazo?" in ctx
        assert "Seu pedido está a caminho." in ctx

    def test_fetch_history_graceful_fallback(self, monkeypatch):
        """fetch_history should return [] when the HTTP call fails."""
        import chat_history
        monkeypatch.setattr(chat_history, "CHAT_HISTORY_ENABLED", True)

        with patch("chat_history.requests.get", side_effect=Exception("connection error")):
            result = chat_history.fetch_history(1, 42)
        assert result == []

    def test_fetch_history_disabled(self, monkeypatch):
        import chat_history
        monkeypatch.setattr(chat_history, "CHAT_HISTORY_ENABLED", False)
        result = chat_history.fetch_history(1, 42)
        assert result == []


# ===========================================================================
# Phase 2a — shopify_mock
# ===========================================================================

class TestShopifyMock:
    """Tests for shopify_mock module."""

    def test_get_order_status_valid(self):
        from shopify_mock import get_order_status
        order = get_order_status("1001")
        assert order is not None
        assert order["number"] == "1001"
        assert order["status"] == "paid"

    def test_get_order_status_with_hash_prefix(self):
        from shopify_mock import get_order_status
        order = get_order_status("#1002")
        assert order is not None
        assert order["tracking"] == "BR123456789BR"

    def test_get_order_status_invalid(self):
        from shopify_mock import get_order_status
        assert get_order_status("9999") is None

    def test_get_order_status_canceled(self):
        from shopify_mock import get_order_status
        order = get_order_status("1003")
        assert order is not None
        assert order["status"] == "canceled"

    def test_search_products_found(self):
        from shopify_mock import search_products
        results = search_products("camiseta")
        assert len(results) > 0
        names = [p["name"].lower() for p in results]
        assert any("camiseta" in n for n in names)

    def test_search_products_no_results(self):
        from shopify_mock import search_products
        results = search_products("xyzabcnotfound")
        assert results == []

    def test_search_products_limit(self):
        from shopify_mock import search_products
        results = search_products("roupa vestuario", limit=2)
        assert len(results) <= 2

    def test_format_order_response(self):
        from shopify_mock import format_order_response, get_order_status
        order = get_order_status("1002")
        assert order is not None
        text = format_order_response(order)
        assert "1002" in text
        assert "BR123456789BR" in text

    def test_format_products_no_results(self):
        from shopify_mock import format_products_response
        text = format_products_response([], "xyz")
        assert "xyz" in text
        assert "Não encontrei" in text

    def test_format_products_with_results(self):
        from shopify_mock import format_products_response, search_products
        products = search_products("camiseta")
        text = format_products_response(products[:3], "camiseta")
        assert "camiseta" in text.lower()


# ===========================================================================
# Phase 2a — shopify_mcp
# ===========================================================================

class TestShopifyMCP:
    """Tests for shopify_mcp module (Storefront MCP client)."""

    def _mock_mcp_response(self, result: Any) -> MagicMock:
        """Return a mock requests.Response with a JSON-RPC 2.0 success body."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"jsonrpc": "2.0", "id": "x", "result": result}
        return mock_resp

    # ------------------------------------------------------------------
    # _mcp_post helper
    # ------------------------------------------------------------------

    def test_mcp_post_returns_result_on_success(self):
        import shopify_mcp
        ok_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        with patch("shopify_mcp.requests.post", return_value=ok_resp):
            result = shopify_mcp._mcp_post("initialize", {})
        assert result == {"protocolVersion": "2024-11-05"}

    def test_mcp_post_returns_none_on_http_error(self):
        import shopify_mcp
        with patch(
            "shopify_mcp.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = shopify_mcp._mcp_post("initialize", {})
        assert result is None

    def test_mcp_post_returns_none_on_jsonrpc_error(self):
        import shopify_mcp
        error_resp = MagicMock()
        error_resp.raise_for_status = MagicMock()
        error_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": "x",
            "error": {"code": -32601, "message": "Method not found"},
        }
        with patch("shopify_mcp.requests.post", return_value=error_resp):
            result = shopify_mcp._mcp_post("unknown_method", {})
        assert result is None

    # ------------------------------------------------------------------
    # _initialize
    # ------------------------------------------------------------------

    def test_initialize_returns_true_on_success(self):
        import shopify_mcp
        ok_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        with patch("shopify_mcp.requests.post", return_value=ok_resp):
            assert shopify_mcp._initialize() is True

    def test_initialize_returns_false_on_failure(self):
        import shopify_mcp
        with patch(
            "shopify_mcp.requests.post",
            side_effect=requests.exceptions.Timeout("timeout"),
        ):
            assert shopify_mcp._initialize() is False

    # ------------------------------------------------------------------
    # search_products
    # ------------------------------------------------------------------

    def _make_search_response(self, products: list[dict]) -> MagicMock:
        """Helper: mock both initialize and tools/call responses."""
        init_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        tool_resp = self._mock_mcp_response(
            {
                "content": [
                    {"type": "text", "text": json.dumps(products)},
                ],
            }
        )
        return [init_resp, tool_resp]

    def test_search_products_returns_normalised_list(self):
        import shopify_mcp
        raw_products = [
            {
                "id": "gid://shopify/Product/1",
                "title": "Camiseta Básica",
                "availableForSale": True,
                "tags": ["vestuario", "camiseta"],
                "onlineStoreUrl": "https://loja.example.com/camiseta",
                "priceRange": {
                    "minVariantPrice": {"amount": "59.90", "currencyCode": "BRL"}
                },
            }
        ]
        side_effects = self._make_search_response(raw_products)
        with patch("shopify_mcp.requests.post", side_effect=side_effects):
            results = shopify_mcp.search_products("camiseta")

        assert len(results) == 1
        assert results[0]["name"] == "Camiseta Básica"
        assert results[0]["available"] is True
        assert "59" in results[0]["price"]

    def test_search_products_returns_empty_when_init_fails(self):
        import shopify_mcp
        with patch(
            "shopify_mcp.requests.post",
            side_effect=requests.exceptions.ConnectionError("down"),
        ):
            results = shopify_mcp.search_products("camiseta")
        assert results == []

    def test_search_products_returns_empty_when_tool_returns_nothing(self):
        import shopify_mcp
        init_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        empty_resp = self._mock_mcp_response({"content": []})
        with patch(
            "shopify_mcp.requests.post", side_effect=[init_resp, empty_resp]
        ):
            results = shopify_mcp.search_products("xyz_not_found")
        assert results == []

    def test_search_products_respects_limit(self):
        import shopify_mcp
        raw_products = [
            {"id": str(i), "title": f"Produto {i}", "availableForSale": True}
            for i in range(10)
        ]
        side_effects = self._make_search_response(raw_products)
        with patch("shopify_mcp.requests.post", side_effect=side_effects):
            results = shopify_mcp.search_products("produto", limit=3)
        assert len(results) <= 3

    def test_search_products_handles_nested_edges(self):
        """Storefront API may wrap products in edges/node GraphQL pattern."""
        import shopify_mcp
        nested_graphql_response = {"products": [{"node": {"id": "p1", "title": "Calça Jeans"}}]}
        init_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        tool_resp = self._mock_mcp_response(
            {"content": [{"type": "text", "text": json.dumps(nested_graphql_response)}]}
        )
        with patch("shopify_mcp.requests.post", side_effect=[init_resp, tool_resp]):
            results = shopify_mcp.search_products("calca")
        assert len(results) == 1
        assert results[0]["name"] == "Calça Jeans"

    def test_search_products_falls_back_to_text_entry_on_non_json(self):
        """If MCP returns plain text, wrap it as a single product entry."""
        import shopify_mcp
        init_resp = self._mock_mcp_response({"protocolVersion": "2024-11-05"})
        tool_resp = self._mock_mcp_response(
            {"content": [{"type": "text", "text": "Tênis disponível em 3 cores"}]}
        )
        with patch("shopify_mcp.requests.post", side_effect=[init_resp, tool_resp]):
            results = shopify_mcp.search_products("tenis")
        assert len(results) == 1
        assert "Tênis" in results[0]["name"]

    # ------------------------------------------------------------------
    # get_order_status
    # ------------------------------------------------------------------

    def test_get_order_status_always_returns_none(self):
        """Storefront MCP does not expose private order data."""
        import shopify_mcp
        assert shopify_mcp.get_order_status("1001") is None
        assert shopify_mcp.get_order_status("9999") is None

    # ------------------------------------------------------------------
    # format_products_response
    # ------------------------------------------------------------------

    def test_format_products_response_empty(self):
        import shopify_mcp
        text = shopify_mcp.format_products_response([], "tênis")
        assert "tênis" in text
        assert "Não encontrei" in text

    def test_format_products_response_with_results(self):
        import shopify_mcp
        products = [
            {
                "id": "p1",
                "name": "Camiseta Básica",
                "price": "BRL 59,90",
                "available": True,
                "category": "vestuario",
                "url": "https://loja.example.com/camiseta",
            }
        ]
        text = shopify_mcp.format_products_response(products, "camiseta")
        assert "Camiseta Básica" in text
        assert "disponível" in text

    def test_format_products_response_unavailable(self):
        import shopify_mcp
        products = [
            {
                "id": "p1",
                "name": "Boné Esgotado",
                "price": "BRL 49,90",
                "available": False,
                "category": "acessorio",
                "url": "",
            }
        ]
        text = shopify_mcp.format_products_response(products, "boné")
        assert "indisponível" in text

    # ------------------------------------------------------------------
    # app.py MCP mode integration
    # ------------------------------------------------------------------

    def test_app_uses_mcp_module_when_mode_is_mcp(self, monkeypatch):
        """When SHOPIFY_MODE=mcp, app._shopify should be shopify_mcp."""
        import shopify_mcp
        import app as bridge_app
        monkeypatch.setattr(bridge_app, "_shopify", shopify_mcp)
        assert bridge_app._shopify is shopify_mcp

    def test_build_prompt_mcp_product_uses_mcp_module(self, monkeypatch):
        """build_prompt should call _shopify.search_products (MCP) in product intent."""
        import app as bridge_app
        import shopify_mcp

        fake_products = [
            {
                "id": "p1",
                "name": "Camiseta MCP",
                "price": "BRL 79,90",
                "available": True,
                "category": "vestuario",
                "url": "https://mcp-store.example.com/camiseta",
            }
        ]

        monkeypatch.setattr(bridge_app, "_shopify", shopify_mcp)
        with patch("shopify_mcp.search_products", return_value=fake_products) as mock_search:
            with patch("app.search_documents", return_value=[]):
                payload = _make_payload(content="Vocês têm camiseta?")
                messages, _trace = bridge_app.build_prompt(payload)

        mock_search.assert_called_once()
        user_msg = messages[1]["content"]
        assert "Camiseta MCP" in user_msg

    def test_build_prompt_mcp_order_shows_not_found(self, monkeypatch):
        """In MCP mode, order lookup returns None → 'Nenhum pedido encontrado'."""
        import app as bridge_app
        import shopify_mcp

        monkeypatch.setattr(bridge_app, "_shopify", shopify_mcp)
        with patch("app.search_documents", return_value=[]):
            payload = _make_payload(content="Qual o status do pedido #1001?")
            messages, _trace = bridge_app.build_prompt(payload)

        user_msg = messages[1]["content"]
        assert "Nenhum pedido encontrado" in user_msg


# ===========================================================================
# Phase 2b — intent detection
# ===========================================================================

class TestIntent:
    """Tests for intent module."""

    def test_detect_order_intent_by_keyword(self):
        from intent import Intent, detect_intent
        assert detect_intent("Qual o status do meu pedido?") == Intent.ORDER

    def test_detect_order_intent_by_number(self):
        from intent import Intent, detect_intent
        assert detect_intent("Pedido #1001") == Intent.ORDER

    def test_detect_product_intent(self):
        from intent import Intent, detect_intent
        assert detect_intent("Vocês têm camiseta branca?") == Intent.PRODUCT

    def test_detect_product_by_price(self):
        from intent import Intent, detect_intent
        assert detect_intent("Qual o preço da calça jeans?") == Intent.PRODUCT

    def test_detect_institutional_returns(self):
        from intent import Intent, detect_intent
        assert detect_intent("Qual a política de devolução?") == Intent.INSTITUTIONAL

    def test_detect_institutional_hours(self):
        from intent import Intent, detect_intent
        assert detect_intent("Qual o horário de atendimento?") == Intent.INSTITUTIONAL

    def test_detect_general_fallback(self):
        from intent import Intent, detect_intent
        assert detect_intent("Oi, tudo bem?") == Intent.GENERAL

    def test_extract_order_number_hash(self):
        from intent import extract_order_number
        assert extract_order_number("Meu pedido é #1002") == "1002"

    def test_extract_order_number_word(self):
        from intent import extract_order_number
        assert extract_order_number("pedido 1001 ainda não chegou") == "1001"

    def test_extract_order_number_none(self):
        from intent import extract_order_number
        assert extract_order_number("oi tudo bem") is None


# ===========================================================================
# Phase 3 — RAG
# ===========================================================================

class TestRAG:
    """Tests for rag module (with RAG_ENABLED=false for unit tests)."""

    def test_chunk_text_short(self):
        from rag import _chunk_text
        text = "Hello world"
        chunks = _chunk_text(text, size=500)
        assert chunks == ["Hello world"]

    def test_chunk_text_long(self):
        from rag import _chunk_text
        text = "A" * 1200
        chunks = _chunk_text(text, size=500, overlap=50)
        assert len(chunks) > 1
        # Each chunk must be non-empty
        assert all(c for c in chunks)

    def test_chunk_text_with_offsets(self):
        from rag import _chunk_text_with_offsets

        text = "Primeiro paragrafo.\n\nSegundo bloco com Recife no final."
        chunks = _chunk_text_with_offsets(text, size=30, overlap=5)

        assert len(chunks) >= 2
        for chunk, start, end in chunks:
            assert chunk == text[start:end].strip()
            assert end > start

    def test_search_documents_disabled(self, monkeypatch):
        import rag
        monkeypatch.setattr(rag, "RAG_ENABLED", False)
        result = rag.search_documents("troca")
        assert result == []

    def test_search_documents_detailed_expands_source_context(self, monkeypatch):
        import rag

        class FakeCollection:
            def count(self):
                return 3

            def query(self, **kwargs):
                return {
                    "documents": [["Recife no final"]],
                    "metadatas": [[{
                        "source": "empresa.md",
                        "source_path": "empresa.md",
                        "doc_id": "doc-1",
                        "entry_type": "chunk",
                        "start_offset": 28,
                        "end_offset": 44,
                    }]],
                    "distances": [[0.2]],
                }

        full_doc = "Loja Sao Paulo na abertura.\n\nLoja Recife: Rua do Bom Jesus, 123."

        monkeypatch.setattr(rag, "RAG_ENABLED", True)
        monkeypatch.setattr(rag, "RAG_RECALL_K", 5)
        monkeypatch.setattr(rag, "RAG_EXPANDED_CONTEXT_CHARS", 200)
        monkeypatch.setattr(rag, "_get_chroma", lambda: FakeCollection())
        monkeypatch.setattr(rag, "_get_embedding", lambda _text: [0.1, 0.2])
        monkeypatch.setattr(rag, "_resolve_source_path", lambda _source, _docs=None: Path("empresa.md"))
        monkeypatch.setattr(rag, "_load_document", lambda _path: full_doc)

        results = rag.search_documents_detailed("qual o endereco da loja recife?")

        assert len(results) == 1
        assert "Rua do Bom Jesus" in results[0].text
        assert results[0].source == "empresa.md"

    def test_search_documents_wrapper_returns_text_only(self, monkeypatch):
        import rag

        fake_results = [
            rag.RAGSearchResult(
                candidate_id="C1",
                text="Trecho expandido",
                source="empresa.md",
                source_path="empresa.md",
                doc_id="doc-1",
                score=1.0,
                distance=0.1,
                start_offset=0,
                end_offset=20,
            )
        ]

        monkeypatch.setattr(rag, "search_documents_detailed", lambda _query, top_k=None, docs_path=None: fake_results)

        assert rag.search_documents("pergunta") == ["Trecho expandido"]

    def test_format_rag_context_empty(self):
        from rag import format_rag_context
        assert format_rag_context([]) == ""

    def test_format_rag_context_nonempty(self):
        from rag import format_rag_context
        chunks = ["Política de troca: 7 dias.", "Garantia: 90 dias."]
        ctx = format_rag_context(chunks)
        assert "Política de troca" in ctx
        assert "Garantia" in ctx

    def test_format_rag_context_respects_max_chars(self, monkeypatch):
        import rag
        monkeypatch.setattr(rag, "RAG_MAX_CONTEXT_CHARS", 20)
        chunks = ["A" * 30, "B" * 30]
        ctx = rag.format_rag_context(chunks)
        # Should have been truncated or empty
        assert len(ctx) <= 20 + len("Informações da empresa:\n")

    def test_search_documents_rag_fallback_no_chroma(self, monkeypatch):
        """search_documents should return [] when chromadb is unavailable."""
        import rag
        monkeypatch.setattr(rag, "RAG_ENABLED", True)
        monkeypatch.setattr(rag, "_chroma_collection", None)
        monkeypatch.setattr(rag, "_chroma_client", None)

        # Simulate ImportError for chromadb
        with patch.dict("sys.modules", {"chromadb": None}):
            result = rag.search_documents("garantia")
        assert result == []

    def test_ingest_documents_disabled(self, monkeypatch):
        import rag
        monkeypatch.setattr(rag, "RAG_ENABLED", False)
        count = rag.ingest_documents()
        assert count == 0

    def test_ingest_documents_missing_path(self, monkeypatch, tmp_path):
        import rag
        monkeypatch.setattr(rag, "RAG_ENABLED", True)
        count = rag.ingest_documents(str(tmp_path / "nonexistent"))
        assert count == 0

    def test_embed_model_check_recovers_after_install(self, monkeypatch):
        import rag

        missing_response = MagicMock()
        missing_response.raise_for_status.return_value = None
        missing_response.json.return_value = {"models": []}

        installed_response = MagicMock()
        installed_response.raise_for_status.return_value = None
        installed_response.json.return_value = {
            "models": [{"name": f"{rag.RAG_EMBED_MODEL}:latest"}]
        }

        monkeypatch.setattr(rag, "_embed_model_checked", False)
        monkeypatch.setattr(rag, "_embed_model_available", False)
        monkeypatch.setattr(rag, "_embedding_disabled_reason", None)
        monkeypatch.setattr(rag, "_embeddings_disabled_until", 0.0)

        with patch("rag.requests.get", return_value=missing_response) as get_tags:
            assert rag._check_embed_model_available() is False
            assert get_tags.call_count == 1

        assert rag._embedding_disabled_reason is not None

        monkeypatch.setattr(rag, "_embeddings_disabled_until", 0.0)
        with patch("rag.requests.get", return_value=installed_response) as get_tags:
            assert rag._check_embed_model_available() is True
            assert get_tags.call_count == 1

        assert rag._embedding_disabled_reason is None


# ===========================================================================
# Phase 4 — app orchestration (Flask endpoints + build_prompt)
# ===========================================================================

class TestAppEndpoints:
    """Tests for Flask endpoints (healthz and webhook)."""

    @pytest.fixture
    def client(self):
        import app as bridge_app
        bridge_app.app.config["TESTING"] = True
        with bridge_app.app.test_client() as c:
            yield c

    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert b"ok" in resp.data

    def test_webhook_ignored_non_message_created(self, client):
        payload = _make_payload()
        payload["event"] = "conversation_created"
        resp = client.post("/webhook", json=payload)
        data = json.loads(resp.data)
        assert data["status"] == "ignored"
        assert data["reason"] == "unsupported_event"

    def test_webhook_ignored_outgoing(self, client):
        payload = _make_payload(msg_type="outgoing")
        resp = client.post("/webhook", json=payload)
        data = json.loads(resp.data)
        assert data["status"] == "ignored"

    def test_webhook_ignored_agent(self, client):
        payload = _make_payload(sender_type="user")
        resp = client.post("/webhook", json=payload)
        data = json.loads(resp.data)
        assert data["status"] == "ignored"
        assert data["reason"] == "agent_message"

    def test_webhook_ignored_resolved(self, client):
        payload = _make_payload(conversation_status="resolved")
        resp = client.post("/webhook", json=payload)
        data = json.loads(resp.data)
        assert data["status"] == "ignored"

    def test_webhook_accepted(self, client):
        payload = _make_payload(content="Olá, preciso de ajuda")
        resp = client.post("/webhook", json=payload,
                           headers={"X-Chatwoot-Delivery": "unique-delivery-1"})
        assert resp.status_code == 202
        data = json.loads(resp.data)
        assert data["status"] == "accepted"

    def test_webhook_duplicate_rejected(self, client):
        payload = _make_payload(content="Olá")
        headers = {"X-Chatwoot-Delivery": "dup-delivery-999"}
        client.post("/webhook", json=payload, headers=headers)
        resp = client.post("/webhook", json=payload, headers=headers)
        data = json.loads(resp.data)
        assert data["status"] == "duplicate"

    def test_webhook_invalid_signature(self, client, monkeypatch):
        import app as bridge_app
        monkeypatch.setattr(bridge_app, "CHATWOOT_WEBHOOK_SECRET", "secret123")
        payload = _make_payload()
        resp = client.post(
            "/webhook",
            json=payload,
            headers={
                "X-Chatwoot-Timestamp": str(int(time.time())),
                "X-Chatwoot-Signature": "sha256=invalidsignature",
            },
        )
        assert resp.status_code == 401

    def test_webhook_valid_signature(self, client, monkeypatch):
        import app as bridge_app
        secret = "testsecret"
        monkeypatch.setattr(bridge_app, "CHATWOOT_WEBHOOK_SECRET", secret)
        ts = str(int(time.time()))
        body = json.dumps(_make_payload()).encode()
        message = f"{ts}.".encode() + body
        sig = "sha256=" + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
        resp = client.post(
            "/webhook",
            data=body,
            content_type="application/json",
            headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": sig,
                     "X-Chatwoot-Delivery": "sig-test-1"},
        )
        assert resp.status_code == 202


class TestBuildPrompt:
    """Tests for the build_prompt orchestration logic."""

    def _make_prompt(self, content: str, **kwargs) -> tuple[list[dict], dict]:
        import app as bridge_app
        payload = _make_payload(content=content, **kwargs)
        return bridge_app.build_prompt(payload)

    def test_returns_system_and_user(self):
        messages, _trace = self._make_prompt("Oi")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_order_intent_with_valid_number(self):
        with patch("app.search_documents", return_value=[]):
            messages, _trace = self._make_prompt("Qual o status do pedido #1001?")
        user_msg = messages[1]["content"]
        assert "Pedido #1001" in user_msg
        assert "Pagamento confirmado" in user_msg

    def test_order_intent_with_invalid_number(self):
        with patch("app.search_documents", return_value=[]):
            messages, _trace = self._make_prompt("Meu pedido é #9999")
        user_msg = messages[1]["content"]
        assert "9999" in user_msg
        assert "Nenhum pedido encontrado" in user_msg

    def test_order_intent_without_number_asks_for_it(self):
        with patch("app.search_documents", return_value=[]):
            messages, _trace = self._make_prompt("Quero saber sobre meu pedido")
        user_msg = messages[1]["content"]
        assert "número do pedido" in user_msg

    def test_product_intent_returns_results(self):
        with patch("app.search_documents", return_value=[]):
            messages, _trace = self._make_prompt("Vocês têm camiseta?")
        user_msg = messages[1]["content"]
        assert "camiseta" in user_msg.lower()

    def test_general_intent_includes_rag_tool_context(self):
        with patch("app.search_documents", return_value=["Loja Altero Recife"]):
            messages, _trace = self._make_prompt("Tem loja em Recife?")
        user_msg = messages[1]["content"]
        assert "Dados disponíveis" in user_msg
        assert "Loja Altero Recife" in user_msg

    def test_general_intent_no_match_adds_fallback_instruction(self):
        with patch("app.search_documents", return_value=[]):
            messages, _trace = self._make_prompt("Tudo bem?")
        user_msg = messages[1]["content"]
        assert "não encontrou no material disponível" in user_msg.lower()

    def test_prompt_contains_channel_and_contact(self):
        messages, _trace = self._make_prompt("Oi")
        user_msg = messages[1]["content"]
        assert "Canal:" in user_msg
        assert "Fulano" in user_msg

    def test_history_included_when_available(self):
        """History is injected into the user message when present."""
        import app as bridge_app

        fake_history = [
            {"role": "Cliente", "content": "Qual o prazo?"},
            {"role": "Bot", "content": "3 dias úteis."},
        ]
        with patch("app.chat_history.fetch_history", return_value=fake_history):
            payload = _make_payload(content="e para o interior?")
            messages, _trace = bridge_app.build_prompt(payload)

        user_msg = messages[1]["content"]
        assert "Qual o prazo?" in user_msg
        assert "3 dias úteis." in user_msg

    def test_history_failure_does_not_crash(self):
        """If history fetch raises, prompt is still built without history."""
        import app as bridge_app

        with patch("app.chat_history.fetch_history", side_effect=Exception("timeout")):
            payload = _make_payload(content="e o prazo?")
            messages, _trace = bridge_app.build_prompt(payload)  # must not raise

        assert len(messages) == 2

    def test_rag_failure_does_not_crash(self, monkeypatch):
        """If RAG raises, prompt is still built without RAG context."""
        import app as bridge_app

        with patch("app.search_documents", side_effect=Exception("rag down")):
            payload = _make_payload(content="Qual a política de troca?")
            messages, _trace = bridge_app.build_prompt(payload)  # must not raise

        assert len(messages) == 2

    def test_rag_runs_for_general_order_and_product(self):
        import app as bridge_app

        with patch("app.search_documents", return_value=[] ) as mock_search:
            bridge_app.build_prompt(_make_payload(content="Oi"))
            bridge_app.build_prompt(_make_payload(content="Pedido #1001"))
            bridge_app.build_prompt(_make_payload(content="Vocês têm camiseta?"))

        assert mock_search.call_count == 3

    def test_order_prompt_combines_shopify_and_rag_context(self):
        with patch("app.search_documents", return_value=["Loja mais próxima: Recife"]):
            messages, _trace = self._make_prompt("Qual o status do pedido #1001?")

        user_msg = messages[1]["content"]
        assert "Pedido #1001" in user_msg
        assert "Loja mais próxima: Recife" in user_msg

    def test_product_prompt_combines_shopify_and_rag_context(self):
        with patch("app.search_documents", return_value=["Atendimento presencial disponível"]):
            messages, _trace = self._make_prompt("Vocês têm camiseta?")

        user_msg = messages[1]["content"]
        assert "camiseta" in user_msg.lower()
        assert "Atendimento presencial disponível" in user_msg


class TestGenerateReplyFallback:
    """Tests for model fallback behaviour."""

    def _payload(self) -> dict:
        return _make_payload(content="Oi")

    def test_primary_model_success(self):
        import app as bridge_app

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Olá!"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.requests.post", return_value=mock_resp):
            reply = bridge_app.generate_reply(self._payload())
        assert reply == "Olá!"

    def test_fallback_used_when_primary_fails(self):
        import app as bridge_app

        call_count = 0

        def fake_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.Timeout("timeout")
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"message": {"content": "Resposta do fallback"}}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("app.requests.post", side_effect=fake_post):
            reply = bridge_app.generate_reply(self._payload())
        assert reply == "Resposta do fallback"
        assert call_count == 2

    def test_both_models_fail_returns_fail_soft_message(self):
        import app as bridge_app

        with patch("app.requests.post", side_effect=Exception("down")):
            reply = bridge_app.generate_reply(self._payload())
        assert reply == bridge_app.OLLAMA_UNAVAILABLE_MESSAGE


