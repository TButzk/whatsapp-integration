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

    def test_search_documents_disabled(self, monkeypatch):
        import rag
        monkeypatch.setattr(rag, "RAG_ENABLED", False)
        result = rag.search_documents("troca")
        assert result == []

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

    def _make_prompt(self, content: str, **kwargs) -> list[dict]:
        import app as bridge_app
        payload = _make_payload(content=content, **kwargs)
        return bridge_app.build_prompt(payload)

    def test_returns_system_and_user(self):
        messages = self._make_prompt("Oi")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_order_intent_with_valid_number(self):
        messages = self._make_prompt("Qual o status do pedido #1001?")
        user_msg = messages[1]["content"]
        assert "Pedido #1001" in user_msg
        assert "Pagamento confirmado" in user_msg

    def test_order_intent_with_invalid_number(self):
        messages = self._make_prompt("Meu pedido é #9999")
        user_msg = messages[1]["content"]
        assert "9999" in user_msg
        assert "Nenhum pedido encontrado" in user_msg

    def test_order_intent_without_number_asks_for_it(self):
        messages = self._make_prompt("Quero saber sobre meu pedido")
        user_msg = messages[1]["content"]
        assert "número do pedido" in user_msg

    def test_product_intent_returns_results(self):
        messages = self._make_prompt("Vocês têm camiseta?")
        user_msg = messages[1]["content"]
        assert "camiseta" in user_msg.lower()

    def test_general_intent_no_tool_context(self):
        messages = self._make_prompt("Tudo bem?")
        user_msg = messages[1]["content"]
        assert "Dados disponíveis" not in user_msg

    def test_prompt_contains_channel_and_contact(self):
        messages = self._make_prompt("Oi")
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
            messages = bridge_app.build_prompt(payload)

        user_msg = messages[1]["content"]
        assert "Qual o prazo?" in user_msg
        assert "3 dias úteis." in user_msg

    def test_history_failure_does_not_crash(self):
        """If history fetch raises, prompt is still built without history."""
        import app as bridge_app

        with patch("app.chat_history.fetch_history", side_effect=Exception("timeout")):
            payload = _make_payload(content="e o prazo?")
            messages = bridge_app.build_prompt(payload)  # must not raise

        assert len(messages) == 2

    def test_rag_failure_does_not_crash(self, monkeypatch):
        """If RAG raises, prompt is still built without RAG context."""
        import app as bridge_app

        with patch("app.search_documents", side_effect=Exception("rag down")):
            payload = _make_payload(content="Qual a política de troca?")
            messages = bridge_app.build_prompt(payload)  # must not raise

        assert len(messages) == 2


class TestGenerateReplyFallback:
    """Tests for model fallback behaviour."""

    def _payload(self) -> dict:
        return _make_payload(content="Oi")

    def test_primary_model_success(self):
        import app as bridge_app

        mock_resp = MagicMock()
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
            mock_resp.json.return_value = {"message": {"content": "Resposta do fallback"}}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("app.requests.post", side_effect=fake_post):
            reply = bridge_app.generate_reply(self._payload())
        assert reply == "Resposta do fallback"
        assert call_count == 2

    def test_both_models_fail_raises(self):
        import app as bridge_app

        with patch("app.requests.post", side_effect=Exception("down")):
            with pytest.raises(Exception):
                bridge_app.generate_reply(self._payload())


