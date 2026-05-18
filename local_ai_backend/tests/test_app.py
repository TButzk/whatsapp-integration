import hashlib
import hmac
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient

from local_ai_backend.config import Settings
from local_ai_backend.integrations.shopify import IntegrationResult
from local_ai_backend.integrations.shopify import ShopifyIntegration
from local_ai_backend.intent import Intent, classify_intent
from local_ai_backend.llm.ollama import OllamaClient
from local_ai_backend.main import create_app
from local_ai_backend.observability.sanitization import sanitize_text
from local_ai_backend.policies import DecisionAction, decide_next_action
from local_ai_backend.queueing import InMemoryMessagePublisher
from local_ai_backend.schemas import ChatwootContact
from local_ai_backend.services.customer_identity import extract_customer_identity


def _build_client() -> TestClient:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
    )
    publisher = InMemoryMessagePublisher()
    return TestClient(create_app(settings, publisher=publisher))


def test_health_endpoint() -> None:
    client = _build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "test-backend"}


def test_ready_endpoint_lists_dependencies() -> None:
    client = _build_client()

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["dependencies"]["redis"]["configured"] is True
    assert payload["dependencies"]["postgres"]["configured"] is True
    assert payload["dependencies"]["ollama"]["configured"] is True
    assert payload["dependencies"]["shopify_customer_lookup"]["configured"] is True
    assert payload["dependencies"]["shopify_customer_lookup"]["detail"] == "disabled"


def test_ready_endpoint_marks_shopify_enabled_when_token_present_even_without_flag() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
        SHOPIFY_CUSTOMER_LOOKUP_ENABLED=False,
        SHOPIFY_CUSTOMER_MCP_ENDPOINT="https://shopify.com/1/account/customer/api/mcp",
        SHOPIFY_CUSTOMER_MCP_TOKEN="token",
    )
    client = TestClient(create_app(settings, publisher=InMemoryMessagePublisher()))

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dependencies"]["shopify_customer_lookup"]["configured"] is True
    assert payload["dependencies"]["shopify_customer_lookup"]["detail"] == "enabled:token"


def test_shopify_customer_oauth_authorize_url_endpoint() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.shopify_integration.create_customer_oauth_authorization = Mock(
        return_value=IntegrationResult(
            status="ok",
            data={
                "authorization_url": "https://shopify.com/authentication/1/oauth/authorize?x=1",
                "state": "abc",
                "redirect_uri": "http://localhost:8000/callback",
                "code_verifier": "verifier",
                "uses_pkce": True,
            },
        )
    )
    client = TestClient(app)

    response = client.get("/admin/shopify/customer-oauth/authorize-url")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "abc"
    assert payload["uses_pkce"] is True
    assert payload["authorization_url"].startswith("https://")


def test_shopify_customer_oauth_exchange_code_endpoint() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.shopify_integration.exchange_customer_oauth_code = Mock(
        return_value=IntegrationResult(
            status="ok",
            data={
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "openid",
            },
        )
    )
    client = TestClient(app)

    response = client.post(
        "/admin/shopify/customer-oauth/exchange-code",
        json={
            "code": "auth-code",
            "redirect_uri": "http://localhost:8000/callback",
            "code_verifier": "verifier",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["token_type"] == "Bearer"
    assert payload["access_token"] == "access"


def test_extract_customer_identity_prefers_chatwoot_contact() -> None:
    contact = ChatwootContact(
        email="Cliente@Loja.Com",
        phone_number="+55 (11) 98888-7777",
    )
    text = (
        "pedido 12345 email outro@dominio.com "
        "telefone 11 99999-0000 cpf 123.456.789-00"
    )

    identity = extract_customer_identity(text, contact)

    assert identity.order_number == "12345"
    assert identity.email == "cliente@loja.com"
    assert identity.phone == "+5511988887777"
    assert identity.cpf == "12345678900"


def test_chatwoot_webhook_accepts_message() -> None:
    client = _build_client()

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "content": "Oi",
            "message_type": "incoming",
            "private": False,
            "conversation": {"id": 42, "status": "open", "channel": "whatsapp"},
            "contact": {"id": 1, "name": "Maria"},
        },
    )

    assert response.status_code == 202
    assert response.json()["channel"] == "chatwoot"
    assert response.json()["message_id"] == "42"


def test_chatwoot_compat_webhook_accepts_message() -> None:
    client = _build_client()

    response = client.post(
        "/webhook",
        json={
            "event": "message_created",
            "content": "Oi",
            "message_type": "incoming",
            "private": False,
            "conversation": {"id": 43, "status": "open", "channel": "whatsapp"},
            "contact": {"id": 2, "name": "Joao"},
        },
    )

    assert response.status_code == 202
    assert response.json()["channel"] == "chatwoot"
    assert response.json()["message_id"] == "43"


def test_chatwoot_auto_reply_compat_webhook_accepts_message() -> None:
    client = _build_client()

    response = client.post(
        "/auto-reply/webhook",
        json={
            "event": "message_created",
            "content": "Oi",
            "message_type": "incoming",
            "private": False,
            "conversation": {"id": 44, "status": "open", "channel": "whatsapp"},
            "contact": {"id": 3, "name": "Ana"},
        },
    )

    assert response.status_code == 202
    assert response.json()["channel"] == "chatwoot"
    assert response.json()["message_id"] == "44"


def test_chatwoot_webhook_posts_reply_when_configured() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.llm_client.chat = Mock(
        return_value=type("_Resp", (), {"content": "Resposta automática"})()
    )
    client = TestClient(app)

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "account": {"id": 7},
            "content": "Oi",
            "message_type": "incoming",
            "private": False,
            "conversation": {"id": 45, "status": "open", "channel": "whatsapp"},
        },
    )

    assert response.status_code == 202
    app.state.chatwoot_client.send_conversation_reply.assert_called_once()
    call_kw = app.state.chatwoot_client.send_conversation_reply.call_args.kwargs
    assert call_kw["account_id"] == 7
    assert call_kw["conversation_id"] == 45
    assert len(call_kw["content"]) > 0


def test_chatwoot_webhook_skips_reply_for_outgoing_message() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.llm_client.chat = Mock(
        return_value=type("_Resp", (), {"content": "Resposta automática"})()
    )
    client = TestClient(app)

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "account": {"id": 7},
            "content": "Nao responder",
            "message_type": "outgoing",
            "private": False,
            "conversation": {"id": 46, "status": "open", "channel": "whatsapp"},
        },
    )

    assert response.status_code == 202
    app.state.chatwoot_client.send_conversation_reply.assert_not_called()


def test_whatsapp_webhook_accepts_message() -> None:
    client = _build_client()

    response = client.post(
        "/webhook/whatsapp",
        json={
            "message_id": "wamid.123",
            "from_number": "+5511999999999",
            "content": "Quero saber o status do meu pedido",
        },
    )

    assert response.status_code == 202
    assert response.json()["channel"] == "whatsapp"
    assert response.json()["message_id"] == "wamid.123"


def test_chatwoot_webhook_rejects_invalid_signature() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        CHATWOOT_WEBHOOK_SECRET="top-secret",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
    )
    client = TestClient(create_app(settings, publisher=InMemoryMessagePublisher()))

    response = client.post(
        "/webhook/chatwoot",
        headers={"X-Chatwoot-Signature": "bad-signature"},
        json={
            "event": "message_created",
            "content": "Oi",
            "message_type": "incoming",
            "private": False,
            "conversation": {"id": 42, "status": "open", "channel": "whatsapp"},
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_signature"


def test_chatwoot_webhook_accepts_valid_signature() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        CHATWOOT_WEBHOOK_SECRET="top-secret",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://haproxy:11434",
    )
    client = TestClient(create_app(settings, publisher=InMemoryMessagePublisher()))
    payload = {
        "event": "message_created",
        "content": "Oi",
        "message_type": "incoming",
        "private": False,
        "conversation": {"id": 42, "status": "open", "channel": "whatsapp"},
    }
    body = __import__("json").dumps(payload).encode("utf-8")
    signature = hmac.new(b"top-secret", body, hashlib.sha256).hexdigest()

    response = client.post(
        "/webhook/chatwoot",
        headers={
            "X-Chatwoot-Signature": signature,
            "Content-Type": "application/json",
        },
        content=body,
    )

    assert response.status_code == 202


def test_internal_process_message_returns_policy_decision() -> None:
    client = _build_client()

    response = client.post(
        "/internal/process-message",
        json={
            "channel": "whatsapp",
            "external_message_id": "wamid.123",
            "conversation_id": "conv-1",
            "content": "Quero rastrear meu pedido",
            "metadata": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "tracking"
    assert payload["action"] == "ask_for_more_info"


def test_intent_classifies_order_status() -> None:
    result = classify_intent("Qual o status do pedido #1234?")

    assert result.intent == Intent.PEDIDO_STATUS
    assert result.order_number == "1234"


def test_sanitize_text_masks_sensitive_data() -> None:
    text = "Pedido #12345 do CPF 123.456.789-00 email maria@loja.com telefone +55 11 99999-8888"

    sanitized = sanitize_text(text)

    assert "#***" in sanitized
    assert "***CPF***" in sanitized
    assert "***EMAIL***" in sanitized
    assert "***FONE***" in sanitized


def test_policy_handoff_on_low_confidence() -> None:
    decision = decide_next_action(classify_intent("blabla sem contexto"))

    assert decision.action == DecisionAction.HANDOFF_HUMAN
    assert decision.reason == "low_confidence"


def test_policy_asks_for_order_identifier() -> None:
    decision = decide_next_action(classify_intent("quero rastrear meu pedido"))

    assert decision.action == DecisionAction.ASK_FOR_MORE_INFO
    assert decision.reason == "missing_order_identifier"


def test_shopify_integration_not_configured() -> None:
    settings = Settings()
    integration = ShopifyIntegration(settings)

    result = integration.get_order_status("1234")

    assert result.status == "integration_not_configured"
    assert result.data is None


def test_shopify_get_order_status_formats_structured_summary() -> None:
    settings = Settings(
        SHOPIFY_CUSTOMER_LOOKUP_ENABLED=True,
        SHOPIFY_CUSTOMER_MCP_ENDPOINT="https://example.test/mcp",
        SHOPIFY_CUSTOMER_MCP_TOKEN="token",
    )
    integration = ShopifyIntegration(settings)
    integration._call_first_available_customer_tool = Mock(  # type: ignore[attr-defined]
        return_value=(
            '{"number":"1234","status":"fulfilled","tracking":"BR123",'
            '"tracking_url":"https://track.test/BR123","created_at":"2026-05-10"}'
        )
    )

    result = integration.get_order_status("1234")

    assert result.status == "ok"
    assert result.data is not None
    assert "Pedido: #1234" in result.data["summary"]
    assert "Status: fulfilled" in result.data["summary"]
    assert "Rastreio: BR123" in result.data["summary"]


def test_shopify_purchase_summary_returns_ambiguous_for_multiple_orders() -> None:
    settings = Settings(
        SHOPIFY_CUSTOMER_LOOKUP_ENABLED=True,
        SHOPIFY_CUSTOMER_MCP_ENDPOINT="https://example.test/mcp",
        SHOPIFY_CUSTOMER_MCP_TOKEN="token",
    )
    integration = ShopifyIntegration(settings)
    integration._call_first_available_customer_tool = Mock(  # type: ignore[attr-defined]
        return_value=(
            '[{"number":"1001","status":"open","created_at":"2026-05-01"},'
            '{"number":"1002","status":"fulfilled","created_at":"2026-05-02"}]'
        )
    )

    result = integration.get_customer_purchase_summary(
        "status do pedido",
        email="cliente@dominio.com",
    )

    assert result.status == "ambiguous"
    assert result.data is not None
    assert result.data["count"] == 2
    assert "#1001" in result.data["summary"]
    assert "#1002" in result.data["summary"]


def test_ollama_client_uses_chat_endpoint() -> None:
    settings = Settings(OLLAMA_BASE_URL="http://ollama.internal")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = __import__("json").loads(request.content.decode("utf-8"))
        assert payload["model"] == settings.ollama_model_fast
        return httpx.Response(
            status_code=200,
            json={"message": {"content": "Resposta curta"}},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, timeout=5.0)
    client = OllamaClient(settings=settings, http_client=http_client)

    response = client.chat(
        system_prompt="Responda em portugues do Brasil.",
        user_message="Oi",
    )

    assert response.model == settings.ollama_model_fast
    assert response.content == "Resposta curta"


# ---------------------------------------------------------------------------
# Conversation state – unit tests
# ---------------------------------------------------------------------------


def test_conversation_state_store_get_creates_new_state() -> None:
    from local_ai_backend.services.conversation_state import ConversationStateStore

    store = ConversationStateStore()
    state = store.get("conv-99")

    assert state.conversation_id == "conv-99"
    assert state.awaiting_identifier is False
    assert state.pending_intent is None


def test_conversation_state_store_save_and_retrieve() -> None:
    from local_ai_backend.services.conversation_state import ConversationStateStore

    store = ConversationStateStore()
    state = store.get("conv-100")
    state.awaiting_identifier = True
    state.pending_intent = "pedido_status"
    store.save(state)

    loaded = store.get("conv-100")
    assert loaded.awaiting_identifier is True
    assert loaded.pending_intent == "pedido_status"


def test_conversation_state_store_clear() -> None:
    from local_ai_backend.services.conversation_state import ConversationStateStore

    store = ConversationStateStore()
    state = store.get("conv-101")
    state.awaiting_identifier = True
    store.save(state)
    store.clear("conv-101")

    fresh = store.get("conv-101")
    assert fresh.awaiting_identifier is False


def test_conversation_state_expires_after_ttl() -> None:
    import time
    from local_ai_backend.services.conversation_state import ConversationState, ConversationStateStore

    store = ConversationStateStore(ttl_seconds=0.01)
    state = store.get("conv-102")
    state.awaiting_identifier = True
    state.pending_intent = "pedido_status"
    store.save(state)

    time.sleep(0.05)

    fresh = store.get("conv-102")
    assert fresh.awaiting_identifier is False
    assert fresh.pending_intent is None


# ---------------------------------------------------------------------------
# process_message with context – unit tests
# ---------------------------------------------------------------------------


def test_process_message_restores_intent_when_awaiting_identifier_with_cpf() -> None:
    from local_ai_backend.processing import process_message
    from local_ai_backend.policies import DecisionAction
    from local_ai_backend.services.conversation_state import ConversationState

    state = ConversationState(
        conversation_id="conv-200",
        awaiting_identifier=True,
        pending_intent="pedido_status",
    )

    result = process_message("123.456.789-00", state=state)

    # CPF alone should be treated as continuation of pedido_status, not handoff
    assert result.intent.value == "pedido_status"
    assert result.action != DecisionAction.HANDOFF_HUMAN


def test_process_message_without_context_cpf_alone_is_unknown() -> None:
    from local_ai_backend.processing import process_message
    from local_ai_backend.policies import DecisionAction

    result = process_message("123.456.789-00", state=None)

    assert result.action == DecisionAction.HANDOFF_HUMAN


def test_process_message_sets_awaiting_identifier_after_asking() -> None:
    from local_ai_backend.processing import process_message
    from local_ai_backend.services.conversation_state import ConversationState

    state = ConversationState(conversation_id="conv-201")
    process_message("qual o status do meu pedido", state=state)

    assert state.awaiting_identifier is True
    assert state.pending_intent == "pedido_status"


def test_process_message_clears_awaiting_after_identifier_received() -> None:
    from local_ai_backend.processing import process_message
    from local_ai_backend.services.conversation_state import ConversationState

    state = ConversationState(
        conversation_id="conv-202",
        awaiting_identifier=True,
        pending_intent="pedido_status",
    )
    process_message("123.456.789-00", state=state)

    # After identifier received the bot should no longer be waiting
    assert state.awaiting_identifier is False
    assert state.pending_intent is None


# ---------------------------------------------------------------------------
# Multi-turn webhook integration test
# ---------------------------------------------------------------------------


def test_multi_turn_webhook_remembers_context() -> None:
    """Turn 1: user asks for order status → bot asks for identifier.
    Turn 2: user sends CPF → bot should NOT handoff to human."""

    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    # LLM mock – return generic reply so we can focus on the policy path
    app.state.llm_client.chat = Mock(
        return_value=type("_R", (), {"content": "Processando seu pedido"})()
    )
    client = TestClient(app)

    base_payload = {
        "event": "message_created",
        "account": {"id": 1},
        "message_type": "incoming",
        "private": False,
        "conversation": {"id": 555, "status": "open", "channel": "whatsapp"},
    }

    # Pre-seed: conversation already past the greeting, in the main menu
    state = app.state.conv_state_store.get("555")
    state.flow_state = "MAIN_MENU"
    app.state.conv_state_store.save(state)

    # Turn 1 – select option "1" (status do pedido) from the menu
    client.post("/webhook/chatwoot", json={**base_payload, "content": "1"})

    # Bot should have asked for CPF / order number
    first_call_args = app.state.chatwoot_client.send_conversation_reply.call_args_list[0]
    first_reply: str = first_call_args.kwargs.get("content", first_call_args.args[2] if len(first_call_args.args) > 2 else "")
    assert "pedido" in first_reply.lower() or "cpf" in first_reply.lower()

    # Turn 2 – user sends CPF alone (no order context in the message itself)
    app.state.chatwoot_client.send_conversation_reply.reset_mock()
    client.post("/webhook/chatwoot", json={**base_payload, "content": "123.456.789-00"})

    # Should have replied (not skipped) – reply was sent
    assert app.state.chatwoot_client.send_conversation_reply.called


def test_order_flow_uses_chatwoot_contact_identifiers_for_shopify_lookup() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.llm_client.chat = Mock(
        return_value=type("_R", (), {"content": "Processando seu pedido"})()
    )
    app.state.shopify_integration.get_customer_purchase_summary = Mock(
        return_value=IntegrationResult(
            status="ok",
            data={"summary": "Pedido #12345 - Em separacao"},
        )
    )

    client = TestClient(app)
    state = app.state.conv_state_store.get("556")
    state.flow_state = "PEDIDO_STATUS"
    app.state.conv_state_store.save(state)

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "account": {"id": 1},
            "message_type": "incoming",
            "private": False,
            "content": "meu cpf 123.456.789-00 e email msg@dominio.com",
            "conversation": {"id": 556, "status": "open", "channel": "whatsapp"},
            "contact": {
                "email": "contato@cliente.com",
                "phone_number": "+55 (11) 97777-2222",
            },
        },
    )

    assert response.status_code == 202
    app.state.shopify_integration.get_customer_purchase_summary.assert_called_once()
    call_kw = app.state.shopify_integration.get_customer_purchase_summary.call_args.kwargs
    assert call_kw["email"] == "contato@cliente.com"
    assert call_kw["phone"] == "+5511977772222"
    assert call_kw["cpf"] == "12345678900"


def test_order_flow_ambiguous_match_asks_confirmation() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.shopify_integration.get_customer_purchase_summary = Mock(
        return_value=IntegrationResult(
            status="ambiguous",
            data={"summary": "Encontrei mais de um pedido com esses dados:\n1. #1001\n2. #1002"},
        )
    )

    client = TestClient(app)
    state = app.state.conv_state_store.get("557")
    state.flow_state = "PEDIDO_STATUS"
    app.state.conv_state_store.save(state)

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "account": {"id": 1},
            "message_type": "incoming",
            "private": False,
            "content": "meu email e cliente@dominio.com",
            "conversation": {"id": 557, "status": "open", "channel": "whatsapp"},
            "contact": {"email": "cliente@dominio.com"},
        },
    )

    assert response.status_code == 202
    sent_reply = app.state.chatwoot_client.send_conversation_reply.call_args.kwargs["content"]
    assert "mais de um pedido" in sent_reply.lower()
    assert "informe o numero do pedido" in sent_reply.lower()


def test_order_flow_not_found_asks_identifier_again() -> None:
    settings = Settings(
        APP_NAME="test-backend",
        REDIS_URL="redis://redis:6379/0",
        QUEUE_NAME="test_queue",
        DATABASE_URL="postgresql+psycopg://postgres:postgres@db:5432/app",
        OLLAMA_BASE_URL="http://ollama.internal",
        CHATWOOT_API_TOKEN="token",
        CHATWOOT_BASE_URL="http://chatwoot.local",
    )
    app = create_app(settings, publisher=InMemoryMessagePublisher())
    app.state.chatwoot_client.can_send_reply = Mock(return_value=True)
    app.state.chatwoot_client.send_conversation_reply = Mock(return_value={"id": 1})
    app.state.shopify_integration.get_order_status = Mock(
        return_value=IntegrationResult(status="not_found", data=None)
    )

    client = TestClient(app)
    state = app.state.conv_state_store.get("558")
    state.flow_state = "PEDIDO_STATUS"
    app.state.conv_state_store.save(state)

    response = client.post(
        "/webhook/chatwoot",
        json={
            "event": "message_created",
            "account": {"id": 1},
            "message_type": "incoming",
            "private": False,
            "content": "pedido #999999",
            "conversation": {"id": 558, "status": "open", "channel": "whatsapp"},
        },
    )

    assert response.status_code == 202
    sent_reply = app.state.chatwoot_client.send_conversation_reply.call_args.kwargs["content"]
    assert "nao localizei pedido" in sent_reply.lower()
    assert "numero do pedido" in sent_reply.lower()