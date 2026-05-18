"""Unit tests for the menu state machine (local_ai_backend/menu.py).

These tests call handle_menu_message directly — no HTTP layer involved.
Each test pre-seeds a ConversationState with a specific flow_state and
asserts the returned MenuResult and the updated state.
"""

from local_ai_backend.menu import (
    BACK_HINT,
    GREETING_TEXT,
    MAIN_MENU_TEXT,
    MenuResult,
    handle_menu_message,
)
from local_ai_backend.services.conversation_state import ConversationState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_state(flow: str, conv_id: str = "test-1") -> ConversationState:
    s = ConversationState(conversation_id=conv_id)
    s.flow_state = flow
    return s


# ---------------------------------------------------------------------------
# START state
# ---------------------------------------------------------------------------


def test_start_any_message_shows_greeting() -> None:
    state = _make_state("START")
    result = handle_menu_message("Oi", state)

    assert result.action == "reply"
    assert GREETING_TEXT in result.reply
    assert "1 - Status do pedido" in result.reply
    assert state.flow_state == "MAIN_MENU"


def test_start_empty_message_shows_greeting() -> None:
    state = _make_state("START")
    result = handle_menu_message("", state)

    assert result.action == "reply"
    assert GREETING_TEXT in result.reply
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# MAIN_MENU state
# ---------------------------------------------------------------------------


def test_main_menu_number_1_enters_pedido_status() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("1", state)

    assert result.action == "reply"
    assert state.flow_state == "PEDIDO_STATUS"
    assert "pedido" in result.reply.lower() or "cpf" in result.reply.lower()


def test_main_menu_number_6_triggers_rag_for_frete() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("6", state)

    assert result.action == "query_rag"
    assert result.rag_query_override is not None
    assert "frete" in result.rag_query_override.lower() or "entrega" in result.rag_query_override.lower()
    assert state.flow_state == "MAIN_MENU"


def test_main_menu_number_9_triggers_handoff() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("9", state)

    assert result.action == "handoff"
    assert state.flow_state == "MAIN_MENU"


def test_main_menu_greeting_shows_menu() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("bom dia", state)

    assert result.action == "reply"
    assert "1 - Status" in result.reply
    assert state.flow_state == "MAIN_MENU"


def test_main_menu_free_text_with_detectable_intent_enters_flow() -> None:
    state = _make_state("MAIN_MENU")
    # "rastreio" → TRACKING intent, confidence 0.95
    result = handle_menu_message("quero rastrear meu pedido", state)

    assert result.action == "reply"
    assert state.flow_state == "TRACKING"


def test_main_menu_unrecognized_text_reshows_menu() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("blablabla xyz", state)

    assert result.action == "reply"
    assert "1 - Status" in result.reply
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# PEDIDO_STATUS / TRACKING sub-flow
# ---------------------------------------------------------------------------


def test_pedido_status_with_cpf_triggers_rag() -> None:
    state = _make_state("PEDIDO_STATUS")
    result = handle_menu_message("123.456.789-00", state)

    assert result.action == "query_rag"
    assert state.flow_state == "MAIN_MENU"


def test_pedido_status_without_identifier_asks_again() -> None:
    state = _make_state("PEDIDO_STATUS")
    result = handle_menu_message("não me lembro", state)

    assert result.action == "reply"
    assert state.flow_state == "PEDIDO_STATUS"
    assert "cpf" in result.reply.lower() or "pedido" in result.reply.lower()


def test_tracking_with_order_number_triggers_rag() -> None:
    state = _make_state("TRACKING")
    result = handle_menu_message("pedido 12345", state)

    assert result.action == "query_rag"
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# PRODUTO_DUVIDA / PRECO sub-flow
# ---------------------------------------------------------------------------


def test_produto_duvida_any_text_triggers_rag() -> None:
    state = _make_state("PRODUTO_DUVIDA")
    result = handle_menu_message("vocês têm capa para celular?", state)

    assert result.action == "query_rag"
    assert state.flow_state == "MAIN_MENU"


def test_preco_any_text_triggers_rag() -> None:
    state = _make_state("PRECO")
    result = handle_menu_message("quanto custa o modelo X?", state)

    assert result.action == "query_rag"
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# RECLAMACAO sub-flow
# ---------------------------------------------------------------------------


def test_reclamacao_entry_asks_for_description() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("8", state)

    assert result.action == "reply"
    assert state.flow_state == "RECLAMACAO"
    assert "descreva" in result.reply.lower() or "problema" in result.reply.lower()


def test_reclamacao_next_message_triggers_handoff() -> None:
    state = _make_state("RECLAMACAO")
    result = handle_menu_message("produto chegou quebrado", state)

    assert result.action == "handoff"
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# Navigation: back keywords work from any state
# ---------------------------------------------------------------------------


def test_back_keyword_0_resets_from_pedido_status() -> None:
    state = _make_state("PEDIDO_STATUS")
    result = handle_menu_message("0", state)

    assert result.action == "reply"
    assert state.flow_state == "MAIN_MENU"
    assert "1 - Status" in result.reply


def test_back_keyword_voltar_resets_from_reclamacao() -> None:
    state = _make_state("RECLAMACAO")
    result = handle_menu_message("voltar", state)

    assert result.action == "reply"
    assert state.flow_state == "MAIN_MENU"


def test_back_keyword_menu_resets_from_produto_duvida() -> None:
    state = _make_state("PRODUTO_DUVIDA")
    result = handle_menu_message("menu", state)

    assert result.action == "reply"
    assert state.flow_state == "MAIN_MENU"


# ---------------------------------------------------------------------------
# POLITICA_TROCA / POLITICA_FRETE / ATACADO (direct RAG on entry)
# ---------------------------------------------------------------------------


def test_politica_troca_entry_triggers_rag_with_override() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("5", state)

    assert result.action == "query_rag"
    assert result.rag_query_override is not None
    assert "troca" in result.rag_query_override.lower() or "devolução" in result.rag_query_override.lower()


def test_atacado_entry_triggers_rag_with_override() -> None:
    state = _make_state("MAIN_MENU")
    result = handle_menu_message("7", state)

    assert result.action == "query_rag"
    assert result.rag_query_override is not None
    assert "atacado" in result.rag_query_override.lower()


# ---------------------------------------------------------------------------
# Unknown flow state defaults safely
# ---------------------------------------------------------------------------


def test_unknown_flow_state_shows_menu() -> None:
    state = _make_state("SOME_LEGACY_STATE")
    result = handle_menu_message("qualquer coisa", state)

    assert result.action == "reply"
    assert "1 - Status" in result.reply
    assert state.flow_state == "MAIN_MENU"
