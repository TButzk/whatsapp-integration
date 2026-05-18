"""Menu-driven conversation flow manager.

Maintains a state machine per conversation so the WhatsApp bot behaves
like a numbered interactive menu:

  START → any message → greeting + main menu
  MAIN_MENU → "1"–"9" → enter sub-flow
  Any state → "0" / "voltar" / "menu" → back to main menu

Public API
----------
  handle_menu_message(content, state) -> MenuResult
  MenuResult.action: "reply" | "query_rag" | "handoff"
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_backend.intent import (
    Intent,
    classify_intent,
    extract_order_number,
    has_customer_identifier,
)
from local_ai_backend.services.conversation_state import ConversationState

# ---------------------------------------------------------------------------
# Text constants
# ---------------------------------------------------------------------------

GREETING_TEXT = "Olá! 😊 Seja bem-vindo(a) ao nosso atendimento!\n\n"

MAIN_MENU_TEXT = (
    "*Menu de Atendimento:*\n\n"
    "1 - Status do pedido\n"
    "2 - Rastreamento\n"
    "3 - Dúvidas sobre produto\n"
    "4 - Preços e promoções\n"
    "5 - Troca e devolução\n"
    "6 - Frete e prazo de entrega\n"
    "7 - Atacado / Revendedor\n"
    "8 - Reclamação\n"
    "9 - Outros / Falar com atendente\n\n"
    "_Digite o número da opção desejada ou 0 para exibir este menu._"
)

BACK_HINT = "\n\n_Mais alguma dúvida? Digite *0* para voltar ao menu._"

# ---------------------------------------------------------------------------
# MenuResult
# ---------------------------------------------------------------------------


@dataclass
class MenuResult:
    """Outcome of processing one user message through the menu state machine.

    action values:
      "reply"     — send `reply` directly, no RAG/LLM needed
      "query_rag" — run RAG (+ LLM fallback) using `rag_query_override or content`
      "handoff"   — send `reply` and signal human handoff
    """

    reply: str
    action: str
    rag_query_override: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OPTION_TO_FLOW: dict[str, str] = {
    "1": "PEDIDO_STATUS",
    "2": "TRACKING",
    "3": "PRODUTO_DUVIDA",
    "4": "PRECO",
    "5": "POLITICA_TROCA",
    "6": "POLITICA_FRETE",
    "7": "ATACADO",
    "8": "RECLAMACAO",
    "9": "OUTROS",
}

_INTENT_TO_FLOW: dict[Intent, str] = {
    Intent.PEDIDO_STATUS: "PEDIDO_STATUS",
    Intent.TRACKING: "TRACKING",
    Intent.PRODUTO_DUVIDA: "PRODUTO_DUVIDA",
    Intent.PRECO: "PRECO",
    Intent.POLITICA_TROCA: "POLITICA_TROCA",
    Intent.POLITICA_FRETE: "POLITICA_FRETE",
    Intent.ATACADO: "ATACADO",
    Intent.RECLAMACAO: "RECLAMACAO",
    Intent.FALAR_COM_HUMANO: "OUTROS",
}

_BACK_KEYWORDS: frozenset[str] = frozenset(
    {"0", "voltar", "menu", "inicio", "início", "home", "sair"}
)

_GREETING_KEYWORDS: frozenset[str] = frozenset(
    {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "hey", "hi", "tudo bem"}
)


def _show_menu(state: ConversationState, prefix: str = "") -> MenuResult:
    state.flow_state = "MAIN_MENU"
    return MenuResult(reply=prefix + MAIN_MENU_TEXT, action="reply")


def _is_greeting(text: str) -> bool:
    return text in _GREETING_KEYWORDS or any(k in text for k in _GREETING_KEYWORDS)


def _flow_entry(flow: str, state: ConversationState) -> MenuResult:
    """Return the initial MenuResult when the user enters a sub-flow."""
    state.flow_state = flow

    if flow == "PEDIDO_STATUS":
        return MenuResult(
            reply=(
                "Para verificar o status do pedido, informe o "
                "*número do pedido* ou o *CPF cadastrado*:"
            ),
            action="reply",
        )
    if flow == "TRACKING":
        return MenuResult(
            reply=(
                "Para rastrear sua encomenda, informe o "
                "*número do pedido* ou *código de rastreio*:"
            ),
            action="reply",
        )
    if flow == "PRODUTO_DUVIDA":
        return MenuResult(
            reply="Qual é a sua dúvida sobre o produto? 😊",
            action="reply",
        )
    if flow == "PRECO":
        return MenuResult(
            reply="Qual produto ou categoria você quer saber sobre preço ou promoção?",
            action="reply",
        )
    if flow == "POLITICA_TROCA":
        state.flow_state = "MAIN_MENU"
        return MenuResult(
            reply="Para dúvidas sobre troca ou devolução, entre em contato com nossa equipe.",
            action="query_rag",
            rag_query_override="política de troca devolução reembolso arrependimento",
        )
    if flow == "POLITICA_FRETE":
        state.flow_state = "MAIN_MENU"
        return MenuResult(
            reply="Para dúvidas sobre frete ou entrega, entre em contato com nossa equipe.",
            action="query_rag",
            rag_query_override="prazo de entrega frete transportadora envio",
        )
    if flow == "ATACADO":
        state.flow_state = "MAIN_MENU"
        return MenuResult(
            reply="Para informações sobre atacado ou revenda, entre em contato com nossa equipe.",
            action="query_rag",
            rag_query_override="atacado revendedor lojista comprar em quantidade",
        )
    if flow == "RECLAMACAO":
        return MenuResult(
            reply=(
                "Lamentamos pelo inconveniente! 😟\n"
                "Por favor, descreva o problema para que possamos te ajudar melhor:"
            ),
            action="reply",
        )
    if flow == "OUTROS":
        state.flow_state = "MAIN_MENU"
        return MenuResult(
            reply=(
                "Vou transferir você para um de nossos atendentes. "
                "Em breve alguém entrará em contato! 👩‍💼"
            ),
            action="handoff",
        )
    # Unknown flow — show menu
    return _show_menu(state)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle_menu_message(content: str, state: ConversationState) -> MenuResult:
    """Resolve the next bot action given a user message and the current flow state."""
    normalized = (content or "").strip().lower()

    # Navigation: back/menu command overrides any flow state
    if normalized in _BACK_KEYWORDS:
        return _show_menu(state)

    current = state.flow_state or "START"

    # ------------------------------------------------------------------
    # START: first message in conversation → greeting + menu
    # ------------------------------------------------------------------
    if current == "START":
        return _show_menu(state, prefix=GREETING_TEXT)

    # ------------------------------------------------------------------
    # MAIN_MENU: waiting for numbered option or free-text intent
    # ------------------------------------------------------------------
    if current == "MAIN_MENU":
        if normalized in _OPTION_TO_FLOW:
            return _flow_entry(_OPTION_TO_FLOW[normalized], state)

        if _is_greeting(normalized):
            return _show_menu(state, prefix="Olá! Como posso te ajudar?\n\n")

        # Free text → detect intent and map to flow
        intent_result = classify_intent(content)
        flow = _INTENT_TO_FLOW.get(intent_result.intent)
        if flow and intent_result.confidence >= 0.70:
            return _flow_entry(flow, state)

        return _show_menu(state, prefix="Não entendi. Escolha uma das opções abaixo:\n\n")

    # ------------------------------------------------------------------
    # PEDIDO_STATUS / TRACKING: waiting for customer identifier
    # ------------------------------------------------------------------
    if current in ("PEDIDO_STATUS", "TRACKING"):
        if has_customer_identifier(content) or extract_order_number(content):
            state.flow_state = "MAIN_MENU"
            return MenuResult(reply="", action="query_rag")
        return MenuResult(
            reply=(
                "Não identifiquei um número de pedido, CPF ou e-mail na sua mensagem. "
                "Por favor, informe um deles para continuar:"
            ),
            action="reply",
        )

    # ------------------------------------------------------------------
    # PRODUTO_DUVIDA / PRECO: any free-text → query RAG/LLM
    # ------------------------------------------------------------------
    if current in ("PRODUTO_DUVIDA", "PRECO"):
        state.flow_state = "MAIN_MENU"
        return MenuResult(reply="", action="query_rag")

    # ------------------------------------------------------------------
    # RECLAMACAO: user just described the complaint → handoff
    # ------------------------------------------------------------------
    if current == "RECLAMACAO":
        state.flow_state = "MAIN_MENU"
        return MenuResult(
            reply=(
                "Obrigado por nos contar. Sua reclamação foi registrada e "
                "um atendente entrará em contato em breve. 🙏"
            ),
            action="handoff",
        )

    # ------------------------------------------------------------------
    # Unknown / legacy state → show menu
    # ------------------------------------------------------------------
    return _show_menu(state)
