"""Fase 2: Detecção de intenção do cliente.

Classifica a mensagem em uma das intenções conhecidas para que o
orquestrador saiba qual subsistema consultar antes de gerar a resposta.

Melhoria 9 — sinais auxiliares:
  ``detect_intent_enriched()`` retorna um :class:`IntentResult` com flags
  booleanas (has_order_id, has_policy_terms, has_product_terms) que
  permitem ao orquestrador tomar decisões de roteamento sem custo LLM
  extra.  ``detect_intent()`` permanece como wrapper retrocompatível.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    ORDER = "order"
    PRODUCT = "product"
    INSTITUTIONAL = "institutional"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# IntentResult — resultado enriquecido (Melhoria 9)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IntentResult:
    """Resultado de detecção de intenção com sinais auxiliares.

    Campos extras permitem ao orquestrador decidir, por exemplo, se deve
    chamar o query planner ou disparar fallback de políticas sem custo LLM.
    """

    intent: Intent
    has_order_id: bool = False
    has_policy_terms: bool = False
    has_product_terms: bool = False
    order_number: Optional[str] = None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ORDER_PATTERNS = [
    r"\b(pedido|ordem|rastreio|rastreamento|entrega|status\s+do\s+pedido)\b",
    r"\b(c[oó]digo|n[uú]mero)\s*(do)?\s*(pedido|ordem)\b",
    r"#\d{3,}",
    r"\bpedido\s+\d{3,}",
]

_PRODUCT_PATTERNS = [
    r"\b(produto|produtos|cat[aá]logo|pre[cç]o|valor|custo|estoque|dispon[ií]vel)\b",
    r"\b(comprar|quero\s+comprar|procuro|voc[eê]s?\s+t[eê]m)\b",
    r"\b(camiseta|cal[cç]a|t[eê]nis|cal[cç]ado|jaqueta|bon[eé]|shorts|meia|cinto)\b",
]

_INSTITUTIONAL_PATTERNS = [
    r"\b(pol[ií]tica|sobre\s+a\s+empresa|miss[aã]o|valores|hist[oó]ria)\b",
    r"\b(hor[aá]rio|funcionamento|contato|endere[cç]o|telefone)\b",
    r"\b(devolu[cç][aã]o|garantia|troca|reembolso)\b",
    r"\b(como\s+funciona|quem\s+s[aã]o|o\s+que\s+[eé]\s+a\s+empresa)\b",
]

# ---------------------------------------------------------------------------
# Palavras-chave de políticas — usadas pelo fallback determinístico
# (Melhoria 4).  Mantidas aqui para reutilização por intent.py e app.py.
# ---------------------------------------------------------------------------

_POLICY_KEYWORDS: frozenset[str] = frozenset({
    "troca", "devolução", "devoluçao", "devolucao", "garantia", "reembolso",
    "prazo", "política", "politica", "funcionamento", "frete", "entrega",
    "horário", "horario", "reenvio", "estorno", "arrependimento", "defeito",
    "reclamação", "reclamacao",
})


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _check_policy_terms(msg_lower: str) -> bool:
    """Retorna True se a mensagem contém termos típicos de políticas/trocas."""
    for kw in _POLICY_KEYWORDS:
        if kw in msg_lower:
            return True
    # Padrões compostos que indicam dúvida sobre política
    return bool(re.search(
        r"\b(como\s+(funciona|faz|faço)|posso\s+(trocar|devolver|cancelar))\b",
        msg_lower,
    ))


def _check_product_terms(msg_lower: str) -> bool:
    """Retorna True se a mensagem contém termos de produto/catálogo."""
    for pattern in _PRODUCT_PATTERNS:
        if re.search(pattern, msg_lower):
            return True
    return False


def detect_intent_enriched(message: str) -> IntentResult:
    """Detecta intenção primária e sinais auxiliares em uma única passada.

    Retorna um :class:`IntentResult` com:
    - intent: intenção principal (ORDER, PRODUCT, INSTITUTIONAL, GENERAL)
    - has_order_id: True se a mensagem contém número de pedido explícito
    - has_policy_terms: True se contém palavras de política/troca/devolução
    - has_product_terms: True se contém termos de produto/catálogo
    - order_number: número do pedido extraído, ou None
    """
    msg_lower = message.lower()

    # Extrair sinais auxiliares (baratos — apenas regex)
    order_number = extract_order_number(message)
    has_order_id = order_number is not None
    has_policy = _check_policy_terms(msg_lower)
    has_product = _check_product_terms(msg_lower)

    # Detectar intenção primária (prioridade: ORDER > PRODUCT > INSTITUTIONAL)
    intent = Intent.GENERAL
    for pattern in _ORDER_PATTERNS:
        if re.search(pattern, msg_lower):
            intent = Intent.ORDER
            break

    if intent == Intent.GENERAL:
        for pattern in _PRODUCT_PATTERNS:
            if re.search(pattern, msg_lower):
                intent = Intent.PRODUCT
                break

    if intent == Intent.GENERAL:
        for pattern in _INSTITUTIONAL_PATTERNS:
            if re.search(pattern, msg_lower):
                intent = Intent.INSTITUTIONAL
                break

    return IntentResult(
        intent=intent,
        has_order_id=has_order_id,
        has_policy_terms=has_policy,
        has_product_terms=has_product,
        order_number=order_number,
    )


def detect_intent(message: str) -> Intent:
    """Return the primary :class:`Intent` inferred from *message*.

    Wrapper retrocompatível — delega para ``detect_intent_enriched()``
    e retorna apenas a intenção primária.
    """
    return detect_intent_enriched(message).intent


def extract_order_number(message: str) -> Optional[str]:
    """Extract an order number string from *message*, or return None."""
    patterns = [
        r"#(\d{3,})",
        r"\bpedido\s+(\d{3,})",
        r"\bordem\s+(\d{3,})",
        r"\bn[uú]mero\s+(\d{3,})",
        r"\bc[oó]digo\s+(\d{3,})",
    ]
    msg_lower = message.lower()
    for pattern in patterns:
        match = re.search(pattern, msg_lower)
        if match:
            return match.group(1)
    return None


def extract_product_query(message: str) -> str:
    """Extract a product-oriented query from a natural-language customer message."""
    cleaned = message.strip().lower()
    if not cleaned:
        return ""

    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = re.sub(
        r"\b(voc[eê]|voc[eê]s|tem|t[eê]m|quero|comprar|procuro|procurando|gostaria|"
        r"mostrar|mostra|me|pra|para|de|do|da|dos|das|um|uma|o|a|os|as|"
        r"quais|qual|produto|produtos|cat[aá]logo|dispon[ií]vel|dispon[ií]veis)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned or message.strip()
