"""Fase 2: Detecção de intenção do cliente.

Classifica a mensagem em uma das intenções conhecidas para que o
orquestrador saiba qual subsistema consultar antes de gerar a resposta.
"""

import re
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    ORDER = "order"
    PRODUCT = "product"
    INSTITUTIONAL = "institutional"
    GENERAL = "general"


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
# Public functions
# ---------------------------------------------------------------------------


def detect_intent(message: str) -> Intent:
    """Return the primary :class:`Intent` inferred from *message*."""
    msg_lower = message.lower()

    for pattern in _ORDER_PATTERNS:
        if re.search(pattern, msg_lower):
            return Intent.ORDER

    for pattern in _PRODUCT_PATTERNS:
        if re.search(pattern, msg_lower):
            return Intent.PRODUCT

    for pattern in _INSTITUTIONAL_PATTERNS:
        if re.search(pattern, msg_lower):
            return Intent.INSTITUTIONAL

    return Intent.GENERAL


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
