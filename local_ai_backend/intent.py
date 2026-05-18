import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    SAUDACAO = "saudacao"
    PEDIDO_STATUS = "pedido_status"
    TRACKING = "tracking"
    PRODUTO_DUVIDA = "produto_duvida"
    PRECO = "preco"
    POLITICA_TROCA = "politica_troca"
    POLITICA_FRETE = "politica_frete"
    ATACADO = "atacado"
    RECLAMACAO = "reclamacao"
    FALAR_COM_HUMANO = "falar_com_humano"
    DESCONHECIDO = "desconhecido"


@dataclass(slots=True)
class IntentResult:
    intent: Intent
    confidence: float
    reasons: list[str]
    order_number: str | None = None
    has_customer_identifier: bool = False


_ORDER_PATTERNS = (
    re.compile(r"#(\d{3,})"),
    re.compile(r"\bpedido\s*(?:numero|n[uú]mero|no|n[oº])?\s*(\d{3,})\b", re.IGNORECASE),
)

_CUSTOMER_IDENTIFIER_PATTERNS = (
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}\b"),
)

_INTENT_RULES: tuple[tuple[Intent, float, tuple[str, ...]], ...] = (
    (Intent.FALAR_COM_HUMANO, 0.98, ("humano", "atendente", "vendedor", "pessoa", "suporte humano")),
    (Intent.RECLAMACAO, 0.92, ("reclama", "problema", "péssimo", "horrível", "ruim", "insatisfeito", "erro")),
    (Intent.ATACADO, 0.95, ("atacado", "revenda", "revendedor", "lojista", "comprar em quantidade")),
    (Intent.TRACKING, 0.95, ("tracking", "rastreio", "rastreamento", "rastrear", "codigo de rastreio", "código de rastreio")),
    (Intent.PEDIDO_STATUS, 0.95, ("status do pedido", "meu pedido", "pedido", "acompanhar pedido", "andamento do pedido")),
    (Intent.PRECO, 0.88, ("preço", "preco", "valor", "quanto custa", "desconto", "promoção", "promocao")),
    (Intent.POLITICA_TROCA, 0.92, ("troca", "devolução", "devolucao", "reembolso", "arrependimento", "garantia")),
    (Intent.POLITICA_FRETE, 0.92, ("frete", "entrega", "prazo de entrega", "envio", "transportadora")),
    (Intent.PRODUTO_DUVIDA, 0.84, ("produto", "catálogo", "catalogo", "estoque", "disponível", "disponivel", "vocês têm", "vocês tem", "tem o produto")),
    (Intent.SAUDACAO, 0.80, ("oi", "ola", "olá", "bom dia", "boa tarde", "boa noite")),
)


def extract_order_number(message: str) -> str | None:
    for pattern in _ORDER_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None


def has_customer_identifier(message: str) -> bool:
    return any(pattern.search(message) for pattern in _CUSTOMER_IDENTIFIER_PATTERNS)


def classify_intent(message: str) -> IntentResult:
    normalized = (message or "").strip().lower()
    if not normalized:
        return IntentResult(
            intent=Intent.DESCONHECIDO,
            confidence=0.0,
            reasons=["empty_message"],
        )

    reasons: list[str] = []
    for intent, confidence, terms in _INTENT_RULES:
        matched_terms = [term for term in terms if term in normalized]
        if matched_terms:
            reasons.extend(f"matched:{term}" for term in matched_terms)
            return IntentResult(
                intent=intent,
                confidence=confidence,
                reasons=reasons,
                order_number=extract_order_number(message),
                has_customer_identifier=has_customer_identifier(message),
            )

    return IntentResult(
        intent=Intent.DESCONHECIDO,
        confidence=0.25,
        reasons=["no_rule_matched"],
        order_number=extract_order_number(message),
        has_customer_identifier=has_customer_identifier(message),
    )