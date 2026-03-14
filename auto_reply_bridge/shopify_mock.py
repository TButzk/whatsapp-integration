"""Fase 2: Módulo Shopify em modo mock.

Fornece dados de pedidos e produtos sem chamar a API real.
Troque SHOPIFY_MODE=real e implemente os clientes HTTP reais
quando estiver pronto para produção.
"""

import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger("auto-reply-bridge.shopify")

SHOPIFY_MODE = os.getenv("SHOPIFY_MODE", "mock").lower()
SHOPIFY_STORE_NAME = os.getenv("SHOPIFY_STORE_NAME", "Loja Demo")
SHOPIFY_CURRENCY = os.getenv("SHOPIFY_CURRENCY", "BRL")
SHOPIFY_MOCK_DELAY_MS = int(os.getenv("SHOPIFY_MOCK_DELAY_MS", "0"))

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_ORDERS: list[dict[str, Any]] = [
    {
        "number": "1001",
        "status": "paid",
        "status_label": "Pagamento confirmado",
        "items": ["Camiseta Básica Branca (P)", "Calça Jeans Skinny (38)"],
        "total": "R$ 189,90",
        "tracking": None,
        "estimated_delivery": "3–5 dias úteis",
        "customer_hint": "joao",
    },
    {
        "number": "1002",
        "status": "fulfilled",
        "status_label": "Pedido enviado",
        "items": ["Tênis Casual Cinza (41)"],
        "total": "R$ 259,90",
        "tracking": "BR123456789BR",
        "estimated_delivery": "1–2 dias úteis",
        "customer_hint": "maria",
    },
    {
        "number": "1003",
        "status": "canceled",
        "status_label": "Pedido cancelado",
        "items": ["Boné Estampado Azul"],
        "total": "R$ 79,90",
        "tracking": None,
        "estimated_delivery": None,
        "customer_hint": "carlos",
    },
]

_MOCK_PRODUCTS: list[dict[str, Any]] = [
    {
        "id": "p1",
        "name": "Camiseta Básica Branca",
        "price": "R$ 59,90",
        "available": True,
        "category": "camiseta vestuario roupa",
        "url": "https://loja.example.com/products/camiseta-basica-branca",
    },
    {
        "id": "p2",
        "name": "Calça Jeans Skinny",
        "price": "R$ 129,90",
        "available": True,
        "category": "calca jeans vestuario roupa",
        "url": "https://loja.example.com/products/calca-jeans-skinny",
    },
    {
        "id": "p3",
        "name": "Tênis Casual Cinza",
        "price": "R$ 259,90",
        "available": True,
        "category": "tenis calcado sapato calcado",
        "url": "https://loja.example.com/products/tenis-casual-cinza",
    },
    {
        "id": "p4",
        "name": "Boné Estampado Azul",
        "price": "R$ 79,90",
        "available": False,
        "category": "bone acessorio chapeu",
        "url": "https://loja.example.com/products/bone-estampado-azul",
    },
    {
        "id": "p5",
        "name": "Jaqueta Corta-Vento",
        "price": "R$ 349,90",
        "available": True,
        "category": "jaqueta vestuario casaco",
        "url": "https://loja.example.com/products/jaqueta-corta-vento",
    },
    {
        "id": "p6",
        "name": "Shorts Esportivo Preto",
        "price": "R$ 89,90",
        "available": True,
        "category": "shorts vestuario esporte roupa",
        "url": "https://loja.example.com/products/shorts-esportivo-preto",
    },
    {
        "id": "p7",
        "name": "Meia Esportiva Kit 3 pares",
        "price": "R$ 49,90",
        "available": True,
        "category": "meia acessorio esporte",
        "url": "https://loja.example.com/products/meia-esportiva-kit",
    },
    {
        "id": "p8",
        "name": "Cinto de Couro Marrom",
        "price": "R$ 69,90",
        "available": True,
        "category": "cinto acessorio couro",
        "url": "https://loja.example.com/products/cinto-de-couro-marrom",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_order_status(
    order_number: str, customer_hint: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Return mock order data for *order_number*, or None if not found.

    *customer_hint* is an optional partial customer name used as a secondary
    filter when orders share the same number prefix (not needed with mock data
    but kept for API parity with the real implementation).
    """
    if SHOPIFY_MOCK_DELAY_MS > 0:
        time.sleep(SHOPIFY_MOCK_DELAY_MS / 1000)

    order_number = order_number.strip().lstrip("#")

    for order in _MOCK_ORDERS:
        if order["number"] == order_number:
            if customer_hint and order.get("customer_hint"):
                if order["customer_hint"] not in customer_hint.lower():
                    return None
            return order

    return None


def search_products(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return mock products relevant to *query*, sorted by relevance."""
    if SHOPIFY_MOCK_DELAY_MS > 0:
        time.sleep(SHOPIFY_MOCK_DELAY_MS / 1000)

    query_lower = query.lower()
    query_words = set(re.split(r"\s+", query_lower))

    scored: list[tuple[int, dict[str, Any]]] = []
    for product in _MOCK_PRODUCTS:
        searchable = (product["name"] + " " + product["category"]).lower()
        score = sum(1 for w in query_words if w and w in searchable)
        if score > 0:
            scored.append((score, product))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_order_response(order: dict[str, Any]) -> str:
    """Return a human-readable string describing an order."""
    lines = [
        f"📦 Pedido #{order['number']} — {order['status_label']}",
        f"Itens: {', '.join(order['items'])}",
        f"Total: {order['total']}",
    ]
    if order.get("tracking"):
        lines.append(f"Rastreio: {order['tracking']}")
    if order.get("estimated_delivery"):
        lines.append(f"Previsão de entrega: {order['estimated_delivery']}")
    return "\n".join(lines)


def format_products_response(products: list[dict[str, Any]], query: str) -> str:
    """Return a human-readable string listing products."""
    if not products:
        return (
            f"Não encontrei produtos relacionados a '{query}' em nosso catálogo. "
            "Posso ajudar com mais alguma coisa?"
        )

    available = [p for p in products if p["available"]]
    unavailable = [p for p in products if not p["available"]]

    lines = [f"Encontrei {len(products)} produto(s) para '{query}':\n"]
    for p in available:
        lines.append(f"• {p['name']} — {p['price']} (disponível)\n  {p['url']}")
    for p in unavailable:
        lines.append(f"• {p['name']} — {p['price']} (indisponível)\n  {p['url']}")

    return "\n".join(lines)
