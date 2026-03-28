"""Fase 2: Módulo Shopify em modo MCP (Model Context Protocol).

Conecta ao Shopify Storefront MCP Server para consultar produtos em tempo real.

Referência: https://shopify.dev/docs/apps/build/storefront-mcp/servers/storefront

Limitação: o Storefront API não expõe dados privados de pedidos de clientes.
Para consultas de pedidos, utilize SHOPIFY_MODE=mock (dados de teste) ou
implemente integração via Admin API em shopify_real.py.

Variáveis de ambiente requeridas:
    SHOPIFY_MCP_ENDPOINT      URL do servidor MCP da Shopify
    SHOPIFY_STOREFRONT_TOKEN  Token público de acesso ao Storefront API

Variáveis opcionais:
    SHOPIFY_STORE_NAME        Nome da loja (exibido nas respostas)
    SHOPIFY_CURRENCY          Moeda padrão (padrão: BRL)
    SHOPIFY_MCP_TIMEOUT       Timeout em segundos (padrão: 30)
"""

import json
import logging
import os
import uuid
from typing import Any, Optional

import requests

logger = logging.getLogger("auto-reply-bridge.shopify")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_MCP_ENDPOINT = os.getenv(
    "SHOPIFY_MCP_ENDPOINT",
    "https://mcp.shopify.com/storefront/v1",
)
SHOPIFY_STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_TOKEN", "")
SHOPIFY_STORE_NAME = os.getenv("SHOPIFY_STORE_NAME", "Loja")
SHOPIFY_CURRENCY = os.getenv("SHOPIFY_CURRENCY", "BRL")
SHOPIFY_MCP_TIMEOUT = int(os.getenv("SHOPIFY_MCP_TIMEOUT", "30"))

_MCP_PROTOCOL_VERSION = "2024-11-05"


# ---------------------------------------------------------------------------
# MCP transport helpers
# ---------------------------------------------------------------------------


def _mcp_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if SHOPIFY_STOREFRONT_TOKEN:
        headers["X-Shopify-Storefront-Access-Token"] = SHOPIFY_STOREFRONT_TOKEN
    return headers


def _mcp_post(method: str, params: dict[str, Any] | None = None) -> Any:
    """Send a JSON-RPC 2.0 request to the MCP endpoint and return the result.

    Returns the ``result`` field of a successful response, or ``None`` on any
    error (network failure, HTTP error, JSON-RPC error).
    """
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {},
    }
    try:
        resp = requests.post(
            SHOPIFY_MCP_ENDPOINT,
            headers=_mcp_headers(),
            json=payload,
            timeout=SHOPIFY_MCP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            logger.error(
                "MCP server returned error for method=%s: %s", method, data["error"]
            )
            return None
        return data.get("result")
    except requests.exceptions.RequestException as exc:
        logger.error(
            "MCP HTTP request failed | endpoint=%s method=%s error=%s",
            SHOPIFY_MCP_ENDPOINT,
            method,
            exc,
        )
        return None
    except Exception as exc:  # pragma: no cover – unexpected parse errors
        logger.error("Unexpected MCP client error: %s", exc)
        return None


def _initialize() -> bool:
    """Perform the MCP initialize handshake. Returns True on success."""
    result = _mcp_post(
        "initialize",
        {
            "protocolVersion": _MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "whatsapp-bridge", "version": "1.0.0"},
        },
    )
    if result is None:
        logger.warning(
            "MCP initialization failed. Check SHOPIFY_MCP_ENDPOINT and SHOPIFY_STOREFRONT_TOKEN."
        )
        return False
    return True


def _call_tool(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Call an MCP tool and return all text content blocks joined, or None."""
    result = _mcp_post("tools/call", {"name": tool_name, "arguments": arguments})
    if result is None:
        return None
    content_blocks: list[dict[str, Any]] = result.get("content") or []
    texts = [
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else None


# ---------------------------------------------------------------------------
# Product helpers
# ---------------------------------------------------------------------------


def _parse_products_from_text(raw: str, limit: int) -> list[dict[str, Any]]:
    """Try to parse a JSON product list from *raw* MCP tool output."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return _normalize_products(data[:limit])
        if isinstance(data, dict):
            items = data.get("products") or data.get("edges") or []
            return _normalize_products(items[:limit])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: wrap the whole text as a single entry so the LLM can use it
    return [
        {
            "id": "mcp-text",
            "name": raw[:200],
            "price": "",
            "available": True,
            "category": "",
            "url": "",
        }
    ]


def _normalize_products(raw_products: list[Any]) -> list[dict[str, Any]]:
    """Normalise Shopify product nodes to the internal product dict format."""
    normalized: list[dict[str, Any]] = []
    for item in raw_products:
        if not isinstance(item, dict):
            continue
        # GraphQL edges/node pattern
        node = item.get("node") or item
        normalized.append(
            {
                "id": str(node.get("id") or node.get("handle") or ""),
                "name": node.get("title") or node.get("name") or "",
                "price": _format_price(node),
                "available": node.get("availableForSale", node.get("available", True)),
                "category": " ".join(
                    (node.get("tags") or [])
                    if isinstance(node.get("tags"), list)
                    else str(node.get("category") or "").split()
                ),
                "url": node.get("onlineStoreUrl") or node.get("url") or "",
            }
        )
    return normalized


def _format_price(product: dict[str, Any]) -> str:
    """Extract and format a price from a Shopify product node."""
    price_range: dict[str, Any] = product.get("priceRange") or {}
    min_price: dict[str, Any] = price_range.get("minVariantPrice") or {}
    amount = min_price.get("amount") or product.get("price", "")
    currency = min_price.get("currencyCode") or SHOPIFY_CURRENCY
    if amount:
        try:
            value = float(amount)
            int_part = f"{int(value):,}".replace(",", ".")
            dec_part = f"{value:.2f}".split(".")[1]
            return f"{currency} {int_part},{dec_part}"
        except (ValueError, TypeError):
            return str(amount)
    return ""


# ---------------------------------------------------------------------------
# Public API  (same interface as shopify_mock.py)
# ---------------------------------------------------------------------------


def search_products(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search products via the Shopify Storefront MCP server.

    Initialises the MCP session, calls the ``search_products`` tool, and
    returns a normalised list compatible with :func:`format_products_response`.
    Returns an empty list if the server is unreachable or returns no results.
    """
    if not _initialize():
        return []

    raw = _call_tool("search_products", {"query": query, "limit": limit})
    if not raw:
        logger.warning(
            "MCP search_products returned no results | query=%r", query
        )
        return []

    return _parse_products_from_text(raw, limit)


def get_order_status(
    order_number: str, customer_hint: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Order lookup via MCP.

    The Shopify Storefront API does not expose private customer order data,
    so this function always returns ``None``.  Use ``SHOPIFY_MODE=mock`` for
    order testing or implement via the Admin API in ``shopify_real.py``.
    """
    logger.info(
        "Order lookup #%s requested but Storefront MCP does not support "
        "private order data. Set SHOPIFY_MODE=mock for order testing.",
        order_number,
    )
    return None


def format_order_response(order: dict[str, Any]) -> str:
    """Format an order dict as human-readable text.

    Delegates to :mod:`shopify_mock` for formatting parity; in practice this
    function is only called when ``get_order_status`` returns a non-None value,
    which the Storefront MCP mode never does.
    """
    from shopify_mock import format_order_response as _fmt  # pragma: no cover

    return _fmt(order)  # pragma: no cover


def format_products_response(products: list[dict[str, Any]], query: str) -> str:
    """Return a human-readable string listing MCP product results."""
    if not products:
        return (
            f"Não encontrei produtos relacionados a '{query}' em nosso catálogo. "
            "Posso ajudar com mais alguma coisa?"
        )

    available = [p for p in products if p.get("available", True)]
    unavailable = [p for p in products if not p.get("available", True)]

    lines = [f"Encontrei {len(products)} produto(s) para '{query}':\n"]
    for p in available:
        line = f"• {p['name']}"
        if p.get("price"):
            line += f" — {p['price']} (disponível)"
        if p.get("url"):
            line += f"\n  {p['url']}"
        lines.append(line)
    for p in unavailable:
        line = f"• {p['name']}"
        if p.get("price"):
            line += f" — {p['price']} (indisponível)"
        if p.get("url"):
            line += f"\n  {p['url']}"
        lines.append(line)

    return "\n".join(lines)
