"""Fase 2: Integração Shopify via MCP (Storefront + Customer Account).

Conecta ao Shopify Storefront MCP Server para consultar catálogo, políticas/FAQ
e recursos de carrinho da loja. Opcionalmente conecta ao Customer Account MCP
para consultas autenticadas de pedidos e conta do cliente.

Referência: https://shopify.dev/docs/apps/build/storefront-mcp/servers/storefront

Limitação importante: o Storefront MCP não expõe dados privados de pedidos de
clientes. Para status de pedido, é necessário configurar Customer Account MCP
com token Bearer autenticado.

Variáveis de ambiente requeridas:
    SHOPIFY_MCP_ENDPOINT  URL do servidor MCP da loja, no formato:
                          https://{sua-loja}.myshopify.com/api/mcp

Variáveis opcionais:
    SHOPIFY_STORE_NAME    Nome da loja (exibido nas respostas)
    SHOPIFY_CURRENCY      Moeda padrão (padrão: BRL)
    SHOPIFY_MCP_TIMEOUT   Timeout em segundos (padrão: 30)
    SHOPIFY_CUSTOMER_MCP_ENDPOINT  Endpoint do Customer Account MCP
    SHOPIFY_CUSTOMER_MCP_TOKEN     Token Bearer para Customer Account MCP
    SHOPIFY_CUSTOMER_MCP_TIMEOUT   Timeout em segundos (padrão: 30)

Obs: o Storefront MCP Server normalmente não requer autenticação, mas algumas
lojas podem restringir acesso no ambiente delas.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

_DOTENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

logger = logging.getLogger("auto-reply-bridge.shopify")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_MCP_ENDPOINT = os.getenv(
    "SHOPIFY_MCP_ENDPOINT",
    "",  # ex: https://{sua-loja}.myshopify.com/api/mcp
)
SHOPIFY_CUSTOMER_MCP_ENDPOINT = os.getenv(
    "SHOPIFY_CUSTOMER_MCP_ENDPOINT",
    "",  # ex: https://shopify.com/{shop_id}/account/customer/api/mcp
)
SHOPIFY_CUSTOMER_MCP_TOKEN = os.getenv("SHOPIFY_CUSTOMER_MCP_TOKEN", "").strip()
SHOPIFY_CUSTOMER_CLIENT_ID = os.getenv("SHOPIFY_CUSTOMER_CLIENT_ID", "").strip()
SHOPIFY_CUSTOMER_CLIENT_SECRET = os.getenv("SHOPIFY_CUSTOMER_CLIENT_SECRET", "").strip()
SHOPIFY_CUSTOMER_REFRESH_TOKEN = os.getenv("SHOPIFY_CUSTOMER_REFRESH_TOKEN", "").strip()
SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT = os.getenv(
    "SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT", ""
).strip()
SHOPIFY_CUSTOMER_SCOPES = os.getenv(
    "SHOPIFY_CUSTOMER_SCOPES",
    "openid email customer-account-api:full customer-account-mcp-api:full",
).strip()
SHOPIFY_STORE_NAME = os.getenv("SHOPIFY_STORE_NAME", "Loja")
SHOPIFY_CURRENCY = os.getenv("SHOPIFY_CURRENCY", "BRL")
SHOPIFY_MCP_TIMEOUT = int(os.getenv("SHOPIFY_MCP_TIMEOUT", "30"))
SHOPIFY_CUSTOMER_MCP_TIMEOUT = int(os.getenv("SHOPIFY_CUSTOMER_MCP_TIMEOUT", "30"))

_MCP_PROTOCOL_VERSION = "2024-11-05"
SUPPORTS_ORDER_LOOKUP = bool(SHOPIFY_CUSTOMER_MCP_ENDPOINT and SHOPIFY_CUSTOMER_MCP_TOKEN)
_TOOLS_CACHE: list[str] | None = None
_CUSTOMER_TOOLS_CACHE: list[str] | None = None
_CUSTOMER_INITIALIZED = False
_CUSTOMER_TOKEN_EXPIRES_AT = 0.0


# ---------------------------------------------------------------------------
# MCP transport helpers
# ---------------------------------------------------------------------------


def _mcp_headers() -> dict[str, str]:
    # Storefront MCP Server does not require authentication.
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def _customer_mcp_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if SHOPIFY_CUSTOMER_MCP_TOKEN:
        headers["Authorization"] = f"Bearer {SHOPIFY_CUSTOMER_MCP_TOKEN}"
    return headers


def _infer_customer_oauth_token_endpoint() -> str:
    """Infer Customer OAuth token endpoint from configured values when possible."""
    if SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT:
        return SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT

    if SHOPIFY_CUSTOMER_MCP_ENDPOINT:
        # Example:
        #   mcp_api: https://shopify.com/79799156988/account/customer/api/mcp
        #   token : https://shopify.com/authentication/79799156988/oauth/token
        parsed = urlparse(SHOPIFY_CUSTOMER_MCP_ENDPOINT)
        m = re.search(r"/(\d+)/account/customer/api/mcp", parsed.path)
        if m:
            shop_id = m.group(1)
            return f"{parsed.scheme}://{parsed.netloc}/authentication/{shop_id}/oauth/token"

    return ""


def _refresh_customer_access_token() -> bool:
    """Refresh access token for Customer MCP using OAuth refresh token."""
    global SHOPIFY_CUSTOMER_MCP_TOKEN, SUPPORTS_ORDER_LOOKUP  # noqa: PLW0603
    global _CUSTOMER_TOKEN_EXPIRES_AT  # noqa: PLW0603

    token_endpoint = _infer_customer_oauth_token_endpoint()
    if not token_endpoint:
        return False
    if not (SHOPIFY_CUSTOMER_CLIENT_ID and SHOPIFY_CUSTOMER_CLIENT_SECRET and SHOPIFY_CUSTOMER_REFRESH_TOKEN):
        return False

    data = {
        "grant_type": "refresh_token",
        "refresh_token": SHOPIFY_CUSTOMER_REFRESH_TOKEN,
    }
    if SHOPIFY_CUSTOMER_SCOPES:
        data["scope"] = SHOPIFY_CUSTOMER_SCOPES

    try:
        response = requests.post(
            token_endpoint,
            data=data,
            auth=(SHOPIFY_CUSTOMER_CLIENT_ID, SHOPIFY_CUSTOMER_CLIENT_SECRET),
            timeout=SHOPIFY_CUSTOMER_MCP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            logger.warning("Customer OAuth token refresh did not return access_token.")
            return False

        SHOPIFY_CUSTOMER_MCP_TOKEN = access_token
        SUPPORTS_ORDER_LOOKUP = bool(SHOPIFY_CUSTOMER_MCP_ENDPOINT and SHOPIFY_CUSTOMER_MCP_TOKEN)

        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            _CUSTOMER_TOKEN_EXPIRES_AT = max(0.0, float(expires_in))
        logger.info("Customer OAuth access token refreshed successfully.")
        return True
    except requests.exceptions.RequestException as exc:
        logger.warning("Customer OAuth refresh failed: %s", exc)
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning("Unexpected Customer OAuth refresh error: %s", exc)
        return False


def _ensure_customer_token() -> bool:
    """Ensure Customer MCP access token is available (direct or via refresh)."""
    global SUPPORTS_ORDER_LOOKUP  # noqa: PLW0603
    if SHOPIFY_CUSTOMER_MCP_TOKEN:
        SUPPORTS_ORDER_LOOKUP = bool(SHOPIFY_CUSTOMER_MCP_ENDPOINT and SHOPIFY_CUSTOMER_MCP_TOKEN)
        return SUPPORTS_ORDER_LOOKUP

    refreshed = _refresh_customer_access_token()
    SUPPORTS_ORDER_LOOKUP = bool(SHOPIFY_CUSTOMER_MCP_ENDPOINT and SHOPIFY_CUSTOMER_MCP_TOKEN)
    return refreshed and SUPPORTS_ORDER_LOOKUP


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


def _customer_mcp_post(method: str, params: dict[str, Any] | None = None) -> Any:
    """Send a JSON-RPC 2.0 request to Customer Account MCP endpoint."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {},
    }
    try:
        resp = requests.post(
            SHOPIFY_CUSTOMER_MCP_ENDPOINT,
            headers=_customer_mcp_headers(),
            json=payload,
            timeout=SHOPIFY_CUSTOMER_MCP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            logger.error(
                "Customer MCP returned error for method=%s: %s",
                method,
                data["error"],
            )
            return None
        return data.get("result")
    except requests.exceptions.RequestException as exc:
        logger.error(
            "Customer MCP HTTP request failed | endpoint=%s method=%s error=%s",
            SHOPIFY_CUSTOMER_MCP_ENDPOINT,
            method,
            exc,
        )
        return None
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected Customer MCP client error: %s", exc)
        return None


def _initialize() -> bool:
    """Perform the MCP initialize handshake. Returns True on success."""
    if not SHOPIFY_MCP_ENDPOINT:
        logger.warning(
            "SHOPIFY_MCP_ENDPOINT is not configured. "
            "Set it to https://{sua-loja}.myshopify.com/api/mcp"
        )
        return False
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
            "MCP initialization failed. Check SHOPIFY_MCP_ENDPOINT: %s",
            SHOPIFY_MCP_ENDPOINT,
        )
        return False
    return True


def _initialize_customer() -> bool:
    """Perform initialize handshake against Customer Account MCP."""
    global _CUSTOMER_INITIALIZED  # noqa: PLW0603
    if _CUSTOMER_INITIALIZED:
        return True

    if not SHOPIFY_CUSTOMER_MCP_ENDPOINT:
        logger.info("SHOPIFY_CUSTOMER_MCP_ENDPOINT is not configured.")
        return False
    if not _ensure_customer_token():
        logger.info("Customer MCP token unavailable (configure token or OAuth refresh vars).")
        return False

    result = _customer_mcp_post(
        "initialize",
        {
            "protocolVersion": _MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "whatsapp-bridge", "version": "1.0.0"},
        },
    )
    if result is None:
        logger.warning(
            "Customer MCP initialization failed. Check endpoint/token configuration."
        )
        return False
    _CUSTOMER_INITIALIZED = True
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


def _call_customer_tool(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Call a Customer Account MCP tool and return text content, if any."""
    if not _ensure_customer_token():
        return None
    result = _customer_mcp_post("tools/call", {"name": tool_name, "arguments": arguments})
    if result is None:
        return None
    content_blocks: list[dict[str, Any]] = result.get("content") or []
    texts = [
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else None


def _list_tools() -> list[str]:
    """Return available MCP tool names (cached per process)."""
    global _TOOLS_CACHE  # noqa: PLW0603
    if _TOOLS_CACHE is not None:
        return _TOOLS_CACHE

    if not _initialize():
        _TOOLS_CACHE = []
        return _TOOLS_CACHE

    result = _mcp_post("tools/list", {}) or {}
    tools = result.get("tools") if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.append(tool["name"])

    _TOOLS_CACHE = names
    return names


def _list_customer_tools() -> list[str]:
    """Return available Customer Account MCP tool names (cached)."""
    global _CUSTOMER_TOOLS_CACHE  # noqa: PLW0603
    if _CUSTOMER_TOOLS_CACHE is not None:
        return _CUSTOMER_TOOLS_CACHE

    if not _initialize_customer():
        _CUSTOMER_TOOLS_CACHE = []
        return _CUSTOMER_TOOLS_CACHE

    result = _customer_mcp_post("tools/list", {}) or {}
    tools = result.get("tools") if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.append(tool["name"])

    _CUSTOMER_TOOLS_CACHE = names
    return names


def is_order_lookup_available() -> bool:
    """Return True when Customer MCP lookup is configured and initialized."""
    return _initialize_customer()


def _call_first_available_tool(
    candidate_names: list[str], arguments: dict[str, Any]
) -> str | None:
    """Call the first available tool name from *candidate_names*."""
    available = set(_list_tools())
    if not available:
        return None

    for name in candidate_names:
        if name in available:
            return _call_tool(name, arguments)

    logger.info(
        "None of the requested MCP tools are available: %s",
        ", ".join(candidate_names),
    )
    return None


def _call_first_available_customer_tool(
    candidate_names: list[str], arguments: dict[str, Any]
) -> str | None:
    """Call first available Customer Account MCP tool from candidate list."""
    available = set(_list_customer_tools())
    if not available:
        return None

    for name in candidate_names:
        if name in available:
            return _call_customer_tool(name, arguments)

    logger.info(
        "None of the requested Customer MCP tools are available: %s",
        ", ".join(candidate_names),
    )
    return None


def _extract_json_from_text(raw: str) -> Any:
    """Try to decode JSON payload from plain or fenced text output."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


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

    raw = _call_first_available_tool(
        ["search_shop_catalog", "search_products"],
        {"query": query, "context": query},
    )
    if not raw:
        logger.warning(
            "MCP search_shop_catalog returned no results | query=%r", query
        )
        return []

    return _parse_products_from_text(raw, limit)


def search_policies(query: str) -> str:
    """Search store policies/FAQs via the Shopify Storefront MCP server.

    Returns the raw text answer or an empty string on any error.
    """
    if not _initialize():
        return ""

    raw = _call_first_available_tool(
        ["search_shop_policies_and_faqs", "search_policies", "search_faqs"],
        {"query": query, "context": query},
    )
    if not raw:
        logger.info("MCP search_shop_policies_and_faqs returned no results | query=%r", query)
        return ""
    return raw


def list_available_resources() -> str:
    """Return a compact string describing Storefront + Customer MCP tools."""
    storefront_tools = _list_tools()
    customer_tools = _list_customer_tools()
    parts: list[str] = []
    if storefront_tools:
        parts.append("Storefront: " + ", ".join(sorted(storefront_tools)))
    if customer_tools:
        parts.append("Customer: " + ", ".join(sorted(customer_tools)))
    if not parts:
        return "Ferramentas MCP indisponiveis no momento."
    return " | ".join(parts)


def _normalize_order_payload(payload: dict[str, Any], fallback_number: str) -> dict[str, Any]:
    """Normalize generic order payload from Customer MCP to app order schema."""
    status = str(payload.get("fulfillmentStatus") or payload.get("status") or "").strip()
    status_label = (
        str(payload.get("statusLabel") or payload.get("displayStatus") or "").strip()
        or status
        or "Status desconhecido"
    )
    order_number = str(
        payload.get("number")
        or payload.get("orderNumber")
        or payload.get("name")
        or fallback_number
    ).strip("# ")

    total = payload.get("total") or payload.get("totalPrice") or payload.get("currentTotalPrice") or ""
    if isinstance(total, dict):
        amount = total.get("amount") or ""
        currency = total.get("currencyCode") or SHOPIFY_CURRENCY
        total = f"{currency} {amount}".strip()
    elif total:
        total = str(total)

    tracking = payload.get("trackingNumber") or payload.get("tracking") or ""
    tracking_url = payload.get("trackingUrl") or payload.get("tracking_url") or ""

    return {
        "number": order_number,
        "status": status or "unknown",
        "status_label": status_label,
        "items": payload.get("items") or payload.get("lineItems") or [],
        "total": total,
        "tracking": tracking,
        "tracking_url": tracking_url,
        "carrier": payload.get("carrier") or payload.get("shippingCarrier") or "",
        "shipment_status": payload.get("shipmentStatus") or "",
        "estimated_delivery": payload.get("estimatedDelivery") or "",
        "purchase_date": payload.get("createdAt") or payload.get("purchaseDate") or "",
        "days_ago": payload.get("daysAgo"),
    }


def get_cart(cart_id: str) -> dict[str, Any] | None:
    """Get cart data by cart id using Storefront MCP tools."""
    if not cart_id:
        return None
    raw = _call_first_available_tool(
        ["get_cart", "fetch_cart", "cart_get"],
        {"cart_id": cart_id, "cartId": cart_id},
    )
    if not raw:
        return None
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw}


def add_to_cart(cart_id: str, variant_id: str, quantity: int = 1) -> dict[str, Any] | None:
    """Add a variant to cart using Storefront MCP tools."""
    if not variant_id:
        return None

    args = {
        "cart_id": cart_id,
        "cartId": cart_id,
        "variant_id": variant_id,
        "variantId": variant_id,
        "quantity": max(quantity, 1),
        "lines": [{"merchandiseId": variant_id, "quantity": max(quantity, 1)}],
    }
    raw = _call_first_available_tool(
        ["cart_add_items", "add_to_cart", "cart_lines_add"],
        args,
    )
    if not raw:
        return None
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw}


def update_cart_item(
    cart_id: str,
    quantity: int,
    *,
    line_id: str = "",
    variant_id: str = "",
) -> dict[str, Any] | None:
    """Update quantity for an existing cart line."""
    if not cart_id:
        return None

    args = {
        "cart_id": cart_id,
        "cartId": cart_id,
        "line_id": line_id,
        "lineId": line_id,
        "variant_id": variant_id,
        "variantId": variant_id,
        "quantity": max(quantity, 1),
        "lines": [
            {
                "id": line_id,
                "quantity": max(quantity, 1),
                "merchandiseId": variant_id,
            }
        ],
    }
    raw = _call_first_available_tool(
        ["cart_update_items", "update_cart_items", "cart_lines_update"],
        args,
    )
    if not raw:
        return None
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw}


def remove_from_cart(
    cart_id: str,
    *,
    line_id: str = "",
    variant_id: str = "",
) -> dict[str, Any] | None:
    """Remove item from cart by line id or variant id."""
    if not cart_id:
        return None

    args = {
        "cart_id": cart_id,
        "cartId": cart_id,
        "line_id": line_id,
        "lineId": line_id,
        "variant_id": variant_id,
        "variantId": variant_id,
        "line_ids": [line_id] if line_id else [],
        "lineIds": [line_id] if line_id else [],
    }
    raw = _call_first_available_tool(
        ["cart_remove_items", "remove_from_cart", "cart_lines_remove"],
        args,
    )
    if not raw:
        return None
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw}


def get_cart_checkout_url(cart_id: str) -> str:
    """Return checkout URL for cart if available."""
    if not cart_id:
        return ""

    cart_data = get_cart(cart_id)
    if isinstance(cart_data, dict):
        for key in ("checkoutUrl", "checkout_url", "checkout"):
            value = cart_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raw = _call_first_available_tool(
        ["get_checkout_url", "cart_get_checkout_url", "cart_checkout_url"],
        {"cart_id": cart_id, "cartId": cart_id},
    )
    if not raw:
        return ""
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        for key in ("checkoutUrl", "checkout_url", "url"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return raw.strip()


def format_cart_response(cart_payload: dict[str, Any] | None, action: str) -> str:
    """Format cart payload into a concise PT-BR response."""
    if not cart_payload:
        return (
            f"Nao consegui {action} o carrinho agora. "
            "Confirme o cart_id e tente novamente."
        )

    if isinstance(cart_payload.get("raw"), str):
        raw = str(cart_payload["raw"]).strip()
        if raw:
            return raw

    checkout_url = ""
    for key in ("checkoutUrl", "checkout_url"):
        value = cart_payload.get(key)
        if isinstance(value, str) and value.strip():
            checkout_url = value.strip()
            break

    lines = cart_payload.get("lines") or cart_payload.get("items") or []
    count = len(lines) if isinstance(lines, list) else None
    msg = f"Carrinho atualizado ({action}) com sucesso."
    if count is not None:
        msg += f" Itens no carrinho: {count}."
    if checkout_url:
        msg += f" Checkout: {checkout_url}"
    return msg


def get_order_status(
    order_number: str, customer_hint: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Lookup order via Customer Account MCP (authenticated)."""
    if not SUPPORTS_ORDER_LOOKUP:
        logger.info(
            "Order lookup requested for #%s but Customer MCP auth is not configured.",
            order_number,
        )
        return None

    sanitized = (order_number or "").strip().lstrip("#")
    if not sanitized:
        return None

    args = {
        "order_number": sanitized,
        "orderNumber": sanitized,
        "query": f"order {sanitized}",
        "customer_hint": customer_hint or "",
    }
    raw = _call_first_available_customer_tool(
        [
            "get_order_status",
            "get_order",
            "find_order",
            "search_customer_orders",
            "list_customer_orders",
        ],
        args,
    )
    if not raw:
        return None

    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("order"), dict):
            return _normalize_order_payload(parsed["order"], sanitized)
        return _normalize_order_payload(parsed, sanitized)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return _normalize_order_payload(parsed[0], sanitized)

    return {
        "number": sanitized,
        "status": "unknown",
        "status_label": "Consulta autenticada disponível no Customer MCP",
        "items": [],
        "total": "",
        "tracking": "",
        "tracking_url": "",
        "carrier": "",
        "shipment_status": "",
        "estimated_delivery": "",
        "purchase_date": "",
        "days_ago": None,
    }


def get_customer_purchase_summary(
    message: str,
    *,
    email: str | None = None,
    phone: str | None = None,
    cpf: str | None = None,
) -> str | None:
    """Return compact purchase summary via Customer Account MCP."""
    if not SUPPORTS_ORDER_LOOKUP:
        return None

    query_parts = [message.strip()]
    if email:
        query_parts.append(f"email:{email}")
    if phone:
        query_parts.append(f"phone:{phone}")
    if cpf:
        query_parts.append(f"cpf:{cpf}")
    query = " ".join(part for part in query_parts if part)

    raw = _call_first_available_customer_tool(
        ["get_customer_purchase_summary", "list_customer_orders", "search_customer_orders"],
        {"query": query},
    )
    if not raw:
        return None

    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, dict):
        summary = parsed.get("summary") or parsed.get("text")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    return raw.strip() if raw.strip() else None


def format_order_response(order: dict[str, Any]) -> str:
    """Format an order dict as human-readable text with tracking metadata."""
    lines = [
        f"Pedido #{order.get('number', '?')} - {order.get('status_label', 'Status desconhecido')}",
    ]

    purchase_date = order.get("purchase_date")
    days_ago = order.get("days_ago")
    if purchase_date and days_ago is not None:
        lines.append(f"Comprado em: {purchase_date} ({days_ago} dias atras)")
    elif purchase_date:
        lines.append(f"Comprado em: {purchase_date}")

    items = order.get("items") or []
    if items:
        rendered_items: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or "Item"
            qty = item.get("quantity") or 1
            price = item.get("unit_price")
            part = f"{name} ({qty}x)"
            if price:
                part += f" - {price}"
            rendered_items.append(part)
        if rendered_items:
            lines.append(f"Itens: {'; '.join(rendered_items)}")

    if order.get("total"):
        lines.append(f"Total: {order['total']}")

    if order.get("tracking"):
        lines.append(f"Rastreio: {order['tracking']}")
    if order.get("tracking_url"):
        lines.append(f"Link de rastreio: {order['tracking_url']}")
    if order.get("carrier"):
        lines.append(f"Transportadora: {order['carrier']}")
    if order.get("shipment_status"):
        lines.append(f"Status de entrega: {order['shipment_status']}")
    if order.get("estimated_delivery"):
        lines.append(f"Previsao de entrega: {order['estimated_delivery']}")

    return "\n".join(lines)


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
