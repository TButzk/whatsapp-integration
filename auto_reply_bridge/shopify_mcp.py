"""LEGACY MODULE - non-operational stub.

This file intentionally keeps the old public API surface so accidental imports do
not crash immediately, but no Shopify MCP calls are performed here anymore.

New and current integrations must use local_ai_backend.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("auto-reply-bridge.shopify-legacy")

LEGACY_MESSAGE = (
    "Modulo legado desativado. Use local_ai_backend para integracoes Shopify."
)


def _warn(function_name: str) -> None:
    logger.warning("legacy_shopify_stub_called function=%s", function_name)


def is_order_lookup_available() -> bool:
    _warn("is_order_lookup_available")
    return False


def search_products(query: str, limit: int = 5) -> list[dict[str, Any]]:
    _warn("search_products")
    return []


def search_policies(query: str) -> str:
    _warn("search_policies")
    return ""


def list_available_resources() -> str:
    _warn("list_available_resources")
    return LEGACY_MESSAGE


def get_cart(cart_id: str) -> dict[str, Any] | None:
    _warn("get_cart")
    return None


def add_to_cart(cart_id: str, variant_id: str, quantity: int = 1) -> dict[str, Any] | None:
    _warn("add_to_cart")
    return None


def update_cart_item(
    cart_id: str,
    quantity: int,
    *,
    line_id: str = "",
    variant_id: str = "",
) -> dict[str, Any] | None:
    _warn("update_cart_item")
    return None


def remove_from_cart(
    cart_id: str,
    *,
    line_id: str = "",
    variant_id: str = "",
) -> dict[str, Any] | None:
    _warn("remove_from_cart")
    return None


def get_cart_checkout_url(cart_id: str) -> str:
    _warn("get_cart_checkout_url")
    return ""


def format_cart_response(cart_payload: dict[str, Any] | None, action: str) -> str:
    _warn("format_cart_response")
    return (
        f"{LEGACY_MESSAGE} Acao solicitada: {action}. "
        "Migrar para local_ai_backend."
    )


def get_order_status(
    order_number: str,
    customer_hint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    _warn("get_order_status")
    return None


def get_customer_purchase_summary(
    message: str,
    *,
    email: str | None = None,
    phone: str | None = None,
    cpf: str | None = None,
) -> str | None:
    _warn("get_customer_purchase_summary")
    return None


def format_order_response(order: dict[str, Any]) -> str:
    _warn("format_order_response")
    return LEGACY_MESSAGE


def format_products_response(products: list[dict[str, Any]], query: str) -> str:
    _warn("format_products_response")
    return (
        f"{LEGACY_MESSAGE} Busca de produtos para '{query}' deve ser feita em local_ai_backend."
    )
