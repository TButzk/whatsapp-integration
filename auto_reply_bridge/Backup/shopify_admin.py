"""Shopify Admin API helpers for real customer order lookups.

This module is intentionally focused on order retrieval. Product search remains
handled by the Storefront MCP server in ``shopify_mcp.py``.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger("auto-reply-bridge.shopify-admin")

SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", "").strip()
SHOPIFY_ADMIN_TOKEN = os.getenv("SHOPIFY_ADMIN_TOKEN", "").strip()
SHOPIFY_ADMIN_API_VERSION = os.getenv("SHOPIFY_ADMIN_API_VERSION", "2025-01").strip()
SHOPIFY_ADMIN_TIMEOUT = int(os.getenv("SHOPIFY_ADMIN_TIMEOUT", "30"))
SHOPIFY_CURRENCY = os.getenv("SHOPIFY_CURRENCY", "BRL")


def _base_url() -> str:
    domain = SHOPIFY_STORE_DOMAIN.replace("https://", "").replace("http://", "").strip("/")
    return f"https://{domain}/admin/api/{SHOPIFY_ADMIN_API_VERSION}"


def _is_configured() -> bool:
    return bool(SHOPIFY_STORE_DOMAIN and SHOPIFY_ADMIN_TOKEN)


def _admin_get(path: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_configured():
        logger.warning(
            "Shopify Admin API is not configured. Set SHOPIFY_STORE_DOMAIN and SHOPIFY_ADMIN_TOKEN."
        )
        return None

    url = f"{_base_url()}/{path.lstrip('/')}"
    try:
        response = requests.get(
            url,
            headers={
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN,
            },
            params=params,
            timeout=SHOPIFY_ADMIN_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except requests.exceptions.RequestException as exc:
        logger.error("Shopify Admin API request failed | url=%s error=%s", url, exc)
        return None
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected Shopify Admin API client error: %s", exc)
        return None


def _parse_iso_datetime(iso_value: str | None) -> datetime | None:
    if not iso_value:
        return None
    try:
        normalized = iso_value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_pt_date(iso_value: str | None) -> str:
    dt = _parse_iso_datetime(iso_value)
    if not dt:
        return "desconhecida"
    return dt.astimezone(timezone.utc).strftime("%d/%m/%Y")


def _days_ago(iso_value: str | None) -> int | None:
    dt = _parse_iso_datetime(iso_value)
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    return max(delta.days, 0)


def _format_money(amount: Any, currency: str | None = None) -> str:
    if amount in (None, ""):
        return ""
    curr = (currency or SHOPIFY_CURRENCY or "BRL").upper()
    try:
        value = float(amount)
        int_part = f"{int(value):,}".replace(",", ".")
        dec_part = f"{value:.2f}".split(".")[1]
        return f"{curr} {int_part},{dec_part}"
    except (TypeError, ValueError):
        return str(amount)


def _format_financial_status(status: str | None) -> str:
    mapping = {
        "pending": "Pagamento pendente",
        "authorized": "Pagamento autorizado",
        "partially_paid": "Pagamento parcial",
        "paid": "Pagamento confirmado",
        "partially_refunded": "Parcialmente reembolsado",
        "refunded": "Reembolsado",
        "voided": "Cancelado",
    }
    key = (status or "").strip().lower()
    return mapping.get(key, "Status de pagamento desconhecido")


def _format_fulfillment_status(status: str | None) -> str:
    mapping = {
        "fulfilled": "Pedido enviado",
        "partial": "Envio parcial",
        "restocked": "Reposto em estoque",
        "on_hold": "Em espera",
        "scheduled": "Envio agendado",
        "unfulfilled": "Aguardando envio",
    }
    key = (status or "").strip().lower()
    if not key:
        return "Aguardando envio"
    return mapping.get(key, "Status de entrega desconhecido")


def _extract_tracking(order: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    fulfillments = order.get("fulfillments")
    if isinstance(fulfillments, list) and fulfillments:
        latest = fulfillments[-1] if isinstance(fulfillments[-1], dict) else {}
        tracking_number = latest.get("tracking_number")
        if not tracking_number:
            numbers = latest.get("tracking_numbers")
            if isinstance(numbers, list) and numbers:
                tracking_number = str(numbers[0])

        tracking_url = latest.get("tracking_url")
        if not tracking_url:
            urls = latest.get("tracking_urls")
            if isinstance(urls, list) and urls:
                tracking_url = str(urls[0])

        carrier = latest.get("tracking_company")
        shipment_status = latest.get("shipment_status")
        return tracking_number, tracking_url, carrier, shipment_status

    return None, None, None, None


def get_order_by_number(order_number: str, customer_hint: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Fetch and normalize an order by its visible number (for example #1001)."""
    normalized_number = order_number.strip().lstrip("#")
    if not normalized_number:
        return None

    payload = _admin_get(
        "orders.json",
        {
            "name": f"#{normalized_number}",
            "status": "any",
            "limit": 3,
            "fields": "id,name,order_number,financial_status,fulfillment_status,created_at,total_price,current_total_price,currency,line_items,customer,fulfillments",
        },
    )
    if not payload:
        return None

    orders = payload.get("orders") or []
    if not isinstance(orders, list) or not orders:
        return None

    selected_order: dict[str, Any] | None = None
    if customer_hint:
        hint = customer_hint.lower().strip()
        for order in orders:
            customer = order.get("customer") or {}
            full_name = " ".join(
                str(v).strip()
                for v in (customer.get("first_name"), customer.get("last_name"))
                if v
            ).lower()
            if hint and full_name and hint in full_name:
                selected_order = order
                break

    if not selected_order:
        first = orders[0]
        if not isinstance(first, dict):
            return None
        selected_order = first

    if not isinstance(selected_order, dict):
        return None

    created_at = selected_order.get("created_at")
    currency = selected_order.get("currency") or SHOPIFY_CURRENCY

    items: list[dict[str, Any]] = []
    for line in selected_order.get("line_items") or []:
        if not isinstance(line, dict):
            continue
        unit_price = line.get("price")
        items.append(
            {
                "name": line.get("title") or "Item",
                "quantity": int(line.get("quantity") or 1),
                "unit_price": _format_money(unit_price, currency),
            }
        )

    total_amount = selected_order.get("current_total_price") or selected_order.get("total_price")
    tracking_number, tracking_url, carrier, shipment_status = _extract_tracking(selected_order)

    financial_label = _format_financial_status(selected_order.get("financial_status"))
    fulfillment_label = _format_fulfillment_status(selected_order.get("fulfillment_status"))

    return {
        "number": str(selected_order.get("order_number") or normalized_number),
        "status_label": f"{fulfillment_label} | {financial_label}",
        "financial_status": financial_label,
        "fulfillment_status": fulfillment_label,
        "created_at_iso": created_at,
        "purchase_date": _format_pt_date(created_at),
        "days_ago": _days_ago(created_at),
        "items": items,
        "total": _format_money(total_amount, currency),
        "tracking": tracking_number,
        "tracking_url": tracking_url,
        "carrier": carrier,
        "shipment_status": shipment_status,
        "estimated_delivery": None,
        "admin_order_url": (
            f"https://{SHOPIFY_STORE_DOMAIN.strip('/').replace('https://', '').replace('http://', '')}/admin/orders/"
            f"{quote(str(selected_order.get('id') or ''))}"
            if selected_order.get("id")
            else ""
        ),
    }


def _only_digits(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def _extract_cpf(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    match = re.search(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b", raw_text)
    if not match:
        return None
    digits = _only_digits(match.group(1))
    return digits if len(digits) == 11 else None


def _extract_product_keywords(message: str) -> list[str]:
    raw_words = re.findall(r"[a-zA-Z0-9áàâãéêíóôõúç]+", (message or "").lower())
    stopwords = {
        "quando",
        "comprei",
        "comprou",
        "produto",
        "pedido",
        "status",
        "rastreamento",
        "rastreio",
        "cliente",
        "pedido",
        "quero",
        "saber",
        "tempo",
        "meu",
        "minha",
        "meus",
        "minhas",
        "do",
        "da",
        "de",
        "que",
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "e",
    }
    terms: list[str] = []
    for word in raw_words:
        if len(word) < 3 or word in stopwords:
            continue
        if word not in terms:
            terms.append(word)
    return terms[:6]


def _order_matches_identity(
    order: dict[str, Any],
    email: str | None,
    phone: str | None,
    cpf_digits: str | None,
) -> bool:
    customer = order.get("customer") or {}
    order_email = str(customer.get("email") or order.get("email") or "").strip().lower()
    if email and order_email and email == order_email:
        return True

    phone_digits = _only_digits(phone)
    if phone_digits:
        candidates = [
            customer.get("phone"),
            order.get("phone"),
            (order.get("shipping_address") or {}).get("phone"),
            (order.get("billing_address") or {}).get("phone"),
        ]
        for candidate in candidates:
            cand_digits = _only_digits(str(candidate or ""))
            if cand_digits and (phone_digits.endswith(cand_digits[-8:]) or cand_digits.endswith(phone_digits[-8:])):
                return True

    if cpf_digits:
        note_attrs = order.get("note_attributes") or []
        for attr in note_attrs:
            if not isinstance(attr, dict):
                continue
            value_digits = _only_digits(str(attr.get("value") or ""))
            if value_digits and value_digits.endswith(cpf_digits[-6:]):
                return True
        for key in ("note", "tags"):
            value_digits = _only_digits(str(order.get(key) or ""))
            if value_digits and value_digits.endswith(cpf_digits[-6:]):
                return True

    return False


def _fetch_customer_orders(
    email: str | None,
    phone: str | None,
    cpf: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not any([email, phone, cpf]):
        return []

    params: dict[str, Any] = {
        "status": "any",
        "limit": min(max(limit, 5), 50),
        "order": "created_at desc",
        "fields": (
            "id,name,order_number,email,phone,financial_status,fulfillment_status,"
            "created_at,total_price,current_total_price,currency,line_items,customer,"
            "fulfillments,note_attributes,shipping_address,billing_address,tags,note"
        ),
    }
    if email:
        params["email"] = email

    payload = _admin_get("orders.json", params)
    if not payload:
        return []

    orders = payload.get("orders") or []
    if not isinstance(orders, list):
        return []

    normalized_email = (email or "").strip().lower() or None
    cpf_digits = _only_digits(cpf)
    filtered: list[dict[str, Any]] = []
    for raw in orders:
        if not isinstance(raw, dict):
            continue
        if _order_matches_identity(raw, normalized_email, phone, cpf_digits):
            filtered.append(raw)
    return filtered


def get_customer_purchase_summary(
    message: str,
    *,
    email: str | None = None,
    phone: str | None = None,
    cpf: str | None = None,
    max_orders: int = 25,
) -> str | None:
    """Return a summary of customer purchases, optionally focused on a product.

    This function is designed for support conversations where the customer does
    not provide an order number but asks about a specific product purchase.
    """
    normalized_email = (email or "").strip().lower() or None
    normalized_phone = (phone or "").strip() or None
    inferred_cpf = _extract_cpf(message)
    normalized_cpf = _only_digits(cpf or inferred_cpf)

    orders = _fetch_customer_orders(
        email=normalized_email,
        phone=normalized_phone,
        cpf=normalized_cpf,
        limit=max_orders,
    )
    if not orders:
        return None

    terms = _extract_product_keywords(message)
    if not terms:
        latest = orders[0]
        created_at = latest.get("created_at")
        date_label = _format_pt_date(created_at)
        days = _days_ago(created_at)
        financial = _format_financial_status(latest.get("financial_status"))
        fulfillment = _format_fulfillment_status(latest.get("fulfillment_status"))
        days_text = f" ({days} dias atras)" if days is not None else ""
        return (
            f"Encontrei pedidos no cadastro informado. O mais recente e o pedido "
            f"#{latest.get('order_number') or '?'} de {date_label}{days_text}, "
            f"com status: {fulfillment} | {financial}. "
            "Se voce quiser, me diga o nome do produto para eu verificar quando ele foi comprado."
        )

    matches: list[dict[str, Any]] = []
    for order in orders:
        order_items = order.get("line_items") or []
        for line in order_items:
            if not isinstance(line, dict):
                continue
            title = str(line.get("title") or "")
            title_low = title.lower()
            if any(term in title_low for term in terms):
                created_at = order.get("created_at")
                matches.append(
                    {
                        "order_number": str(order.get("order_number") or "?"),
                        "date": _format_pt_date(created_at),
                        "days_ago": _days_ago(created_at),
                        "title": title,
                        "quantity": int(line.get("quantity") or 1),
                        "financial": _format_financial_status(order.get("financial_status")),
                        "fulfillment": _format_fulfillment_status(order.get("fulfillment_status")),
                    }
                )

    if not matches:
        terms_text = " ".join(terms)
        return (
            f"Encontrei pedidos no cadastro, mas nao localizei compra do produto '{terms_text}' "
            "nos pedidos recentes."
        )

    lines = [
        f"Encontrei {len(matches)} compra(s) relacionada(s) a '{' '.join(terms)}':"
    ]
    for item in matches[:5]:
        days = item.get("days_ago")
        days_text = f" ({days} dias atras)" if days is not None else ""
        lines.append(
            f"- {item['date']}{days_text}: {item['title']} ({item['quantity']}x) "
            f"no pedido #{item['order_number']} - {item['fulfillment']} | {item['financial']}"
        )

    return "\n".join(lines)
