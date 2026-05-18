from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import secrets
import re
import uuid
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from local_ai_backend.config import (
    Settings,
    get_effective_shopify_customer_timeout,
    is_shopify_customer_lookup_enabled,
)


@dataclass(slots=True)
class IntegrationResult:
    status: str
    data: dict[str, Any] | None = None


class ShopifyIntegration:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._customer_initialized = False
        self._customer_tools_cache: list[str] | None = None

    def _not_configured(self) -> IntegrationResult:
        return IntegrationResult(status="integration_not_configured")

    def is_customer_lookup_available(self) -> bool:
        if not is_shopify_customer_lookup_enabled(self._settings):
            return False
        if not self._settings.shopify_customer_mcp_endpoint.strip():
            return False
        return self._ensure_customer_token()

    def _customer_headers(self) -> dict[str, str]:
        token = self._settings.shopify_customer_mcp_token.strip()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _infer_customer_oauth_token_endpoint(self) -> str:
        explicit = self._settings.shopify_customer_oauth_token_endpoint.strip()
        if explicit:
            return explicit

        endpoint = self._settings.shopify_customer_mcp_endpoint.strip()
        if not endpoint:
            return ""
        parsed = urlparse(endpoint)
        match = re.search(r"/(\d+)/account/customer/api/mcp", parsed.path)
        if not match:
            return ""
        shop_id = match.group(1)
        return f"{parsed.scheme}://{parsed.netloc}/authentication/{shop_id}/oauth/token"

    def _infer_customer_oauth_authorization_endpoint(self) -> str:
        explicit = self._settings.shopify_customer_oauth_authorization_endpoint.strip()
        if explicit:
            return explicit

        token_endpoint = self._infer_customer_oauth_token_endpoint()
        if token_endpoint.endswith("/oauth/token"):
            return token_endpoint.replace("/oauth/token", "/oauth/authorize")
        return ""

    def create_customer_oauth_authorization(
        self,
        *,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> IntegrationResult:
        auth_endpoint = self._infer_customer_oauth_authorization_endpoint()
        client_id = self._settings.shopify_customer_client_id.strip()
        effective_redirect = (redirect_uri or self._settings.shopify_customer_redirect_uri).strip()

        if not (auth_endpoint and client_id and effective_redirect):
            return IntegrationResult(status="invalid_input")

        oauth_state = (state or secrets.token_urlsafe(18)).strip()
        use_pkce = bool(self._settings.shopify_customer_oauth_use_pkce)
        code_verifier: str | None = None
        code_challenge: str | None = None

        if use_pkce:
            code_verifier = secrets.token_urlsafe(64)
            digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
            code_challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": effective_redirect,
            "scope": self._settings.shopify_customer_scopes.strip(),
            "state": oauth_state,
        }
        if use_pkce and code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        auth_url = f"{auth_endpoint}?{urlencode(params)}"
        return IntegrationResult(
            status="ok",
            data={
                "authorization_url": auth_url,
                "state": oauth_state,
                "redirect_uri": effective_redirect,
                "code_verifier": code_verifier,
                "uses_pkce": use_pkce,
            },
        )

    def exchange_customer_oauth_code(
        self,
        *,
        code: str,
        redirect_uri: str | None = None,
        code_verifier: str | None = None,
    ) -> IntegrationResult:
        token_endpoint = self._infer_customer_oauth_token_endpoint()
        client_id = self._settings.shopify_customer_client_id.strip()
        client_secret = self._settings.shopify_customer_client_secret.strip()
        effective_redirect = (redirect_uri or self._settings.shopify_customer_redirect_uri).strip()

        if not (token_endpoint and client_id and code.strip() and effective_redirect):
            return IntegrationResult(status="invalid_input")

        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code.strip(),
            "client_id": client_id,
            "redirect_uri": effective_redirect,
        }
        if code_verifier:
            data["code_verifier"] = code_verifier.strip()

        timeout = float(get_effective_shopify_customer_timeout(self._settings))
        try:
            with httpx.Client(timeout=timeout) as client:
                kwargs: dict[str, Any] = {"data": data}
                if client_secret:
                    kwargs["auth"] = (client_id, client_secret)
                response = client.post(token_endpoint, **kwargs)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return IntegrationResult(status="request_failed", data={"error": str(exc)[:300]})

        if not isinstance(payload, dict):
            return IntegrationResult(status="request_failed")

        access_token = str(payload.get("access_token") or "").strip()
        refresh_token = str(payload.get("refresh_token") or "").strip()

        if access_token:
            self._settings.shopify_customer_mcp_token = access_token
        if refresh_token:
            self._settings.shopify_customer_refresh_token = refresh_token

        return IntegrationResult(
            status="ok" if access_token else "request_failed",
            data={
                "access_token": access_token or None,
                "refresh_token": refresh_token or None,
                "token_type": payload.get("token_type"),
                "expires_in": payload.get("expires_in"),
                "scope": payload.get("scope"),
            },
        )

    def _refresh_customer_access_token(self) -> bool:
        token_endpoint = self._infer_customer_oauth_token_endpoint()
        if not token_endpoint:
            return False

        client_id = self._settings.shopify_customer_client_id.strip()
        client_secret = self._settings.shopify_customer_client_secret.strip()
        refresh_token = self._settings.shopify_customer_refresh_token.strip()
        scopes = self._settings.shopify_customer_scopes.strip()

        if not (client_id and client_secret and refresh_token):
            return False

        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        if scopes:
            data["scope"] = scopes

        timeout = float(get_effective_shopify_customer_timeout(self._settings))

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    token_endpoint,
                    data=data,
                    auth=(client_id, client_secret),
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            return False

        self._settings.shopify_customer_mcp_token = access_token
        return True

    def _ensure_customer_token(self) -> bool:
        if self._settings.shopify_customer_mcp_token.strip():
            return True
        return self._refresh_customer_access_token()

    def _customer_mcp_post(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        timeout = float(get_effective_shopify_customer_timeout(self._settings))

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    self._settings.shopify_customer_mcp_endpoint.strip(),
                    headers=self._customer_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return None

        if not isinstance(data, dict):
            return None
        if "error" in data:
            return None
        return data.get("result")

    def _initialize_customer(self) -> bool:
        if self._customer_initialized:
            return True
        if not self.is_customer_lookup_available():
            return False

        result = self._customer_mcp_post(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "local-ai-backend", "version": "0.1.0"},
            },
        )
        self._customer_initialized = result is not None
        return self._customer_initialized

    def _list_customer_tools(self) -> list[str]:
        if self._customer_tools_cache is not None:
            return self._customer_tools_cache
        if not self._initialize_customer():
            self._customer_tools_cache = []
            return self._customer_tools_cache

        result = self._customer_mcp_post("tools/list", {})
        names: list[str] = []
        if isinstance(result, dict):
            tools = result.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                        names.append(tool["name"])
        self._customer_tools_cache = names
        return names

    def _call_customer_tool(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        result = self._customer_mcp_post("tools/call", {"name": tool_name, "arguments": arguments})
        if not isinstance(result, dict):
            return None

        content_blocks = result.get("content")
        if not isinstance(content_blocks, list):
            return None

        texts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if not texts:
            return None
        return "\n".join(texts)

    def _call_first_available_customer_tool(
        self,
        candidate_names: list[str],
        arguments: dict[str, Any],
    ) -> str | None:
        available = set(self._list_customer_tools())
        if not available:
            return None
        for tool_name in candidate_names:
            if tool_name in available:
                return self._call_customer_tool(tool_name, arguments)
        return None

    def _extract_json_from_text(self, raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _coerce_order_candidates(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        for key in ("orders", "results", "items", "matches"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        if any(k in payload for k in ("order_number", "number", "id", "status")):
            return [payload]
        return []

    def _pick_str(self, order: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = order.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _format_order_summary(self, order: dict[str, Any]) -> str:
        number = self._pick_str(order, "number", "order_number", "name", "orderName", "id")
        status = self._pick_str(
            order,
            "status_label",
            "status",
            "display_status",
            "financial_status",
            "fulfillment_status",
        )
        tracking = self._pick_str(order, "tracking", "tracking_number", "tracking_code")
        tracking_url = self._pick_str(order, "tracking_url", "trackingUrl")
        purchase_date = self._pick_str(order, "purchase_date", "created_at", "createdAt")
        total = self._pick_str(order, "total", "current_total_price", "total_price")

        lines = [
            f"Pedido: #{number}" if number else "Pedido localizado",
            f"Status: {status}" if status else "Status: indisponivel",
        ]
        if purchase_date:
            lines.append(f"Data: {purchase_date}")
        if total:
            lines.append(f"Total: {total}")
        if tracking:
            lines.append(f"Rastreio: {tracking}")
        if tracking_url:
            lines.append(f"Link de rastreio: {tracking_url}")
        return "\n".join(lines)

    def _format_ambiguous_summary(self, candidates: list[dict[str, Any]]) -> str:
        lines = ["Encontrei mais de um pedido com esses dados:"]
        for idx, order in enumerate(candidates[:3], start=1):
            number = self._pick_str(order, "number", "order_number", "name", "id")
            date = self._pick_str(order, "purchase_date", "created_at", "createdAt")
            status = self._pick_str(order, "status_label", "status", "fulfillment_status")
            chunk = f"{idx}. #{number}" if number else f"{idx}. Pedido sem numero"
            if date:
                chunk += f" | {date}"
            if status:
                chunk += f" | {status}"
            lines.append(chunk)
        return "\n".join(lines)

    def get_order_status(self, order_number: str) -> IntegrationResult:
        if not self.is_customer_lookup_available():
            return self._not_configured()

        sanitized = (order_number or "").strip().lstrip("#")
        if not sanitized:
            return IntegrationResult(status="invalid_input")

        raw = self._call_first_available_customer_tool(
            ["get_order_status", "get_order", "find_order", "search_customer_orders"],
            {"query": f"order:{sanitized}"},
        )
        if not raw:
            return IntegrationResult(status="not_found")

        parsed = self._extract_json_from_text(raw)
        candidates = self._coerce_order_candidates(parsed) if parsed is not None else []

        if len(candidates) > 1:
            return IntegrationResult(
                status="ambiguous",
                data={
                    "count": len(candidates),
                    "summary": self._format_ambiguous_summary(candidates),
                    "payload": parsed,
                    "raw": raw,
                },
            )

        if len(candidates) == 1:
            summary = self._format_order_summary(candidates[0])
            return IntegrationResult(
                status="ok",
                data={"summary": summary, "payload": candidates[0], "raw": raw},
            )

        if isinstance(parsed, dict):
            return IntegrationResult(status="ok", data={"raw": raw, "payload": parsed})
        return IntegrationResult(status="ok", data={"raw": raw})

    def get_customer_purchase_summary(
        self,
        message: str,
        *,
        email: str | None = None,
        phone: str | None = None,
        cpf: str | None = None,
    ) -> IntegrationResult:
        if not self.is_customer_lookup_available():
            return self._not_configured()

        query_parts = [(message or "").strip()]
        if email:
            query_parts.append(f"email:{email}")
        if phone:
            query_parts.append(f"phone:{phone}")
        if cpf:
            query_parts.append(f"cpf:{cpf}")
        query = " ".join(part for part in query_parts if part)
        if not query:
            return IntegrationResult(status="invalid_input")

        raw = self._call_first_available_customer_tool(
            ["get_customer_purchase_summary", "list_customer_orders", "search_customer_orders"],
            {"query": query},
        )
        if not raw:
            return IntegrationResult(status="not_found")

        parsed = self._extract_json_from_text(raw)
        candidates = self._coerce_order_candidates(parsed) if parsed is not None else []

        if len(candidates) > 1:
            return IntegrationResult(
                status="ambiguous",
                data={
                    "count": len(candidates),
                    "summary": self._format_ambiguous_summary(candidates),
                    "payload": parsed,
                    "raw": raw,
                    "query": query,
                },
            )

        if len(candidates) == 1:
            return IntegrationResult(
                status="ok",
                data={
                    "summary": self._format_order_summary(candidates[0]),
                    "payload": candidates[0],
                    "raw": raw,
                    "query": query,
                },
            )

        if isinstance(parsed, dict):
            summary = parsed.get("summary") or parsed.get("text")
            if isinstance(summary, str) and summary.strip():
                return IntegrationResult(
                    status="ok",
                    data={"summary": summary.strip(), "raw": raw, "query": query},
                )
            return IntegrationResult(status="ok", data={"payload": parsed, "raw": raw, "query": query})
        return IntegrationResult(status="ok", data={"summary": raw.strip(), "query": query})

    def get_fulfillment_tracking(self, order_number: str) -> IntegrationResult:
        return self._not_configured()

    def search_products(self, query: str) -> IntegrationResult:
        return self._not_configured()

    def get_product_price(self, product_id: str) -> IntegrationResult:
        return self._not_configured()

    def create_cart_link(self, items: list[dict[str, Any]]) -> IntegrationResult:
        return self._not_configured()