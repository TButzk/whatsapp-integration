"""Fase 1: Histórico de conversa do Chatwoot.

Busca as mensagens recentes de uma conversa e as formata
como contexto compacto para o modelo.
"""

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger("auto-reply-bridge.history")

CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "http://localhost:65271").rstrip("/")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHAT_HISTORY_ENABLED = os.getenv("CHAT_HISTORY_ENABLED", "true").lower() == "true"
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "10"))
CHAT_HISTORY_MAX_CHARS = int(os.getenv("CHAT_HISTORY_MAX_CHARS", "3000"))
CHAT_HISTORY_INCLUDE_AGENT = os.getenv("CHAT_HISTORY_INCLUDE_AGENT", "true").lower() == "true"
CHAT_HISTORY_REQUEST_TIMEOUT = int(os.getenv("CHAT_HISTORY_REQUEST_TIMEOUT", "15"))


def _normalize_message_type(msg_type: Any) -> Optional[str]:
    """Normalize Chatwoot message type values to incoming/outgoing labels.

    Chatwoot may return message_type as string ("incoming"/"outgoing")
    or as integers (0 incoming, 1 outgoing).
    """
    if isinstance(msg_type, str):
        lowered = msg_type.lower().strip()
        if lowered in {"incoming", "outgoing"}:
            return lowered
        if lowered.isdigit():
            try:
                msg_type = int(lowered)
            except ValueError:
                return None
        else:
            return None

    if isinstance(msg_type, int):
        if msg_type == 0:
            return "incoming"
        if msg_type == 1:
            return "outgoing"

    return None


def _role_label(msg: dict[str, Any]) -> Optional[str]:
    """Return a human-readable role label for a Chatwoot message."""
    normalized_type = _normalize_message_type(msg.get("message_type"))

    if normalized_type == "incoming":
        return "Cliente"

    if normalized_type == "outgoing":
        sender = msg.get("sender") or {}
        sender_type = (
            sender.get("type")
            or msg.get("sender_type")
            or ""
        )
        if isinstance(sender_type, str) and sender_type.lower() == "agent_bot":
            return "Bot"
        return "Atendente"

    return None


def _filter_and_format(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Filter irrelevant messages and return formatted history list."""
    formatted: list[dict[str, str]] = []
    seen_contents: set[str] = set()

    messages_sorted = sorted(messages, key=lambda m: m.get("created_at", 0))

    for msg in messages_sorted:
        if msg.get("private"):
            continue

        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if content in seen_contents:
            continue
        seen_contents.add(content)

        role = _role_label(msg)
        if role is None:
            continue

        if not CHAT_HISTORY_INCLUDE_AGENT and role == "Atendente":
            continue

        formatted.append({"role": role, "content": content})

    # Keep only the most recent N messages
    if len(formatted) > CHAT_HISTORY_MAX_MESSAGES:
        formatted = formatted[-CHAT_HISTORY_MAX_MESSAGES:]

    # Trim from oldest until total chars fit
    total_chars = 0
    result: list[dict[str, str]] = []
    for msg in reversed(formatted):
        entry_len = len(msg["content"]) + len(msg["role"]) + 5
        if total_chars + entry_len > CHAT_HISTORY_MAX_CHARS:
            break
        total_chars += entry_len
        result.insert(0, msg)

    return result


def _extract_messages(data: Any) -> list[dict[str, Any]]:
    """Extract Chatwoot messages from known response shapes."""
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]

    if not isinstance(data, dict):
        return []

    candidates: list[Any] = [
        data.get("payload"),
        data.get("messages"),
        (data.get("data") or {}).get("payload") if isinstance(data.get("data"), dict) else None,
        (data.get("data") or {}).get("messages") if isinstance(data.get("data"), dict) else None,
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [m for m in candidate if isinstance(m, dict)]

    return []


def fetch_history(account_id: int, conversation_id: int) -> list[dict[str, str]]:
    """Fetch and filter recent messages from a Chatwoot conversation.

    Returns an empty list on any error (graceful fallback).
    """
    if not CHAT_HISTORY_ENABLED:
        return []

    try:
        headers = {"api_access_token": CHATWOOT_API_TOKEN}
        if CHATWOOT_API_TOKEN:
            headers["Authorization"] = f"Bearer {CHATWOOT_API_TOKEN}"

        response = requests.get(
            (
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}"
                f"/conversations/{conversation_id}/messages"
            ),
            headers=headers,
            timeout=CHAT_HISTORY_REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            body_preview = (response.text or "").strip().replace("\n", " ")[:300]
            logger.warning(
                "History HTTP error | status=%s account=%s conversation=%s body=%s",
                response.status_code,
                account_id,
                conversation_id,
                body_preview,
            )
        response.raise_for_status()
        data = response.json()
        messages = _extract_messages(data)
        if not messages:
            logger.info(
                "History empty | account=%s conversation=%s response_keys=%s",
                account_id,
                conversation_id,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
        return _filter_and_format(messages)
    except Exception as exc:
        logger.warning("Failed to fetch conversation history: %s", exc)
        return []


def format_history_context(history: list[dict[str, str]]) -> str:
    """Format history list as a compact plain-text block."""
    if not history:
        return ""
    lines = [f"{msg['role']}: {msg['content']}" for msg in history]
    return "\n".join(lines)
