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


def _role_label(msg: dict[str, Any]) -> Optional[str]:
    """Return a human-readable role label for a Chatwoot message."""
    msg_type = msg.get("message_type")
    if msg_type == "incoming":
        return "Cliente"
    if msg_type == "outgoing":
        sender = msg.get("sender") or {}
        if sender.get("type") == "agent_bot":
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


def fetch_history(account_id: int, conversation_id: int) -> list[dict[str, str]]:
    """Fetch and filter recent messages from a Chatwoot conversation.

    Returns an empty list on any error (graceful fallback).
    """
    if not CHAT_HISTORY_ENABLED:
        return []

    try:
        response = requests.get(
            (
                f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}"
                f"/conversations/{conversation_id}/messages"
            ),
            headers={"api_access_token": CHATWOOT_API_TOKEN},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        messages: list[dict[str, Any]] = data.get("payload") or []
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
