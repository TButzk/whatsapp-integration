import hashlib
import hmac
import logging
import os
import queue
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

import chat_history
import intent as intent_mod
import shopify_mock
from intent import Intent
from rag import format_rag_context, search_documents


load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("auto-reply-bridge")

app = Flask(__name__)
job_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
processed_deliveries: dict[str, float] = {}
processed_lock = threading.Lock()

CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "http://localhost:65271").rstrip("/")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_WEBHOOK_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3:medium")
OLLAMA_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
    "Voce e um atendente virtual objetivo e educado. Responda em portugues do Brasil.",
)
OLLAMA_MAIN_TIMEOUT = int(os.getenv("OLLAMA_MAIN_TIMEOUT", "90"))
OLLAMA_FALLBACK_TIMEOUT = int(os.getenv("OLLAMA_FALLBACK_TIMEOUT", "60"))
# Keep backward-compatible alias
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", str(OLLAMA_MAIN_TIMEOUT)))
MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "1200"))
IGNORE_BOT_PREFIX = os.getenv("IGNORE_BOT_PREFIX", "!botoff")


def verify_signature(raw_body: bytes) -> bool:
    if not CHATWOOT_WEBHOOK_SECRET:
        return True

    timestamp = request.headers.get("X-Chatwoot-Timestamp", "")
    signature = request.headers.get("X-Chatwoot-Signature", "")
    if not timestamp or not signature:
        return False

    try:
        request_age = abs(int(time.time()) - int(timestamp))
    except ValueError:
        return False

    if request_age > 300:
        return False

    message = f"{timestamp}.".encode() + raw_body
    expected = "sha256=" + hmac.new(
        CHATWOOT_WEBHOOK_SECRET.encode(), message, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def is_duplicate(delivery_id: str) -> bool:
    now = time.time()
    with processed_lock:
        expired_keys = [key for key, seen_at in processed_deliveries.items() if now - seen_at > 900]
        for key in expired_keys:
            processed_deliveries.pop(key, None)

        if delivery_id in processed_deliveries:
            return True

        processed_deliveries[delivery_id] = now
        return False


def should_reply(payload: dict[str, Any]) -> tuple[bool, str]:
    if payload.get("event") != "message_created":
        return False, "unsupported_event"

    if payload.get("message_type") != "incoming":
        return False, "not_incoming"

    if payload.get("private"):
        return False, "private_note"

    content = (payload.get("content") or "").strip()
    if not content:
        return False, "empty_content"

    if content.startswith(IGNORE_BOT_PREFIX):
        return False, "bot_disabled_by_prefix"

    conversation = payload.get("conversation") or {}
    if conversation.get("status") == "resolved":
        return False, "resolved_conversation"

    sender = payload.get("sender") or {}
    if sender.get("type") == "user":
        return False, "agent_message"

    return True, "ok"


def _call_ollama(messages: list[dict[str, str]], model: str, timeout: int) -> str:
    """Call the Ollama chat API and return the response text."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "stream": False, "messages": messages},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("Ollama returned an empty response")
    return content[:MAX_RESPONSE_CHARS]


def build_prompt(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Phase 4 orchestration: build the full prompt for the LLM.

    1. Detect intent
    2. Collect conversation history
    3. Consult Shopify mock (ORDER / PRODUCT intents)
    4. Consult RAG (INSTITUTIONAL intent)
    5. Compose final messages list
    """
    content = (payload.get("content") or "").strip()
    contact = payload.get("contact") or {}
    conversation = payload.get("conversation") or {}
    account = payload.get("account") or {}
    contact_name = contact.get("name") or "cliente"
    channel = conversation.get("channel") or "whatsapp"
    account_id = account.get("id")
    conversation_id = conversation.get("id")

    t0 = time.time()

    # ---- 1. Detect intent -----------------------------------------------
    detected_intent = intent_mod.detect_intent(content)

    # ---- 2. Conversation history ----------------------------------------
    history: list[dict[str, str]] = []
    if account_id and conversation_id:
        try:
            history = chat_history.fetch_history(
                int(account_id), int(conversation_id)
            )
        except Exception as exc:
            logger.warning("History fetch failed: %s", exc)

    history_context = chat_history.format_history_context(history)

    # ---- 3. Tool context (Shopify / RAG) --------------------------------
    tool_context = ""

    if detected_intent == Intent.ORDER:
        order_number = intent_mod.extract_order_number(content)
        if order_number:
            order = shopify_mock.get_order_status(order_number)
            if order:
                tool_context = shopify_mock.format_order_response(order)
            else:
                tool_context = (
                    f"Nenhum pedido encontrado com o número #{order_number}. "
                    "Verifique se o número está correto."
                )
        else:
            tool_context = (
                "Para consultar um pedido, por favor informe o número do pedido."
            )

    elif detected_intent == Intent.PRODUCT:
        products = shopify_mock.search_products(content, limit=3)
        tool_context = shopify_mock.format_products_response(products, content)

    elif detected_intent == Intent.INSTITUTIONAL:
        try:
            chunks = search_documents(content)
            tool_context = format_rag_context(chunks)
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)

    # ---- 4. Build user message ------------------------------------------
    sections: list[str] = [
        f"Canal: {channel}",
        f"Cliente: {contact_name}",
    ]

    if history_context:
        sections.append(f"Histórico recente da conversa:\n{history_context}")

    if tool_context:
        sections.append(f"Dados disponíveis para esta resposta:\n{tool_context}")

    sections.append(f"Mensagem atual do cliente: {content}")
    sections.append(
        "Responda de forma objetiva e útil. "
        "Se faltar contexto, faça uma pergunta curta. "
        "Não invente dados que não foram fornecidos acima."
    )

    user_message = "\n\n".join(sections)

    # ---- 5. Metrics log --------------------------------------------------
    prompt_chars = len(OLLAMA_SYSTEM_PROMPT) + len(user_message)
    logger.info(
        "Prompt built | intent=%s history_msgs=%d rag_chunks=%d prompt_chars=%d elapsed=%.2fs",
        detected_intent,
        len(history),
        len(tool_context.split("---")) if tool_context else 0,
        prompt_chars,
        time.time() - t0,
    )

    return [
        {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_reply(payload: dict[str, Any]) -> str:
    """Generate a reply, falling back to OLLAMA_FALLBACK_MODEL on error."""
    messages = build_prompt(payload)
    t0 = time.time()

    try:
        reply = _call_ollama(messages, OLLAMA_MODEL, OLLAMA_MAIN_TIMEOUT)
        logger.info(
            "Reply generated | model=%s total_time=%.2fs", OLLAMA_MODEL, time.time() - t0
        )
        return reply
    except Exception as primary_exc:
        logger.warning(
            "Primary model %s failed (%s); trying fallback %s",
            OLLAMA_MODEL,
            primary_exc,
            OLLAMA_FALLBACK_MODEL,
        )

    reply = _call_ollama(messages, OLLAMA_FALLBACK_MODEL, OLLAMA_FALLBACK_TIMEOUT)
    logger.info(
        "Reply generated | model=%s (fallback) total_time=%.2fs",
        OLLAMA_FALLBACK_MODEL,
        time.time() - t0,
    )
    return reply


def send_chatwoot_message(payload: dict[str, Any], reply_text: str) -> None:
    account = payload.get("account") or {}
    conversation = payload.get("conversation") or {}
    account_id = account.get("id")
    conversation_id = conversation.get("id")

    if not account_id or not conversation_id:
        raise ValueError("Webhook payload missing account or conversation identifiers")

    if not CHATWOOT_API_TOKEN:
        raise ValueError("CHATWOOT_API_TOKEN is not configured")

    response = requests.post(
        (
            f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/messages"
        ),
        headers={
            "Content-Type": "application/json",
            "api_access_token": CHATWOOT_API_TOKEN,
        },
        json={
            "content": reply_text,
            "message_type": "outgoing",
            "private": False,
            "content_type": "text",
            "content_attributes": {},
        },
        timeout=30,
    )
    response.raise_for_status()


def worker() -> None:
    while True:
        payload = job_queue.get()
        try:
            reply_text = generate_reply(payload)
            send_chatwoot_message(payload, reply_text)
            logger.info("Auto-reply sent for conversation %s", (payload.get("conversation") or {}).get("id"))
        except Exception:
            logger.exception("Failed to process auto-reply job")
        finally:
            job_queue.task_done()


@app.get("/healthz")
def healthz() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.post("/webhook")
def webhook() -> tuple[Response, int] | Response:
    raw_body = request.get_data()
    if not verify_signature(raw_body):
        return jsonify({"status": "invalid_signature"}), 401

    payload = request.get_json(silent=True) or {}
    should_process, reason = should_reply(payload)
    if not should_process:
        return jsonify({"status": "ignored", "reason": reason})

    delivery_id = request.headers.get("X-Chatwoot-Delivery") or str(payload.get("id") or time.time())
    if is_duplicate(delivery_id):
        return jsonify({"status": "duplicate"})

    job_queue.put(payload)
    return jsonify({"status": "accepted"}), 202


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=8000)