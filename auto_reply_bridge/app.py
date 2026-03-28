import hashlib
import hmac
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

_DOTENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

import chat_history
import importlib
import intent as intent_mod
from intent import Intent
from rag import format_rag_context, ingest_documents, search_documents

# ---------------------------------------------------------------------------
# Shopify module — selected at startup based on SHOPIFY_MODE env variable.
# Supported values:
#   mock  (default) — local mock data, no network calls
#   mcp             — Shopify Storefront MCP Server (requires SHOPIFY_MCP_ENDPOINT
#                     and SHOPIFY_STOREFRONT_TOKEN)
# ---------------------------------------------------------------------------
_SHOPIFY_MODE = os.getenv("SHOPIFY_MODE", "mock").lower()
if _SHOPIFY_MODE not in ("mock", "mcp"):
    import warnings
    warnings.warn(
        f"Unknown SHOPIFY_MODE={_SHOPIFY_MODE!r}; falling back to 'mock'.",
        stacklevel=1,
    )
    _SHOPIFY_MODE = "mock"
_shopify = importlib.import_module("shopify_mcp" if _SHOPIFY_MODE == "mcp" else "shopify_mock")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("auto-reply-bridge")

app = Flask(__name__)
job_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
processed_deliveries: dict[str, float] = {}
processed_lock = threading.Lock()


def _read_bounded_int_env(
    var_name: str, default: int, min_value: int, max_value: int
) -> int:
    value = os.getenv(var_name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default=%s", var_name, value, default)
        return default

    if parsed < min_value:
        logger.warning("%s=%s below minimum %s; clamping", var_name, parsed, min_value)
        return min_value
    if parsed > max_value:
        logger.warning("%s=%s above maximum %s; clamping", var_name, parsed, max_value)
        return max_value
    return parsed

CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "http://localhost:65271").rstrip("/")
CHATWOOT_API_TOKEN = os.getenv("CHATWOOT_API_TOKEN", "")
CHATWOOT_WEBHOOK_SECRET = os.getenv("CHATWOOT_WEBHOOK_SECRET", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b-it-q4_K_M")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3:medium")
OLLAMA_SYSTEM_PROMPT = os.getenv(
    "OLLAMA_SYSTEM_PROMPT",
    "Voce e um atendente virtual objetivo e educado. Responda em portugues do Brasil.",
)
OLLAMA_TIMEOUT_MAX_SECONDS = _read_bounded_int_env(
    "OLLAMA_TIMEOUT_MAX_SECONDS", default=300, min_value=30, max_value=1800
)
OLLAMA_MAIN_TIMEOUT = _read_bounded_int_env(
    "OLLAMA_MAIN_TIMEOUT", default=90, min_value=5, max_value=OLLAMA_TIMEOUT_MAX_SECONDS
)
OLLAMA_FALLBACK_TIMEOUT = _read_bounded_int_env(
    "OLLAMA_FALLBACK_TIMEOUT", default=60, min_value=5, max_value=OLLAMA_TIMEOUT_MAX_SECONDS
)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_UNAVAILABLE_MESSAGE = os.getenv(
    "OLLAMA_UNAVAILABLE_MESSAGE",
    "No momento estou com instabilidade para responder. Pode tentar novamente em instantes?",
)
# Keep backward-compatible alias
OLLAMA_REQUEST_TIMEOUT = _read_bounded_int_env(
    "OLLAMA_REQUEST_TIMEOUT",
    default=OLLAMA_MAIN_TIMEOUT,
    min_value=5,
    max_value=OLLAMA_TIMEOUT_MAX_SECONDS,
)
MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "1200"))
IGNORE_BOT_PREFIX = os.getenv("IGNORE_BOT_PREFIX", "!botoff")


def _read_optional_int_env(var_name: str) -> int | None:
    value = os.getenv(var_name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; ignoring", var_name, value)
        return None


OLLAMA_NUM_GPU = _read_optional_int_env("OLLAMA_NUM_GPU")

logger.info(
    "Ollama settings | primary=%s fallback=%s keep_alive=%s main_timeout=%ss fallback_timeout=%ss timeout_cap=%ss num_gpu=%s",
    OLLAMA_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAIN_TIMEOUT,
    OLLAMA_FALLBACK_TIMEOUT,
    OLLAMA_TIMEOUT_MAX_SECONDS,
    OLLAMA_NUM_GPU if OLLAMA_NUM_GPU is not None else "default",
)


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
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if OLLAMA_NUM_GPU is not None:
        # num_gpu controls how many layers should be offloaded to GPU in Ollama.
        payload["options"] = {"num_gpu": OLLAMA_NUM_GPU}

    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        body_preview = (response.text or "").strip().replace("\n", " ")[:300]
        logger.warning(
            "Ollama HTTP error | model=%s status=%s url=%s body=%s",
            model,
            response.status_code,
            url,
            body_preview,
        )
    response.raise_for_status()
    data = response.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("Ollama returned an empty response")
    return content[:MAX_RESPONSE_CHARS]


def build_prompt(payload: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Phase 4 orchestration: build the full prompt for the LLM.

    1. Detect intent
    2. Collect conversation history
    3. Consult RAG documents (all intents)
    4. Consult Shopify mock (ORDER / PRODUCT intents)
    5. Compose final messages list

    Returns (messages, trace) where *trace* carries all intermediate data
    for a single consolidated debug log in :func:`generate_reply`.
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
    shopify_context = ""
    rag_context = ""
    tool_context = ""
    rag_chunks_count = 0
    raw_rag_chunks: list[Any] = []

    try:
        raw_rag_chunks = search_documents(content)
        rag_chunks_count = len(raw_rag_chunks)
        rag_context = format_rag_context(raw_rag_chunks)
    except Exception as exc:
        logger.warning("RAG search failed: %s", exc)

    if detected_intent == Intent.ORDER:
        order_number = intent_mod.extract_order_number(content)
        if order_number:
            order = _shopify.get_order_status(order_number)
            if order:
                shopify_context = _shopify.format_order_response(order)
            else:
                shopify_context = (
                    f"Nenhum pedido encontrado com o número #{order_number}. "
                    "Verifique se o número está correto."
                )
        else:
            shopify_context = (
                "Para consultar um pedido, por favor informe o número do pedido."
            )

    elif detected_intent == Intent.PRODUCT:
        products = _shopify.search_products(content, limit=3)
        shopify_context = _shopify.format_products_response(products, content)

    context_parts = [part for part in (shopify_context, rag_context) if part]
    tool_context = "\n\n---\n\n".join(context_parts)

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
    if not tool_context:
        sections.append(
            "Nenhuma informação relevante foi encontrada nos documentos para esta mensagem. "
            "Avise que não encontrou no material disponível e peça mais detalhes objetivos."
        )
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
        rag_chunks_count,
        prompt_chars,
        time.time() - t0,
    )

    messages = [
        {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    trace: dict[str, Any] = {
        "user_input": content,
        "intent": str(detected_intent),
        "history_context": history_context,
        "rag_chunks": [
            getattr(chunk, "page_content", None) or (chunk if isinstance(chunk, str) else str(chunk))
            for chunk in raw_rag_chunks
        ],
        "tool_context": tool_context,
        "system_prompt": OLLAMA_SYSTEM_PROMPT,
        "user_message": user_message,
    }

    return messages, trace


def _log_full_trace(trace: dict[str, Any], model_used: str, reply: str, elapsed: float) -> None:
    """Emit a single log entry with the complete request/response lifecycle."""
    sep = "─" * 72
    lines = [
        "",
        sep,
        "TRACE — FULL PIPELINE",
        sep,
        f"[INTENT]        {trace['intent']}",
        "",
        "[ENTRADA — mensagem do usuário]",
        trace["user_input"],
    ]

    if trace["history_context"]:
        lines += ["", "[HISTÓRICO DA CONVERSA]", trace["history_context"]]

    if trace["rag_chunks"]:
        lines += ["", f"[EMBEDDINGS — {len(trace['rag_chunks'])} chunk(s) recuperado(s)]"]
        for i, chunk in enumerate(trace["rag_chunks"], 1):
            lines.append(f"  [{i}] {chunk}")
    else:
        lines += ["", "[EMBEDDINGS]    (nenhum chunk recuperado)"]

    if trace["tool_context"]:
        lines += ["", "[CONTEXTO DE FERRAMENTA (Shopify / RAG formatado)]", trace["tool_context"]]

    lines += [
        "",
        "[PROMPT — system]",
        trace["system_prompt"],
        "",
        "[PROMPT — user message enviada ao modelo]",
        trace["user_message"],
        "",
        f"[RESPOSTA — model={model_used}  elapsed={elapsed:.2f}s]",
        reply,
        sep,
    ]

    logger.info("\n".join(lines))


def generate_reply(payload: dict[str, Any]) -> str:
    """Generate a reply with model fallback and fail-soft final response."""
    messages, trace = build_prompt(payload)
    t0 = time.time()
    logger.info(
        "Calling Ollama | base_url=%s primary=%s fallback=%s",
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_FALLBACK_MODEL,
    )

    try:
        reply = _call_ollama(messages, OLLAMA_MODEL, OLLAMA_MAIN_TIMEOUT)
        elapsed = time.time() - t0
        logger.info(
            "Reply generated | model=%s total_time=%.2fs", OLLAMA_MODEL, elapsed
        )
        _log_full_trace(trace, OLLAMA_MODEL, reply, elapsed)
        return reply
    except Exception as primary_exc:
        logger.warning(
            "Primary model %s failed (%s); trying fallback %s",
            OLLAMA_MODEL,
            primary_exc,
            OLLAMA_FALLBACK_MODEL,
        )

    try:
        reply = _call_ollama(messages, OLLAMA_FALLBACK_MODEL, OLLAMA_FALLBACK_TIMEOUT)
        elapsed = time.time() - t0
        logger.info(
            "Reply generated | model=%s (fallback) total_time=%.2fs",
            OLLAMA_FALLBACK_MODEL,
            elapsed,
        )
        _log_full_trace(trace, OLLAMA_FALLBACK_MODEL, reply, elapsed)
        return reply
    except requests.exceptions.RequestException as fallback_exc:
        logger.warning(
            "Fallback model %s failed (%s: %s); sending fail-soft message",
            OLLAMA_FALLBACK_MODEL,
            type(fallback_exc).__name__,
            fallback_exc,
        )
        return OLLAMA_UNAVAILABLE_MESSAGE[:MAX_RESPONSE_CHARS]
    except Exception:
        logger.exception(
            "Fallback model %s failed unexpectedly; sending fail-soft message",
            OLLAMA_FALLBACK_MODEL,
        )
        return OLLAMA_UNAVAILABLE_MESSAGE[:MAX_RESPONSE_CHARS]


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


@app.post("/chat")
def chat() -> tuple[Response, int] | Response:
    """Direct chat endpoint for testing the RAG + LLM pipeline without Chatwoot.

    Minimal required body:
        { "content": "sua pergunta aqui" }

    Optional fields (same shape as the Chatwoot webhook payload):
        contact.name, conversation.channel, conversation.id, account.id
    """
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify({"error": "missing or empty 'content' field"}), 400

    # Build a payload compatible with build_prompt / generate_reply
    payload: dict[str, Any] = {
        "event": "message_created",
        "message_type": "incoming",
        "private": False,
        "content": content,
        "contact": body.get("contact") or {},
        "conversation": body.get("conversation") or {},
        "account": body.get("account") or {},
        "sender": body.get("sender") or {"type": "contact"},
    }

    t0 = time.time()
    try:
        reply = generate_reply(payload)
    except Exception:
        logger.exception("/chat endpoint: error generating reply")
        return jsonify({"error": "internal error generating reply"}), 500

    return jsonify(
        {
            "reply": reply,
            "elapsed_seconds": round(time.time() - t0, 2),
        }
    )


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


# Auto-ingest RAG documents on startup (runs in background, non-blocking)
threading.Thread(target=ingest_documents, daemon=True, name="rag-ingest").start()

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=8000)