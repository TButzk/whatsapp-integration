import hashlib
import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from html import escape
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
from intent import Intent, IntentResult
from rag import (
    RAGSearchResult,
    format_rag_context,
    format_rag_evidence_context,
    ingest_documents,
    search_documents,
    search_documents_detailed,
)

# ---------------------------------------------------------------------------
# Shopify module — selected at startup based on SHOPIFY_MODE env variable.
# Supported values:
#   mock  (default) — local mock data, no network calls
#   mcp             — Shopify Storefront MCP Server (requires SHOPIFY_MCP_ENDPOINT)
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

# ---------------------------------------------------------------------------
# Inference file logger — one file per run, one block per request.
# Configured via INFERENCE_LOG_PATH (default: auto_reply_bridge/logs/inferences.log)
# ---------------------------------------------------------------------------
_INFERENCE_LOG_PATH = Path(
    os.getenv("INFERENCE_LOG_PATH", str(Path(__file__).resolve().parent / "logs" / "inferences.log"))
)
_INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_inference_logger = logging.getLogger("auto-reply-bridge.inference")
_inference_logger.setLevel(logging.DEBUG)
_inference_logger.propagate = False  # Don't duplicate to root handler
_inference_fh = logging.FileHandler(_INFERENCE_LOG_PATH, encoding="utf-8")
_inference_fh.setFormatter(logging.Formatter("%(message)s"))
_inference_logger.addHandler(_inference_fh)
logger.info("Inference log: %s", _INFERENCE_LOG_PATH)

app = Flask(__name__)
job_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
processed_deliveries: dict[str, float] = {}
processed_lock = threading.Lock()
prompt_config_lock = threading.Lock()


def _default_prompt_config() -> dict[str, Any]:
    default_prompt = os.getenv(
        "OLLAMA_SYSTEM_PROMPT",
        "Voce e um atendente virtual objetivo e educado. Responda em portugues do Brasil.",
    )
    return {
        "default_prompt": default_prompt,
        "conversation_prompts": {},
    }


def _load_prompt_config() -> dict[str, Any]:
    default_config = _default_prompt_config()
    try:
        if not PROMPT_CONFIG_PATH.exists():
            return default_config

        raw = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_config

        default_prompt = raw.get("default_prompt")
        if not isinstance(default_prompt, str) or not default_prompt.strip():
            default_prompt = default_config["default_prompt"]

        raw_overrides = raw.get("conversation_prompts")
        conversation_prompts: dict[str, str] = {}
        if isinstance(raw_overrides, dict):
            for key, value in raw_overrides.items():
                if not isinstance(value, str) or not value.strip():
                    continue
                conv_key = str(key).strip()
                if conv_key:
                    conversation_prompts[conv_key] = value.strip()

        return {
            "default_prompt": default_prompt.strip(),
            "conversation_prompts": conversation_prompts,
        }
    except Exception as exc:
        logger.warning("Failed to load prompt config from %s: %s", PROMPT_CONFIG_PATH, exc)
        return default_config


def _save_prompt_config_unlocked() -> None:
    PROMPT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_CONFIG_PATH.write_text(
        json.dumps(_prompt_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_conversation_key(conversation_id: Any) -> str | None:
    if conversation_id is None:
        return None
    text = str(conversation_id).strip()
    if not text or not text.isdigit():
        return None
    return text


def _resolve_system_prompt(conversation_id: Any) -> tuple[str, str]:
    conv_key = _normalize_conversation_key(conversation_id)
    with prompt_config_lock:
        if conv_key:
            override = _prompt_config["conversation_prompts"].get(conv_key, "").strip()
            if override:
                return override, f"conversation:{conv_key}"
        return _prompt_config["default_prompt"], "default"


def _set_default_prompt(prompt: str) -> None:
    clean = (prompt or "").strip()
    if not clean:
        raise ValueError("Prompt padrão não pode ser vazio")
    with prompt_config_lock:
        _prompt_config["default_prompt"] = clean
        _save_prompt_config_unlocked()


def _set_conversation_prompt(conversation_id: Any, prompt: str) -> str:
    conv_key = _normalize_conversation_key(conversation_id)
    if conv_key is None:
        raise ValueError("conversation_id inválido")
    clean = (prompt or "").strip()
    if not clean:
        raise ValueError("Prompt da conversa não pode ser vazio")
    with prompt_config_lock:
        _prompt_config["conversation_prompts"][conv_key] = clean
        _save_prompt_config_unlocked()
    return conv_key


def _delete_conversation_prompt(conversation_id: Any) -> str:
    conv_key = _normalize_conversation_key(conversation_id)
    if conv_key is None:
        raise ValueError("conversation_id inválido")
    with prompt_config_lock:
        _prompt_config["conversation_prompts"].pop(conv_key, None)
        _save_prompt_config_unlocked()
    return conv_key


_prompt_config: dict[str, Any] = _default_prompt_config()


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
PROMPT_CONFIG_PATH = Path(
    os.getenv(
        "PROMPT_CONFIG_PATH",
        str(Path(__file__).resolve().parent / "data" / "prompt_config.json"),
    )
)
_prompt_config = _load_prompt_config()
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
AUTO_REOPEN_CONVERSATION = os.getenv("AUTO_REOPEN_CONVERSATION", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Melhoria 5 — controle de verbosidade do trace de inferência.
# Quando False (padrão), emite apenas resumo (intent, scores, tempos).
# Quando True, emite trace completo, porém sempre sanitizado.
# ---------------------------------------------------------------------------
DEBUG_INFERENCE_TRACE = os.getenv("DEBUG_INFERENCE_TRACE", "false").lower() == "true"
INFERENCE_AUDIT_TRACE = os.getenv("INFERENCE_AUDIT_TRACE", "false").lower() == "true"
INFERENCE_LOG_FULL_TEXT = os.getenv("INFERENCE_LOG_FULL_TEXT", "false").lower() == "true"


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
OLLAMA_NUM_CTX = _read_optional_int_env("OLLAMA_NUM_CTX")
if OLLAMA_NUM_CTX is not None and OLLAMA_NUM_CTX < 256:
    logger.warning("OLLAMA_NUM_CTX=%s is too low; ignoring", OLLAMA_NUM_CTX)
    OLLAMA_NUM_CTX = None

logger.info(
    "Ollama settings | primary=%s fallback=%s keep_alive=%s main_timeout=%ss fallback_timeout=%ss timeout_cap=%ss num_gpu=%s num_ctx=%s",
    OLLAMA_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAIN_TIMEOUT,
    OLLAMA_FALLBACK_TIMEOUT,
    OLLAMA_TIMEOUT_MAX_SECONDS,
    OLLAMA_NUM_GPU if OLLAMA_NUM_GPU is not None else "default",
    OLLAMA_NUM_CTX if OLLAMA_NUM_CTX is not None else "default",
)

# ---------------------------------------------------------------------------
# Query Planner — LLM-powered query analysis before retrieval
# ---------------------------------------------------------------------------
QUERY_PLANNER_ENABLED = os.getenv("QUERY_PLANNER_ENABLED", "true").lower() == "true"
QUERY_PLANNER_MODEL = os.getenv("QUERY_PLANNER_MODEL", "").strip() or OLLAMA_MODEL
QUERY_PLANNER_TIMEOUT = _read_bounded_int_env(
    "QUERY_PLANNER_TIMEOUT", default=30, min_value=5, max_value=OLLAMA_TIMEOUT_MAX_SECONDS
)

_PLANNER_SYSTEM_PROMPT = (
    "Voce e um assistente de planejamento de busca. "
    "Dada a mensagem do cliente e o contexto da conversa, extraia as melhores queries de busca.\n\n"
    "Responda EXATAMENTE neste formato (uma chave por linha, sem explicacoes extras):\n"
    "intent: <order|product|institutional|general>\n"
    "rag_query: <melhor frase para busca semantica nos documentos da empresa, ou NONE>\n"
    "product_query: <termo limpo para buscar produto no catalogo, ou NONE>\n"
    "policy_query: <pergunta sobre politicas/trocas/devolucoes da loja, ou NONE>\n\n"
    "Regras:\n"
    "- rag_query: use palavras-chave relevantes em portugues, sem artigos desnecessarios\n"
    "- product_query: apenas o nome/tipo do produto, sem frases completas. Ex: 'snowboard', 'tenis branco'\n"
    "- policy_query: a pergunta sobre politica da loja, se aplicavel\n"
    "- Se o campo nao se aplica, escreva NONE\n"
    "- intent: corrija a intencao se o regex errou"
)

logger.info(
    "Query planner | enabled=%s model=%s timeout=%ss",
    QUERY_PLANNER_ENABLED,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_TIMEOUT,
)


def _parse_planner_response(raw: str) -> dict[str, str]:
    """Parse key: value lines from planner LLM output."""
    result: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key in ("intent", "rag_query", "product_query", "policy_query"):
            result[key] = "" if value.upper() == "NONE" else value
    return result


def _call_query_planner(
    content: str,
    regex_intent: str,
    history_context: str,
    api_calls: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Call the LLM to plan optimised search queries.

    Returns a dict with keys: intent, rag_query, product_query, policy_query.
    On any failure returns an empty dict (caller falls back to current behavior).
    """
    if not QUERY_PLANNER_ENABLED:
        return {}

    user_parts = []
    if history_context:
        user_parts.append(f"Historico recente:\n{history_context}")
    user_parts.append(f"Intencao detectada por regex: {regex_intent}")
    user_parts.append(f"Mensagem do cliente: {content}")

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    try:
        payload: dict[str, Any] = {
            "model": QUERY_PLANNER_MODEL,
            "stream": False,
            "messages": messages,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        options: dict[str, Any] = {}
        if OLLAMA_NUM_GPU is not None:
            options["num_gpu"] = OLLAMA_NUM_GPU
        if OLLAMA_NUM_CTX is not None:
            options["num_ctx"] = OLLAMA_NUM_CTX
        if options:
            payload["options"] = options

        planner_url = f"{OLLAMA_BASE_URL}/api/chat"
        request_chars = sum(len((m.get("content") or "")) for m in messages)
        t_api = time.time()
        response = requests.post(
            planner_url,
            json=payload,
            timeout=QUERY_PLANNER_TIMEOUT,
        )
        elapsed_ms = (time.time() - t_api) * 1000
        response.raise_for_status()
        data = response.json()
        if api_calls is not None:
            _append_api_call(
                api_calls,
                service="ollama",
                operation="query_planner",
                method="POST",
                url=planner_url,
                status=response.status_code,
                elapsed_ms=elapsed_ms,
                request_bytes=request_chars,
                response_bytes=len(response.text or ""),
                extra={
                    "model": QUERY_PLANNER_MODEL,
                    "timeout_s": QUERY_PLANNER_TIMEOUT,
                    "eval_count": data.get("eval_count"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                },
            )
        raw_reply = ((data.get("message") or {}).get("content") or "").strip()
        if not raw_reply:
            logger.warning("Query planner returned empty response")
            return {}

        parsed = _parse_planner_response(raw_reply)
        logger.info(
            "Query planner result | raw=%r parsed=%s",
            raw_reply[:200],
            parsed,
        )
        return parsed
    except Exception as exc:
        if api_calls is not None:
            _append_api_call(
                api_calls,
                service="ollama",
                operation="query_planner",
                method="POST",
                url=f"{OLLAMA_BASE_URL}/api/chat",
                status="error",
                elapsed_ms=0,
                error=str(exc),
                extra={"model": QUERY_PLANNER_MODEL, "timeout_s": QUERY_PLANNER_TIMEOUT},
            )
        logger.warning("Query planner failed (falling back to raw query): %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Melhoria 5 — sanitização de dados sensíveis nos logs.
# Remove CPF, telefone, e-mail e números de pedido antes de gravar.
# ---------------------------------------------------------------------------

_SANITIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # CPF: 123.456.789-00 ou 12345678900
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "***CPF***"),
    # Telefone: +55 11 91234-5678, (11) 91234-5678, 11912345678 etc.
    (re.compile(r"(?:\+?\d{1,3}[\s-]?)?\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}\b"), "***FONE***"),
    # E-mail
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "***EMAIL***"),
    # Número de pedido com # (ex.: #12345)
    (re.compile(r"#\d{4,}"), "#***"),
]

# Limite de caracteres por chunk nos logs (Melhoria 5)
_LOG_CHUNK_MAX_CHARS = int(os.getenv("INFERENCE_LOG_CHUNK_MAX_CHARS", "2000"))


def _sanitize_for_log(text: str) -> str:
    """Remove dados sensíveis (CPF, telefone, e-mail, pedido) de um texto para log."""
    if not text:
        return text
    for pattern, replacement in _SANITIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _truncate_for_log(text: str, max_chars: int = _LOG_CHUNK_MAX_CHARS) -> str:
    """Trunca texto longo para logs, adicionando indicador de corte."""
    if INFERENCE_LOG_FULL_TEXT:
        return text
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text) - max_chars} chars truncados]"


def _append_api_call(
    api_calls: list[dict[str, Any]],
    *,
    service: str,
    operation: str,
    method: str,
    url: str,
    status: int | str,
    elapsed_ms: float,
    request_bytes: int | None = None,
    response_bytes: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "service": service,
        "operation": operation,
        "method": method,
        "url": url,
        "status": status,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if request_bytes is not None:
        entry["request_bytes"] = request_bytes
    if response_bytes is not None:
        entry["response_bytes"] = response_bytes
    if error:
        entry["error"] = error
    if extra:
        entry["extra"] = extra
    api_calls.append(entry)


# ---------------------------------------------------------------------------
# Melhoria 3 — heurística para decidir quando usar o query planner LLM.
# Evita custo desnecessário em casos simples onde regex + busca direta
# são suficientes.
# ---------------------------------------------------------------------------


def should_use_query_planner(
    content: str,
    intent_result: IntentResult,
    history_context: str,
) -> tuple[bool, str]:
    """Decide se o query planner LLM deve ser chamado.

    Retorna (usar_planner: bool, motivo: str) para rastreabilidade.

    Regras (em ordem de prioridade):
    1. Se QUERY_PLANNER_ENABLED=false → nunca usar
    2. ORDER + número de pedido explícito → skip (busca direta é melhor)
    3. PRODUCT + mensagem curta → skip (regex já extraiu bem)
    4. INSTITUTIONAL → skip (fallback determinístico de políticas é melhor)
    5. GENERAL → usar (precisa de ajuda para rotear)
    6. Mensagem longa (>100 chars) → usar (pode ser ambígua)
    7. Default → skip
    """
    if not QUERY_PLANNER_ENABLED:
        return False, "planner_disabled"

    intent = intent_result.intent
    msg_len = len(content.strip())

    # ORDER com número de pedido: busca direta no Shopify, planner desnecessário
    if intent == Intent.ORDER and intent_result.has_order_id:
        return False, "order_with_id"

    # PRODUCT com mensagem curta: regex extrai bem o termo de busca
    if intent == Intent.PRODUCT and msg_len < 50:
        return False, "simple_product"

    # INSTITUTIONAL: o fallback determinístico de políticas (Melhoria 4)
    # já cobre esse caso sem precisar de LLM
    if intent == Intent.INSTITUTIONAL:
        return False, "institutional_direct"

    # GENERAL: sem sinal claro de intenção — planner ajuda a rotear
    if intent == Intent.GENERAL:
        return True, "general_needs_planner"

    # Mensagem longa pode ser ambígua mesmo com intenção detectada
    if msg_len > 100:
        return True, "long_message"

    # Default: skip para economizar latência
    return False, "default_skip"


# ---------------------------------------------------------------------------
# Melhoria 4 — fallback determinístico para busca de políticas.
# Dispara busca RAG focada em políticas quando há forte indício
# (palavras-chave), independente do planner.
# ---------------------------------------------------------------------------


def _build_policy_search_query(content: str, intent_result: IntentResult) -> str:
    """Retorna query de busca para políticas, ou string vazia se não aplicável.

    Dispara quando:
    - intent_result.has_policy_terms é True, OU
    - intenção é INSTITUTIONAL
    """
    if intent_result.has_policy_terms or intent_result.intent == Intent.INSTITUTIONAL:
        # Usar a mensagem original como query — mais natural para busca semântica
        return content.strip()
    return ""


_CART_KEYWORDS = (
    "carrinho",
    "cart",
    "checkout",
    "finalizar compra",
    "adicionar no carrinho",
    "remover do carrinho",
)


def _is_cart_message(content: str) -> bool:
    text = content.lower()
    return any(keyword in text for keyword in _CART_KEYWORDS)


def _extract_cart_id(content: str, payload: dict[str, Any]) -> str:
    text = content.strip()
    patterns = [
        r"\bgid://shopify/Cart/[A-Za-z0-9_\-=%]+",
        r"\bcart[_-]?id\s*[:=]\s*([A-Za-z0-9_\-=%]+)",
        r"\bcarrinho\s*[:#]?\s*([A-Za-z0-9_\-=%]{8,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if match.group(0).startswith("gid://shopify/Cart/"):
            return match.group(0)
        return (match.group(1) if match.lastindex else match.group(0)).strip()

    contact_obj = payload.get("contact") or {}
    attrs = contact_obj.get("additional_attributes") or {}
    for key in ("shopify_cart_id", "cart_id", "shopifyCartId"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _extract_variant_id(content: str) -> str:
    text = content.strip()
    gid_match = re.search(r"\bgid://shopify/ProductVariant/\d+", text, flags=re.IGNORECASE)
    if gid_match:
        return gid_match.group(0)

    num_match = re.search(r"\bvariant(?:e)?\s*(?:id)?\s*[:#]?\s*(\d{4,})\b", text, flags=re.IGNORECASE)
    if num_match:
        return f"gid://shopify/ProductVariant/{num_match.group(1)}"

    return ""


def _extract_quantity(content: str, default: int = 1) -> int:
    text = content.lower()
    qty_match = re.search(r"\b(?:qtd|quantidade|qty)\s*[:=]?\s*(\d{1,3})\b", text)
    if qty_match:
        return max(1, int(qty_match.group(1)))

    generic_match = re.search(r"\b(\d{1,3})\s*(?:un|unid|unidade|itens?)\b", text)
    if generic_match:
        return max(1, int(generic_match.group(1)))

    return default


def _detect_cart_action(content: str) -> str:
    text = content.lower()
    if any(term in text for term in ("checkout", "finalizar", "pagar", "fechar compra")):
        return "checkout"
    if any(term in text for term in ("remover", "tirar", "excluir", "deletar")):
        return "remove"
    if any(term in text for term in ("atualizar", "alterar", "mudar quantidade", "quantidade")):
        return "update"
    if any(term in text for term in ("adicionar", "incluir", "colocar")):
        return "add"
    return "get"


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


def _normalize_webhook_message_type(msg_type: Any) -> str | None:
    """Normalize Chatwoot webhook message_type values.

    Chatwoot may send message_type as strings ("incoming"/"outgoing")
    or integers (0 incoming, 1 outgoing).
    """
    if isinstance(msg_type, str):
        normalized = msg_type.strip().lower()
        if normalized in {"incoming", "outgoing"}:
            return normalized

    if isinstance(msg_type, bool):
        # bool is subclass of int; avoid treating True/False as message types
        return None

    if isinstance(msg_type, int):
        if msg_type == 0:
            return "incoming"
        if msg_type == 1:
            return "outgoing"

    return None


def should_reply(payload: dict[str, Any]) -> tuple[bool, str]:
    if payload.get("event") != "message_created":
        return False, "unsupported_event"

    message_type = _normalize_webhook_message_type(payload.get("message_type"))
    if message_type != "incoming":
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

def _call_ollama(
    messages: list[dict[str, str]],
    model: str,
    timeout: int,
    api_calls: list[dict[str, Any]] | None = None,
    operation: str = "generate_reply",
) -> str:
    """Call the Ollama chat API and return the response text."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    options: dict[str, Any] = {}
    if OLLAMA_NUM_GPU is not None:
        # num_gpu controls how many layers should be offloaded to GPU in Ollama.
        options["num_gpu"] = OLLAMA_NUM_GPU
    if OLLAMA_NUM_CTX is not None:
        options["num_ctx"] = OLLAMA_NUM_CTX
    if options:
        payload["options"] = options

    request_chars = sum(len((m.get("content") or "")) for m in messages)
    t_api = time.time()
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:
        if api_calls is not None:
            _append_api_call(
                api_calls,
                service="ollama",
                operation=operation,
                method="POST",
                url=url,
                status="error",
                elapsed_ms=(time.time() - t_api) * 1000,
                request_bytes=request_chars,
                error=str(exc),
                extra={"model": model, "timeout_s": timeout},
            )
        raise

    elapsed_ms = (time.time() - t_api) * 1000
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
    if api_calls is not None:
        _append_api_call(
            api_calls,
            service="ollama",
            operation=operation,
            method="POST",
            url=url,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
            request_bytes=request_chars,
            response_bytes=len(response.text or ""),
            extra={
                "model": model,
                "timeout_s": timeout,
                "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
            },
        )
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("Ollama returned an empty response")
    return content[:MAX_RESPONSE_CHARS]


def build_prompt(payload: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Orchestration: build the full prompt for the LLM.

    1. Detect intent (regex, fast) — agora com sinais auxiliares (Melhoria 9)
    2. Collect conversation history
    3. Query Planner — condicional (Melhoria 3)
    4. Retrieve: RAG docs + políticas (fallback determinístico) + Shopify
    5. Compose final messages list — com evidências estruturadas (Melhorias 1, 6)

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
    system_prompt, system_prompt_source = _resolve_system_prompt(conversation_id)

    t0 = time.time()
    timings: dict[str, float] = {}

    # ---- 1. Detect intent (regex — fast) — Melhoria 9 ------------------
    t_intent = time.time()
    intent_result = intent_mod.detect_intent_enriched(content)
    regex_intent = intent_result.intent
    timings["intent_ms"] = round((time.time() - t_intent) * 1000, 1)

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

    # ---- 3. Query Planner (condicional — Melhoria 3) --------------------
    t_planner = time.time()
    planner_used, planner_skip_reason = should_use_query_planner(
        content, intent_result, history_context,
    )

    api_calls: list[dict[str, Any]] = []
    planned: dict[str, str] = {}
    planner_raw = ""
    if planner_used:
        planned = _call_query_planner(
            content,
            str(regex_intent),
            history_context,
            api_calls=api_calls,
        )
        if planned:
            planner_raw = str(planned)
    timings["planner_ms"] = round((time.time() - t_planner) * 1000, 1)

    # Resolve final intent (planner can override regex)
    planned_intent_str = planned.get("intent", "").strip().lower()
    intent_map = {
        "order": Intent.ORDER,
        "product": Intent.PRODUCT,
        "institutional": Intent.INSTITUTIONAL,
        "general": Intent.GENERAL,
    }
    detected_intent = intent_map.get(planned_intent_str, regex_intent)

    # Resolve search queries
    rag_query = planned.get("rag_query", "").strip() or content
    product_search_query = planned.get("product_query", "").strip()
    policy_query = planned.get("policy_query", "").strip()

    # Fallback: if planner didn't produce product_query for PRODUCT intent
    if detected_intent == Intent.PRODUCT and not product_search_query:
        product_search_query = intent_mod.extract_product_query(content) or content

    # ---- 4. Retrieve: RAG + políticas + Shopify -------------------------
    shopify_context = ""
    rag_evidence_context = ""
    policy_evidence_context = ""
    policy_context = ""
    tool_context = ""
    rag_results_raw_count = 0
    rag_results_filtered_count = 0
    raw_rag_results: list[RAGSearchResult] = []
    policy_results: list[RAGSearchResult] = []

    # 4a. RAG document search — agora usa search_documents_detailed (Melhoria 1)
    t_rag = time.time()
    try:
        raw_rag_results = search_documents_detailed(rag_query)
        rag_results_raw_count = len(raw_rag_results)
        rag_evidence_context = format_rag_evidence_context(raw_rag_results)
    except Exception as exc:
        logger.warning("RAG search failed: %s", exc)
    timings["rag_ms"] = round((time.time() - t_rag) * 1000, 1)

    # 4b. Fallback determinístico de políticas (Melhoria 4)
    # Dispara busca de políticas quando há forte indício, mesmo sem planner.
    policy_fallback_query = _build_policy_search_query(content, intent_result)
    if policy_fallback_query and not policy_query:
        # Usar fallback determinístico — planner não gerou policy_query
        policy_query = policy_fallback_query

    if policy_query:
        try:
            policy_results = search_documents_detailed(policy_query)
            if policy_results:
                policy_evidence_context = format_rag_evidence_context(
                    policy_results,
                )
        except Exception as exc:
            logger.warning("Policy RAG search failed: %s", exc)

    # 4c. Shopify policy search (se planner gerou policy_query e módulo suporta)
    if policy_query and hasattr(_shopify, "search_policies"):
        try:
            shopify_policy = _shopify.search_policies(policy_query)
            if shopify_policy:
                policy_context = shopify_policy
        except Exception as exc:
            logger.warning("Shopify policy search failed: %s", exc)

    # 4d. Shopify context (cart, order or product)
    t_shopify = time.time()
    if _is_cart_message(content):
        cart_action = _detect_cart_action(content)
        cart_id = _extract_cart_id(content, payload)
        variant_id = _extract_variant_id(content)
        quantity = _extract_quantity(content, default=1)

        if not cart_id and cart_action in ("get", "checkout", "update", "remove"):
            shopify_context = (
                "Para operar o carrinho preciso do cart_id. "
                "Envie o identificador do carrinho (ex.: gid://shopify/Cart/...)."
            )
        elif cart_action == "add":
            if not variant_id:
                shopify_context = (
                    "Para adicionar no carrinho preciso do variant_id do produto "
                    "(ex.: gid://shopify/ProductVariant/1234567890)."
                )
            elif hasattr(_shopify, "add_to_cart") and hasattr(_shopify, "format_cart_response"):
                cart_payload = _shopify.add_to_cart(cart_id, variant_id, quantity=quantity)
                shopify_context = _shopify.format_cart_response(cart_payload, "adicao")
            else:
                shopify_context = "Operacoes de carrinho nao estao disponiveis neste modo."
        elif cart_action == "update":
            if hasattr(_shopify, "update_cart_item") and hasattr(_shopify, "format_cart_response"):
                cart_payload = _shopify.update_cart_item(
                    cart_id,
                    quantity,
                    variant_id=variant_id,
                )
                shopify_context = _shopify.format_cart_response(cart_payload, "atualizacao")
            else:
                shopify_context = "Operacoes de carrinho nao estao disponiveis neste modo."
        elif cart_action == "remove":
            if hasattr(_shopify, "remove_from_cart") and hasattr(_shopify, "format_cart_response"):
                cart_payload = _shopify.remove_from_cart(cart_id, variant_id=variant_id)
                shopify_context = _shopify.format_cart_response(cart_payload, "remocao")
            else:
                shopify_context = "Operacoes de carrinho nao estao disponiveis neste modo."
        elif cart_action == "checkout":
            if hasattr(_shopify, "get_cart_checkout_url"):
                checkout_url = _shopify.get_cart_checkout_url(cart_id)
                if checkout_url:
                    shopify_context = f"Checkout do carrinho: {checkout_url}"
                else:
                    shopify_context = (
                        "Nao consegui obter o checkout agora. "
                        "Confirme o cart_id e tente novamente."
                    )
            else:
                shopify_context = "Operacoes de checkout do carrinho nao estao disponiveis neste modo."
        else:
            if hasattr(_shopify, "get_cart") and hasattr(_shopify, "format_cart_response"):
                cart_payload = _shopify.get_cart(cart_id)
                shopify_context = _shopify.format_cart_response(cart_payload, "consulta")
            else:
                shopify_context = "Operacoes de carrinho nao estao disponiveis neste modo."

    elif detected_intent == Intent.ORDER:
        def _extract_customer_identifiers() -> tuple[str | None, str | None, str | None]:
            contact_obj = payload.get("contact") or {}
            attrs = contact_obj.get("additional_attributes") or {}

            email = (
                (contact_obj.get("email") or attrs.get("email") or "").strip().lower()
                or None
            )
            phone = (
                (contact_obj.get("phone_number") or attrs.get("phone_number") or "").strip()
                or None
            )
            cpf = (
                (attrs.get("cpf") or "").strip()
                or None
            )

            if not cpf:
                cpf_match = re.search(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b", content)
                if cpf_match:
                    cpf = cpf_match.group(1)

            return email, phone, cpf

        order_number = intent_result.order_number or intent_mod.extract_order_number(content)
        capability_checker = getattr(_shopify, "is_order_lookup_available", None)
        if callable(capability_checker):
            supports_order_lookup = bool(capability_checker())
        else:
            supports_order_lookup = bool(getattr(_shopify, "SUPPORTS_ORDER_LOOKUP", True))
        if order_number:
            if supports_order_lookup:
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
                    "Este canal usa somente Shopify Storefront MCP e nao tem acesso "
                    "a dados privados de pedidos. Para consultar status do pedido "
                    f"#{order_number}, encaminhe para atendimento humano/autenticado."
                )
        else:
            summary = None
            email, phone, cpf = _extract_customer_identifiers()
            if supports_order_lookup and hasattr(_shopify, "get_customer_purchase_summary"):
                try:
                    summary = _shopify.get_customer_purchase_summary(
                        content,
                        email=email,
                        phone=phone,
                        cpf=cpf,
                    )
                except Exception as exc:
                    logger.warning("Customer purchase summary failed: %s", exc)

            if summary:
                shopify_context = summary
            else:
                if supports_order_lookup:
                    shopify_context = (
                        "Para consultar um pedido, informe o numero do pedido. "
                        "Se preferir, informe tambem e-mail/telefone (ou CPF com validacao adicional) "
                        "para localizar pedidos recentes no cadastro."
                    )
                else:
                    shopify_context = (
                        "Este canal usa somente Shopify Storefront MCP e nao consulta "
                        "historico de pedidos de clientes. Encaminhe para atendimento humano "
                        "ou canal autenticado para status de pedido."
                    )

    elif detected_intent == Intent.PRODUCT:
        products = _shopify.search_products(product_search_query, limit=3)
        shopify_context = _shopify.format_products_response(products, product_search_query)

    elif hasattr(_shopify, "list_available_resources") and any(
        term in content.lower() for term in ("recurso", "ferramenta", "capacidades", "mcp")
    ):
        shopify_context = _shopify.list_available_resources()
    timings["shopify_ms"] = round((time.time() - t_shopify) * 1000, 1)

    # ---- 5. Build user message (Melhoria 6 — prompt robusto) ------------
    # Separa claramente cada tipo de contexto com rótulos para o modelo.
    sections: list[str] = [
        f"Canal: {channel}",
        f"Cliente: {contact_name}",
    ]

    # [HISTÓRICO]
    if history_context:
        sections.append(f"[HISTÓRICO DA CONVERSA]\n{history_context}")

    # [EVIDÊNCIAS DOCUMENTAIS] — Melhoria 1: inclui fonte, score, trecho
    has_doc_evidence = bool(rag_evidence_context)
    if rag_evidence_context:
        sections.append(
            f"[EVIDÊNCIAS DOCUMENTAIS]\n"
            f"As evidências abaixo foram recuperadas dos documentos da empresa. "
            f"Use-as como base principal para responder.\n\n{rag_evidence_context}"
        )

    # [POLÍTICAS] — Melhoria 4: fallback determinístico
    all_policy = "\n\n".join(
        p for p in (policy_evidence_context, policy_context) if p
    )
    if all_policy:
        sections.append(f"[POLÍTICAS DA EMPRESA]\n{all_policy}")

    # [CONTEXTO SHOPIFY]
    if shopify_context:
        sections.append(f"[CONTEXTO SHOPIFY]\n{shopify_context}")

    # [MENSAGEM DO CLIENTE]
    sections.append(f"[MENSAGEM DO CLIENTE]\n{content}")

    # Instruções de grounding para o modelo (Melhoria 6)
    if has_doc_evidence or all_policy or shopify_context:
        grounding_instructions = (
            "INSTRUÇÕES DE RESPOSTA:\n"
            "- Responda com base PRIMEIRO nas evidências documentais e políticas fornecidas acima.\n"
            "- NÃO invente políticas, preços, prazos nem detalhes que não estejam nas evidências.\n"
            "- Se a evidência for insuficiente para uma resposta completa, diga explicitamente "
            "que não encontrou informação suficiente no material disponível.\n"
            "- Faça pergunta curta apenas se realmente faltar informação crítica para ajudar.\n"
            "- Responda de forma objetiva e útil em português do Brasil."
        )
        if detected_intent == Intent.PRODUCT and shopify_context:
            grounding_instructions += (
                "\n- IMPORTANTE: os produtos listados no CONTEXTO SHOPIFY existem no catálogo. "
                "Apresente-os ao cliente. Não diga que não temos o produto se ele aparece acima."
            )
    else:
        # Nenhuma evidência — instruir modelo a não inventar (Melhoria 6)
        grounding_instructions = (
            "INSTRUÇÕES DE RESPOSTA:\n"
            "- NOTA: Nenhuma evidência documental foi encontrada para esta mensagem.\n"
            "- NÃO assuma informações sobre a empresa, políticas ou produtos.\n"
            "- Informe que não encontrou no material disponível e peça mais detalhes objetivos.\n"
            "- Responda de forma objetiva e útil em português do Brasil."
        )
    sections.append(grounding_instructions)

    user_message = "\n\n".join(sections)

    # ---- 6. Metrics log (Melhoria 7 — timings) --------------------------
    timings["total_ms"] = round((time.time() - t0) * 1000, 1)
    prompt_chars = len(system_prompt) + len(user_message)
    logger.info(
        "Prompt built | intent=%s planner=%s(%s) history_msgs=%d rag_evidence=%d prompt_chars=%d timings=%s",
        detected_intent,
        planner_used,
        planner_skip_reason,
        len(history),
        rag_results_raw_count,
        prompt_chars,
        timings,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # ---- Melhoria 7 — trace enriquecido para observabilidade ------------
    trace: dict[str, Any] = {
        "trace_id": uuid.uuid4().hex[:10],
        "webhook_meta": {
            "event": payload.get("event"),
            "message_type": payload.get("message_type"),
            "private": payload.get("private"),
            "account_id": account_id,
            "conversation_id": conversation_id,
            "contact_id": contact.get("id"),
            "sender_type": (payload.get("sender") or {}).get("type"),
        },
        "user_input": content,
        "regex_intent": str(regex_intent),
        "intent": str(detected_intent),
        "intent_signals": {
            "has_order_id": intent_result.has_order_id,
            "has_policy_terms": intent_result.has_policy_terms,
            "has_product_terms": intent_result.has_product_terms,
            "order_number": intent_result.order_number,
        },
        "planner_used": planner_used,
        "planner_skip_reason": planner_skip_reason,
        "planner_raw": planner_raw,
        "planned_rag_query": rag_query,
        "planned_product_query": product_search_query,
        "planned_policy_query": policy_query,
        "policy_fallback_used": bool(policy_fallback_query and not planned.get("policy_query")),
        "product_search_query": product_search_query,
        "history_context": history_context,
        "history_messages": history,
        "rag_evidence": [
            {
                "source": r.source,
                "source_path": r.source_path,
                "score": round(r.score, 3),
                "distance": round(r.distance, 4) if r.distance is not None else None,
            }
            for r in raw_rag_results
        ],
        "rag_files_used": sorted(
            {
                (r.source_path or r.source)
                for r in (raw_rag_results + policy_results)
                if (r.source_path or r.source)
            }
        ),
        "rag_results_raw_count": rag_results_raw_count,
        "rag_chunks": [r.text for r in raw_rag_results],
        "policy_rag_chunks": [r.text for r in policy_results],
        "shopify_context": shopify_context,
        "policy_context": all_policy,
        "tool_context": rag_evidence_context,
        "contexts_used": {
            "history": bool(history_context),
            "rag": bool(rag_evidence_context),
            "policy": bool(all_policy),
            "shopify": bool(shopify_context),
        },
        "api_calls": api_calls,
        "system_prompt": system_prompt,
        "system_prompt_source": system_prompt_source,
        "user_message": user_message,
        "timings": timings,
    }

    return messages, trace


def _log_full_trace(trace: dict[str, Any], model_used: str, reply: str, elapsed: float) -> None:
    """Emit one block per inference to both console logger and the inference file.

    Melhoria 5: quando DEBUG_INFERENCE_TRACE=false (padrão), emite apenas
    resumo compacto.  Quando true, emite trace completo porém sanitizado.
    Melhoria 7: inclui timings, scores, decisão do planner e contagens.
    """
    sep = "─" * 72
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Resumo compacto (sempre emitido) ----
    timings = trace.get("timings", {})
    rag_evidence = trace.get("rag_evidence", [])
    summary_lines = [
        "",
        sep,
        f"TRACE SUMMARY  [{ts}]  trace_id={trace.get('trace_id', '-')}  model={model_used}  elapsed={elapsed:.2f}s",
        sep,
        f"  regex_intent     = {trace.get('regex_intent', '?')}",
        f"  final_intent     = {trace.get('intent', '?')}",
        f"  planner_used     = {trace.get('planner_used', '?')}  reason={trace.get('planner_skip_reason', '?')}",
        f"  policy_fallback  = {trace.get('policy_fallback_used', False)}",
        f"  rag_evidence     = {trace.get('rag_results_raw_count', 0)} retrieved",
        f"  external_calls   = {len(trace.get('api_calls', []))}",
        f"  intent_signals   = {trace.get('intent_signals', {})}",
    ]
    # Score de cada evidência (Melhoria 7)
    for ev in rag_evidence:
        summary_lines.append(
            f"    - {ev.get('source', '?')}  score={ev.get('score', '?')}  dist={ev.get('distance', '?')}"
        )
    summary_lines.append(f"  timings          = {timings}")
    summary_lines.append(sep)

    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)
    _inference_logger.info(summary_text)

    # ---- Trace completo (DEBUG_INFERENCE_TRACE ou modo de auditoria) ----
    if not (DEBUG_INFERENCE_TRACE or INFERENCE_AUDIT_TRACE):
        return

    # Sanitizar dados sensíveis antes de gravar (Melhoria 5)
    s = _sanitize_for_log
    t = _truncate_for_log

    lines = [
        "",
        f"TRACE — FULL PIPELINE  [{ts}]  trace_id={trace.get('trace_id', '-')}",
        sep,
        "[WEBHOOK META]",
        s(str(trace.get("webhook_meta", {}))),
        f"[REGEX INTENT]  {trace.get('regex_intent', '?')}",
        f"[FINAL INTENT]  {trace['intent']}",
        f"[INTENT SIGNALS]  {trace.get('intent_signals', {})}",
        "",
        "[ENTRADA — mensagem do usuário]",
        s(trace["user_input"]),
    ]

    # Query Planner block
    if trace.get("planner_used"):
        lines += [
            "",
            f"[QUERY PLANNER — used=True]",
            s(str(trace.get("planner_raw", ""))),
            f"  rag_query      = {s(trace.get('planned_rag_query', ''))}",
            f"  product_query  = {s(trace.get('planned_product_query', ''))}",
            f"  policy_query   = {s(trace.get('planned_policy_query', ''))}",
        ]
    else:
        lines += [
            "",
            f"[QUERY PLANNER] skipped — reason={trace.get('planner_skip_reason', '?')}",
        ]

    if trace.get("policy_fallback_used"):
        lines += ["", "[POLICY FALLBACK] determinístico ativado (Melhoria 4)"]

    if trace.get("product_search_query"):
        lines += ["", "[SHOPIFY QUERY — busca de produto]", s(trace["product_search_query"])]

    if trace.get("history_context"):
        lines += ["", "[HISTÓRICO DA CONVERSA — formatado]", s(t(trace["history_context"]))]
    if trace.get("history_messages"):
        lines += [
            "",
            "[HISTÓRICO DA CONVERSA — estruturado]",
            s(t(str(trace["history_messages"]), 4000)),
        ]

    # Evidências RAG com scores (Melhoria 7)
    rag_chunks = trace.get("rag_chunks", [])
    if rag_chunks:
        lines += ["", f"[EVIDÊNCIAS RAG — {len(rag_chunks)} recuperada(s)]"]
        for i, (chunk, ev) in enumerate(
            zip(rag_chunks, rag_evidence + [{}] * max(0, len(rag_chunks) - len(rag_evidence))),
            1,
        ):
            score_info = f"  score={ev.get('score', '?')}  dist={ev.get('distance', '?')}" if ev else ""
            lines.append(f"  [{i}]{score_info}")
            lines.append(f"  {s(t(chunk))}")
    else:
        lines += ["", "[EVIDÊNCIAS RAG]  (nenhuma recuperada)"]

    shopify_ctx = trace.get("shopify_context", "")
    if shopify_ctx:
        lines += ["", "[SHOPIFY — dados brutos]", s(t(shopify_ctx))]
    else:
        lines += ["", "[SHOPIFY]  (vazio)"]

    policy_ctx = trace.get("policy_context", "")
    if policy_ctx:
        lines += ["", "[POLÍTICAS]", s(t(policy_ctx))]

    lines += ["", "[CONTEXTOS UTILIZADOS]", s(str(trace.get("contexts_used", {})))]

    files_used = trace.get("rag_files_used") or []
    if files_used:
        lines += ["", "[ARQUIVOS USADOS PARA CONTEXTO]"]
        for file_name in files_used:
            lines.append(f"  - {file_name}")

    lines += [
        "",
        "[PROMPT — system]",
        t(trace["system_prompt"]),
        "",
        "[PROMPT — user message]",
        s(t(trace["user_message"], 2000)),
    ]

    if INFERENCE_AUDIT_TRACE and trace.get("api_calls"):
        lines += ["", "[CONSUMO DE APIs EXTERNAS]"]
        for call in trace["api_calls"]:
            lines.append(
                "  - {service}/{operation} {method} status={status} elapsed={elapsed}ms req={req} resp={resp}".format(
                    service=call.get("service", "?"),
                    operation=call.get("operation", "?"),
                    method=call.get("method", "?"),
                    status=call.get("status", "?"),
                    elapsed=call.get("elapsed_ms", "?"),
                    req=call.get("request_bytes", "?"),
                    resp=call.get("response_bytes", "?"),
                )
            )
            if call.get("extra"):
                lines.append(f"    extra={s(t(str(call.get('extra')), 1000))}")
            if call.get("error"):
                lines.append(f"    error={s(t(str(call.get('error')), 1000))}")

    lines += [
        "",
        f"[RESPOSTA — model={model_used}  elapsed={elapsed:.2f}s]",
        s(reply),
        sep,
    ]

    full_text = "\n".join(lines)
    _inference_logger.info(full_text)


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
        reply = _call_ollama(
            messages,
            OLLAMA_MODEL,
            OLLAMA_MAIN_TIMEOUT,
            api_calls=trace.get("api_calls"),
            operation="generate_reply_primary",
        )
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
        reply = _call_ollama(
            messages,
            OLLAMA_FALLBACK_MODEL,
            OLLAMA_FALLBACK_TIMEOUT,
            api_calls=trace.get("api_calls"),
            operation="generate_reply_fallback",
        )
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


def reopen_chatwoot_conversation(payload: dict[str, Any]) -> None:
    """Best-effort reopen of the conversation to keep it visible in Open views."""
    account = payload.get("account") or {}
    conversation = payload.get("conversation") or {}
    account_id = account.get("id")
    conversation_id = conversation.get("id")

    if not account_id or not conversation_id:
        raise ValueError("Webhook payload missing account or conversation identifiers")

    if not CHATWOOT_API_TOKEN:
        raise ValueError("CHATWOOT_API_TOKEN is not configured")

    response = requests.patch(
        (
            f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}"
        ),
        headers={
            "Content-Type": "application/json",
            "api_access_token": CHATWOOT_API_TOKEN,
        },
        json={
            "status": "open",
        },
        timeout=30,
    )
    response.raise_for_status()


def worker() -> None:
    while True:
        payload = job_queue.get()
        try:
            reply_text = generate_reply(payload)

            if AUTO_REOPEN_CONVERSATION:
                try:
                    reopen_chatwoot_conversation(payload)
                except Exception as exc:
                    logger.warning(
                        "Failed to reopen conversation before reply: %s",
                        exc,
                    )

            send_chatwoot_message(payload, reply_text)
            logger.info("Auto-reply sent for conversation %s", (payload.get("conversation") or {}).get("id"))
        except Exception:
            logger.exception("Failed to process auto-reply job")
        finally:
            job_queue.task_done()


@app.get("/healthz")
def healthz() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.route("/prompt-config", methods=["GET", "POST"])
def prompt_config_page() -> Response:
    message = ""
    error = ""

    if request.method == "POST":
        action = (request.form.get("action") or "").strip()
        try:
            if action == "save_default":
                _set_default_prompt(request.form.get("default_prompt") or "")
                message = "Prompt padrão atualizado com sucesso."
            elif action == "save_conversation":
                conv_id = request.form.get("conversation_id")
                _set_conversation_prompt(conv_id, request.form.get("conversation_prompt") or "")
                message = f"Prompt da conversa {conv_id} atualizado."
            elif action == "delete_conversation":
                conv_id = request.form.get("conversation_id")
                _delete_conversation_prompt(conv_id)
                message = f"Override da conversa {conv_id} removido."
            else:
                error = "Ação inválida."
        except Exception as exc:
            error = str(exc)

    with prompt_config_lock:
        default_prompt = _prompt_config.get("default_prompt", "")
        conversation_prompts = dict(_prompt_config.get("conversation_prompts", {}))

    rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(conv_id)}</td>"
            f"<td><pre>{escape(prompt_text)}</pre></td>"
            "<td>"
            "<form method='post'>"
            "<input type='hidden' name='action' value='delete_conversation' />"
            f"<input type='hidden' name='conversation_id' value='{escape(conv_id)}' />"
            "<button type='submit'>Remover</button>"
            "</form>"
            "</td>"
            "</tr>"
        )
        for conv_id, prompt_text in sorted(
            conversation_prompts.items(),
            key=lambda item: int(item[0]),
        )
    )
    if not rows:
        rows = "<tr><td colspan='3'><em>Sem overrides por conversa.</em></td></tr>"

    html = f"""
<!doctype html>
<html lang='pt-BR'>
<head>
  <meta charset='utf-8' />
  <title>Prompt Config</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 24px; max-width: 1100px; }}
    h1 {{ margin-bottom: 8px; }}
    textarea {{ width: 100%; min-height: 150px; }}
    input[type='text'], input[type='number'] {{ width: 260px; padding: 6px; }}
    button {{ padding: 8px 12px; margin-top: 8px; }}
    .ok {{ color: #166534; }}
    .err {{ color: #991b1b; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin-top: 16px; }}
  </style>
</head>
<body>
  <h1>Configuração de Prompt do Agente</h1>
  <p>Edite o texto de regras que o agente lê antes de responder.</p>
  <p class='ok'>{escape(message)}</p>
  <p class='err'>{escape(error)}</p>

  <div class='card'>
    <h2>Prompt Padrão (todas as conversas)</h2>
    <form method='post'>
      <input type='hidden' name='action' value='save_default' />
      <textarea name='default_prompt'>{escape(default_prompt)}</textarea>
      <br />
      <button type='submit'>Salvar Prompt Padrão</button>
    </form>
  </div>

  <div class='card'>
    <h2>Override por Conversa</h2>
    <form method='post'>
      <input type='hidden' name='action' value='save_conversation' />
      <label>Conversation ID</label><br />
      <input type='number' name='conversation_id' min='1' required />
      <br /><br />
      <label>Prompt específico da conversa</label>
      <textarea name='conversation_prompt' required></textarea>
      <br />
      <button type='submit'>Salvar Override da Conversa</button>
    </form>
  </div>

  <div class='card'>
    <h2>Overrides Ativos</h2>
    <table>
      <thead><tr><th>Conversation ID</th><th>Prompt</th><th>Ações</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.get("/api/prompt-config")
def get_prompt_config() -> Response:
    with prompt_config_lock:
        data = {
            "default_prompt": _prompt_config.get("default_prompt", ""),
            "conversation_prompts": dict(_prompt_config.get("conversation_prompts", {})),
        }
    return jsonify(data)


@app.post("/api/prompt-config/default")
def update_default_prompt() -> tuple[Response, int] | Response:
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt")
    try:
        _set_default_prompt(prompt or "")
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400


@app.post("/api/prompt-config/conversation/<int:conversation_id>")
def update_conversation_prompt(conversation_id: int) -> tuple[Response, int] | Response:
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt")
    try:
        key = _set_conversation_prompt(conversation_id, prompt or "")
        return jsonify({"status": "ok", "conversation_id": key})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400


@app.delete("/api/prompt-config/conversation/<int:conversation_id>")
def delete_conversation_prompt(conversation_id: int) -> tuple[Response, int] | Response:
    try:
        key = _delete_conversation_prompt(conversation_id)
        return jsonify({"status": "ok", "conversation_id": key})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400


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


# ---------------------------------------------------------------------------
# Melhoria 8 — inicialização explícita para prontidão de produção.
#
# Em execução local (python app.py), initialize_app() é chamada
# automaticamente.  Em ambientes WSGI/Gunicorn, o deployer deve chamar
# initialize_app() explicitamente no ponto de entrada ou via app factory,
# por exemplo:
#
#   from app import app, initialize_app
#   initialize_app()
#   # gunicorn app:app
#
# Isso evita que threads de ingestão sejam iniciadas durante o import
# do módulo (o que pode causar problemas com forking em Gunicorn preload).
# ---------------------------------------------------------------------------

_initialized = False


def initialize_app() -> None:
    """Inicializa threads de background (ingestão RAG + worker).

    Seguro para chamar múltiplas vezes — só executa na primeira chamada.
    Em WSGI com múltiplos workers, cada worker deve chamar uma vez.
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return
    _initialized = True

    # Ingestão RAG em background (não-bloqueante)
    threading.Thread(target=ingest_documents, daemon=True, name="rag-ingest").start()
    logger.info("RAG ingest thread started")

    # Worker de processamento de jobs (auto-reply)
    threading.Thread(target=worker, daemon=True, name="reply-worker").start()
    logger.info("Reply worker thread started")


if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=8000)
