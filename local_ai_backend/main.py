import asyncio
import json
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response, status

from local_ai_backend.config import Settings, get_settings
from local_ai_backend.integrations.shopify import ShopifyIntegration
from local_ai_backend.llm.ollama import OllamaClient
from local_ai_backend.menu import BACK_HINT, MAIN_MENU_TEXT, MenuResult, handle_menu_message
from local_ai_backend.observability.sanitization import sanitize_text
from local_ai_backend.processing import process_message as process_message_content
from local_ai_backend.queueing import InMemoryMessagePublisher, MessagePublisher, QueueMessage
from local_ai_backend.rag import Embedder, QAStore, find_best_answer
from local_ai_backend.schemas import (
    AcceptedResponse,
    AdminKnowledgeImportRequest,
    ChatwootWebhookPayload,
    HealthResponse,
    ImportQARequest,
    ImportQAResponse,
    InternalProcessMessageRequest,
    LlmStatusResponse,
    ProcessingResponse,
    QAPairListItem,
    QAPairListResponse,
    ReadyResponse,
    ChatwootContact,
    ShopifyCustomerOAuthAuthorizeResponse,
    ShopifyCustomerOAuthExchangeRequest,
    ShopifyCustomerOAuthExchangeResponse,
    WhatsAppWebhookPayload,
)
from local_ai_backend.security import verify_hmac_signature
from local_ai_backend.services.chatwoot_client import ChatwootClient
from local_ai_backend.services.conversation_state import ConversationStateStore
from local_ai_backend.services.customer_identity import extract_customer_identity
from local_ai_backend.services.health import build_ready_response


logger = logging.getLogger("local-ai-backend")

# Uvicorn configures only its own loggers; the root logger has no handlers by
# default, so INFO messages from this app would be silently dropped.  We add
# a StreamHandler here so logs are always visible regardless of log_config.
_log_formatter = logging.Formatter("%(levelname)-8s  %(name)s - %(message)s")

if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_log_formatter)
    logger.addHandler(_console_handler)
    logger.propagate = False

# File handler – rotates at 10 MB, keeps 5 backups.
_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_log_dir, exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(_log_dir, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")
)
logger.addHandler(_file_handler)

# Sub-loggers (rag, etc.) inherit from the parent if not separately configured
logging.getLogger("local-ai-backend.rag").parent = logger


def _autoload_qa_from_file(
    path: str,
    store: "QAStore",
    embedder: "Embedder",
    settings: "Settings",
) -> int:
    """Synchronous: embed and upsert all Q&A pairs from a JSON file.

    Skips the load if the vectorstore collection already has data.
    Returns the number of pairs imported (0 if skipped).
    """
    if store.count > 0:
        logger.info(
            "rag_autoload_skipped collection_already_has=%d pairs — "
            "call POST /admin/knowledge/import-qa to force re-import",
            store.count,
        )
        return 0

    if not os.path.isfile(path):
        logger.warning("rag_autoload_file_not_found path=%s", path)
        return 0

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    imported = 0
    for pair in raw:
        qa_id = pair.get("id", "")
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        if not qa_id or not question or not answer:
            logger.warning("rag_autoload_skip_invalid_pair id=%s", qa_id)
            continue
        texts = [question] + [a.strip() for a in pair.get("aliases", []) if a.strip()]
        try:
            embeddings = [embedder.embed(t) for t in texts]
        except Exception as exc:
            logger.warning("rag_autoload_embed_failed qa_id=%s: %s", qa_id, exc)
            continue
        store.upsert_pair(
            qa_id=qa_id,
            question=question,
            answer=answer,
            category=pair.get("category", "geral"),
            texts=texts,
            embeddings=embeddings,
        )
        imported += 1

    logger.info("rag_autoload_complete imported=%d file=%s", imported, path)
    return imported


def create_app(
    settings: Settings | None = None,
    publisher: MessagePublisher | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        # Set log level from settings (the handler already exists from module init)
        _lvl = getattr(logging, app_settings.log_level.upper(), logging.INFO)
        logger.setLevel(_lvl)

        path = (app_settings.rag_qa_autoload_path or "").strip()
        if (
            path
            and app_settings.rag_enabled
            and fastapi_app.state.rag_store is not None
            and fastapi_app.state.rag_embedder is not None
        ):
            await asyncio.to_thread(
                _autoload_qa_from_file,
                path,
                fastapi_app.state.rag_store,
                fastapi_app.state.rag_embedder,
                app_settings,
            )

        # Log startup summary
        rag_status = "disabled"
        if app_settings.rag_enabled:
            rag_status = (
                f"enabled — {fastapi_app.state.rag_store.count} pairs indexed"
                if fastapi_app.state.rag_store is not None
                else "enabled but store failed to initialize"
            )
        logger.info(
            "=== %s started — auto_reply=%s  rag=%s ===",
            app_settings.app_name,
            app_settings.auto_reply_enabled,
            rag_status,
        )

        yield  # server starts accepting requests here

    app = FastAPI(title=app_settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.publisher = publisher or InMemoryMessagePublisher()
    app.state.llm_client = OllamaClient(app_settings)
    app.state.chatwoot_client = ChatwootClient(app_settings)
    app.state.conv_state_store = ConversationStateStore()
    app.state.shopify_integration = ShopifyIntegration(app_settings)
    if app_settings.rag_enabled:
        try:
            app.state.rag_embedder = Embedder(app_settings)
            app.state.rag_store = QAStore(app_settings)
        except Exception as exc:
            logger.warning("rag_init_failed — RAG disabled for this session: %s", exc)
            app.state.rag_embedder = None
            app.state.rag_store = None
    else:
        app.state.rag_embedder = None
        app.state.rag_store = None

    def _resolve_account_id(payload: ChatwootWebhookPayload) -> int | None:
        if payload.account_id is not None:
            return payload.account_id
        if payload.account and payload.account.id is not None:
            return payload.account.id
        return None

    def _is_incoming_message(payload: ChatwootWebhookPayload) -> bool:
        msg_type = payload.message_type
        if isinstance(msg_type, int):
            return msg_type == 0
        if isinstance(msg_type, str):
            return msg_type.lower() == "incoming"
        return True

    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks emitted by qwen3 reasoning models."""
        if "</think>" in text:
            return text.split("</think>", 1)[-1].strip()
        return text

    def _llm_call(
        system_prompt: str,
        user_message: str,
        conversation_id: str | None,
        label: str = "LLM",
    ) -> str | None:
        """Single wrapper for every LLM inference — logs in/out, strips thinking tokens."""
        logger.info(
            "%s_IN   conv=%s | SYS: %.150s | USR: %.150s",
            label,
            conversation_id,
            sanitize_text(system_prompt),
            sanitize_text(user_message),
        )
        try:
            resp = app.state.llm_client.chat(
                system_prompt=system_prompt,
                user_message=user_message,
                use_smart_model=False,
            )
            text = _strip_think((resp.content or "").strip())
            logger.info(
                "%s_OUT  conv=%s | %.250s",
                label,
                conversation_id,
                sanitize_text(text),
            )
            return text or None
        except Exception as exc:
            logger.warning("%s_FAIL conv=%s: %s", label, conversation_id, exc)
            return None

    def _validate_rag_match(
        question: str, answer: str, conversation_id: str | None = None
    ) -> bool:
        """Ask the LLM whether the RAG candidate answer is relevant to the question.

        Returns True  → relevant, safe to send directly.
        Returns False → false positive; discard and fall through to LLM generation.
        Fail-open: if the validation call itself errors, allow the answer through.
        """
        system = (
            "Você é um validador de respostas de atendimento ao cliente. "
            "Analise se a RESPOSTA CANDIDATA é realmente relevante e correta "
            "para a PERGUNTA DO CLIENTE. "
            "Responda SOMENTE com SIM ou NAO — nenhuma outra palavra."
        )
        user_msg = (
            f"PERGUNTA DO CLIENTE: {question}\n\n"
            f"RESPOSTA CANDIDATA: {answer}"
        )
        verdict = _llm_call(system, user_msg, conversation_id, label="RAG_VALIDATE")
        if verdict is None:
            logger.warning("RAG_VALIDATE_FAIL conv=%s — allowing answer", conversation_id)
            return True
        is_valid = verdict.upper().startswith("SIM")
        logger.info(
            "RAG_VALIDATE_RESULT conv=%s verdict=%r valid=%s",
            conversation_id, verdict[:30], is_valid,
        )
        return is_valid

    def _generate_auto_reply(
        content: str,
        conversation_id: str | None = None,
        contact: ChatwootContact | None = None,
    ) -> str:
        state = (
            app.state.conv_state_store.get(str(conversation_id))
            if conversation_id is not None
            else None
        )
        previous_flow = state.flow_state if state is not None else None

        if state is not None:
            menu_result = handle_menu_message(content, state)
            app.state.conv_state_store.save(state)
        else:
            menu_result = MenuResult(reply=MAIN_MENU_TEXT, action="reply")

        logger.debug(
            "menu action=%s flow=%s conv=%s",
            menu_result.action,
            state.flow_state if state else None,
            conversation_id,
        )

        # Direct reply — no RAG/LLM needed
        if menu_result.action in ("reply", "handoff"):
            return menu_result.reply[: app_settings.max_response_chars]

        # --- query_rag: RAG → validate → LLM → fallback ---
        rag_query = (menu_result.rag_query_override or content).strip()
        rag_hint: str | None = None

        if previous_flow in {"PEDIDO_STATUS", "TRACKING"}:
            identity = extract_customer_identity(content, contact)
            shopify_result = None
            lookup_attempted = False

            logger.info(
                "ORDER_LOOKUP_IDENT conv=%s has_order=%s has_email=%s has_phone=%s has_cpf=%s",
                conversation_id,
                bool(identity.order_number),
                bool(identity.email),
                bool(identity.phone),
                bool(identity.cpf),
            )

            if identity.order_number:
                lookup_attempted = True
                shopify_result = app.state.shopify_integration.get_order_status(identity.order_number)
            elif identity.has_identifier():
                lookup_attempted = True
                shopify_result = app.state.shopify_integration.get_customer_purchase_summary(
                    content,
                    email=identity.email,
                    phone=identity.phone,
                    cpf=identity.cpf,
                )

            if shopify_result:
                if shopify_result.status == "ambiguous":
                    summary = str((shopify_result.data or {}).get("summary") or "").strip()
                    response = (
                        "Encontrei mais de um pedido para esses dados. "
                        "Para confirmar, me informe o numero do pedido "
                        "(ou data da compra)."
                    )
                    if summary:
                        response = f"{response}\n\n{summary}"
                    return response[: app_settings.max_response_chars]

                if shopify_result.status in {"not_found", "invalid_input"}:
                    return (
                        "Nao localizei pedido com os dados informados. "
                        "Me envie o numero do pedido, ou confirme o e-mail/telefone do cadastro."
                    )[: app_settings.max_response_chars]

                if shopify_result.status == "ok" and shopify_result.data:
                    summary = str(
                        shopify_result.data.get("summary")
                        or shopify_result.data.get("raw")
                        or ""
                    ).strip()
                    if summary:
                        rag_query = (
                            f"{rag_query}\n\n"
                            f"[CONTEXTO_STATUS_PEDIDO]\n"
                            f"{summary}"
                        )

            if lookup_attempted and (shopify_result is None):
                return (
                    "Nao consegui consultar o status agora. "
                    "Tente novamente em instantes ou me envie o numero do pedido para confirmar."
                )[: app_settings.max_response_chars]

        logger.info(
            "RAG_QUERY conv=%s | %s",
            conversation_id,
            sanitize_text(rag_query)[:200],
        )

        if app.state.rag_store is not None and app.state.rag_embedder is not None:
            try:
                hit = find_best_answer(
                    rag_query,
                    app.state.rag_store,
                    app.state.rag_embedder,
                    app_settings,
                )
                if hit is not None:
                    logger.info(
                        "RAG_HIT  conv=%s score=%.3f qa_id=%s | Q: %.80s | A: %.80s",
                        conversation_id, hit.score, hit.qa_id,
                        hit.question, hit.answer,
                    )
                    if hit.score >= app_settings.rag_min_score_direct:
                        if _validate_rag_match(rag_query, hit.answer, conversation_id):
                            logger.info(
                                "RAG_DIRECT_OK conv=%s qa_id=%s score=%.3f",
                                conversation_id, hit.qa_id, hit.score,
                            )
                            return (hit.answer + BACK_HINT)[: app_settings.max_response_chars]
                        logger.info(
                            "RAG_DIRECT_REJECTED conv=%s qa_id=%s — falling to LLM",
                            conversation_id, hit.qa_id,
                        )
                        # Don't use the rejected hit as a hint either
                    else:
                        rag_hint = hit.answer
                        logger.info(
                            "RAG_HINT conv=%s score=%.3f qa_id=%s",
                            conversation_id, hit.score, hit.qa_id,
                        )
                else:
                    logger.info("RAG_MISS conv=%s | no hit above min threshold", conversation_id)
            except Exception as exc:
                logger.warning("rag_lookup_failed: %s", exc)
        else:
            logger.info("RAG_SKIP conv=%s | RAG not initialised", conversation_id)

        system_prompt = (
            "Voce e um atendente de loja em portugues do Brasil. "
            "Seja curto, claro e nao invente dados. "
            "Se faltar informacao, peca os dados necessarios."
        )
        if rag_hint:
            system_prompt = (
                "Voce e um atendente de loja em portugues do Brasil. "
                "Reformule a seguinte resposta oficial de forma natural e curta, "
                "sem inventar nenhum dado novo:\n\n"
                f"Resposta oficial: {rag_hint}"
            )

        text = _llm_call(system_prompt, rag_query, conversation_id)
        if text:
            return (text + BACK_HINT)[: app_settings.max_response_chars]

        fallback = menu_result.reply or MAIN_MENU_TEXT
        return fallback[: app_settings.max_response_chars]

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=app_settings.app_name)

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        return build_ready_response(app_settings)

    @app.get("/admin/llm/status", response_model=LlmStatusResponse)
    async def llm_status() -> LlmStatusResponse:
        return LlmStatusResponse(
            base_url=app_settings.ollama_base_url,
            fast_model=app_settings.ollama_model_fast,
            smart_model=app_settings.ollama_model_smart,
            timeout_seconds=app_settings.llm_timeout_seconds,
            max_tokens=app_settings.llm_max_tokens,
            temperature=app_settings.llm_temperature,
        )

    @app.get(
        "/admin/shopify/customer-oauth/authorize-url",
        response_model=ShopifyCustomerOAuthAuthorizeResponse,
    )
    async def shopify_customer_oauth_authorize_url(
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> ShopifyCustomerOAuthAuthorizeResponse:
        result = app.state.shopify_integration.create_customer_oauth_authorization(
            redirect_uri=redirect_uri,
            state=state,
        )
        if result.status != "ok" or not result.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="shopify_customer_oauth_not_configured",
            )
        return ShopifyCustomerOAuthAuthorizeResponse(**result.data)

    @app.post(
        "/admin/shopify/customer-oauth/exchange-code",
        response_model=ShopifyCustomerOAuthExchangeResponse,
    )
    async def shopify_customer_oauth_exchange_code(
        payload: ShopifyCustomerOAuthExchangeRequest,
    ) -> ShopifyCustomerOAuthExchangeResponse:
        result = app.state.shopify_integration.exchange_customer_oauth_code(
            code=payload.code,
            redirect_uri=payload.redirect_uri,
            code_verifier=payload.code_verifier,
        )
        if result.status != "ok" or not result.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(result.data or {}).get("error", "shopify_customer_oauth_exchange_failed"),
            )
        return ShopifyCustomerOAuthExchangeResponse(ok=True, **result.data)

    @app.get("/admin/shopify/customer-oauth/callback")
    async def shopify_customer_oauth_callback(
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> Response:
        if error or not code:
            desc = error_description or error or "unknown"
            html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>OAuth – Erro</title></head>
<body style="font-family:monospace;padding:2rem">
<h2 style="color:red">&#10060; Erro OAuth</h2>
<pre>{desc}</pre>
</body></html>"""
            return Response(content=html, media_type="text/html", status_code=400)

        # Retrieve stored code_verifier from authorize step via state lookup is
        # not needed here – code_verifier was kept in the script/session.
        # For the automated callback flow we attempt exchange without verifier
        # first; if PKCE is required the user should use the manual script.
        result = app.state.shopify_integration.exchange_customer_oauth_code(
            code=code,
            redirect_uri=app_settings.shopify_customer_redirect_uri or None,
        )

        if result.status == "ok" and result.data and result.data.get("access_token"):
            tok = result.data["access_token"]
            ref = result.data.get("refresh_token") or ""
            scope = result.data.get("scope") or ""
            html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>OAuth – Sucesso</title></head>
<body style="font-family:monospace;padding:2rem">
<h2 style="color:green">&#10003; Autorizado com sucesso!</h2>
<p><b>Access token</b> salvo em mem&oacute;ria. Cole tamb&eacute;m no .env:</p>
<pre style="background:#f4f4f4;padding:1rem">SHOPIFY_CUSTOMER_MCP_TOKEN={tok}</pre>
<pre style="background:#f4f4f4;padding:1rem">SHOPIFY_CUSTOMER_REFRESH_TOKEN={ref}</pre>
<p>Scope: {scope}</p>
<p><small>Reinicie a API para persistir via .env.</small></p>
</body></html>"""
            return Response(content=html, media_type="text/html")

        err_detail = (result.data or {}).get("error", "exchange_failed")
        html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><title>OAuth – Falha</title></head>
<body style="font-family:monospace;padding:2rem">
<h2 style="color:orange">&#9888; Troca de token falhou</h2>
<pre>{err_detail}</pre>
<p>Tente usar o script run-shopify-customer-oauth.ps1 com PKCE.</p>
</body></html>"""
        return Response(content=html, media_type="text/html", status_code=400)

    @app.post(
        "/auto-reply/webhook",
        response_model=AcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @app.post(
        "/webhook",
        response_model=AcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @app.post(
        "/webhook/chatwoot",
        response_model=AcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def webhook_chatwoot(
        request: Request,
        payload: ChatwootWebhookPayload,
        x_chatwoot_signature: str | None = Header(default=None),
    ) -> AcceptedResponse:
        body = await request.body()
        if not verify_hmac_signature(
            body,
            x_chatwoot_signature,
            app_settings.chatwoot_webhook_secret,
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_signature")

        external_message_id = str(
            payload.message.id if payload.message and payload.message.id is not None else payload.conversation.id
        )
        queued_message = QueueMessage(
            channel="chatwoot",
            external_message_id=external_message_id,
            conversation_id=str(payload.conversation.id),
            content=payload.content or (payload.message.content if payload.message else ""),
            metadata={"event": payload.event, "private": payload.private},
        )
        message_id = app.state.publisher.publish(queued_message)

        content = (queued_message.content or "").strip()
        account_id = _resolve_account_id(payload)
        conversation_id = payload.conversation.id
        can_send_reply = app.state.chatwoot_client.can_send_reply()

        logger.info(
            "chatwoot_webhook_received account_id=%s conversation_id=%s message_type=%s private=%s auto_reply_enabled=%s can_send_reply=%s",
            account_id,
            conversation_id,
            payload.message_type,
            payload.private,
            app_settings.auto_reply_enabled,
            can_send_reply,
        )

        if (
            app_settings.auto_reply_enabled
            and _is_incoming_message(payload)
            and not payload.private
            and content
            and account_id is not None
            and can_send_reply
        ):
            pre_flow = app.state.conv_state_store.get(str(conversation_id)).flow_state
            logger.info(
                "MSG_IN  conv=%s flow=%s | %s",
                conversation_id,
                pre_flow,
                sanitize_text(content)[:300],
            )
            reply_text = _generate_auto_reply(
                content,
                conversation_id=str(conversation_id),
                contact=payload.contact,
            )
            if reply_text:
                logger.info(
                    "MSG_OUT conv=%s | %s",
                    conversation_id,
                    sanitize_text(reply_text)[:300],
                )
                try:
                    app.state.chatwoot_client.send_conversation_reply(
                        account_id=account_id,
                        conversation_id=conversation_id,
                        content=reply_text,
                    )
                except Exception as exc:
                    logger.warning("Failed to post reply to Chatwoot: %s", exc)
        else:
            logger.info(
                "chatwoot_webhook_reply_skipped incoming=%s has_content=%s has_account_id=%s",
                _is_incoming_message(payload),
                bool(content),
                account_id is not None,
            )

        return AcceptedResponse(
            channel="chatwoot",
            reason=f"message accepted for conversation {payload.conversation.id}",
            message_id=message_id,
        )

    @app.post(
        "/webhook/whatsapp",
        response_model=AcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def webhook_whatsapp(
        request: Request,
        payload: WhatsAppWebhookPayload,
        x_webhook_signature: str | None = Header(default=None),
    ) -> AcceptedResponse:
        body = await request.body()
        if not verify_hmac_signature(
            body,
            x_webhook_signature,
            app_settings.whatsapp_webhook_secret,
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_signature")

        queued_message = QueueMessage(
            channel="whatsapp",
            external_message_id=payload.message_id,
            conversation_id=payload.metadata.get("conversation_id") if payload.metadata else None,
            content=payload.content,
            metadata=payload.metadata,
        )
        message_id = app.state.publisher.publish(queued_message)
        return AcceptedResponse(
            channel="whatsapp",
            reason=f"message accepted from {payload.from_number}",
            message_id=message_id,
        )

    @app.post(
        "/internal/process-message",
        response_model=ProcessingResponse,
        status_code=status.HTTP_200_OK,
    )
    async def process_message(
        payload: InternalProcessMessageRequest,
    ) -> ProcessingResponse:
        result = process_message_content(payload.content)
        return ProcessingResponse(
            intent=result.intent.value,
            confidence=result.confidence,
            action=result.action.value,
            reason=result.reason,
            customer_message=result.customer_message,
            audit_reasons=result.audit_reasons,
        )

    @app.post(
        "/admin/knowledge/import",
        response_model=AcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def import_knowledge(
        payload: AdminKnowledgeImportRequest,
    ) -> AcceptedResponse:
        reason = "knowledge import scheduled"
        if payload.source_paths:
            reason = f"knowledge import scheduled for {len(payload.source_paths)} source path(s)"
        return AcceptedResponse(channel="admin", reason=reason)

    @app.post(
        "/admin/knowledge/import-qa",
        response_model=ImportQAResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_qa(payload: ImportQARequest) -> ImportQAResponse:
        if not app_settings.rag_enabled or app.state.rag_store is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rag_not_enabled",
            )
        imported = 0
        for pair in payload.pairs:
            all_texts = [pair.question] + pair.aliases
            try:
                embeddings = [app.state.rag_embedder.embed(t) for t in all_texts]
            except Exception as exc:
                logger.warning("embed_failed qa_id=%s: %s", pair.id, exc)
                continue
            app.state.rag_store.upsert_pair(
                qa_id=pair.id,
                question=pair.question,
                answer=pair.answer,
                category=pair.category,
                texts=all_texts,
                embeddings=embeddings,
            )
            imported += 1
        logger.info("import_qa imported=%d collection=%s", imported, app_settings.rag_collection_name)
        return ImportQAResponse(
            imported=imported,
            collection=app_settings.rag_collection_name,
        )

    @app.get(
        "/admin/knowledge/qa-pairs",
        response_model=QAPairListResponse,
    )
    async def list_qa_pairs() -> QAPairListResponse:
        if not app_settings.rag_enabled or app.state.rag_store is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rag_not_enabled",
            )
        stored = app.state.rag_store.list_all()
        return QAPairListResponse(
            total=len(stored),
            pairs=[
                QAPairListItem(
                    qa_id=p.qa_id,
                    question=p.question,
                    answer=p.answer,
                    category=p.category,
                )
                for p in stored
            ],
        )

    return app


app = create_app()