from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="local-ai-backend", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8010, alias="APP_PORT")
    auto_reply_enabled: bool = Field(default=True, alias="AUTO_REPLY_ENABLED")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    queue_name: str = Field(default="incoming_messages", alias="QUEUE_NAME")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/local_ai_backend",
        alias="DATABASE_URL",
    )

    ollama_base_url: str = Field(default="http://localhost:11435", alias="OLLAMA_BASE_URL")
    ollama_model_fast: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL_FAST")
    ollama_model_smart: str = Field(default="qwen3:14b", alias="OLLAMA_MODEL_SMART")
    llm_timeout_seconds: int = Field(default=45, alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=500, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    chatwoot_base_url: str = Field(default="http://localhost:65271", alias="CHATWOOT_BASE_URL")
    chatwoot_api_token: str = Field(default="", alias="CHATWOOT_API_TOKEN")
    max_response_chars: int = Field(default=1200, alias="MAX_RESPONSE_CHARS")

    chatwoot_webhook_secret: str = Field(default="", alias="CHATWOOT_WEBHOOK_SECRET")
    whatsapp_webhook_secret: str = Field(default="", alias="WHATSAPP_WEBHOOK_SECRET")

    # Shopify Customer Account MCP lookup (order/customer private data)
    shopify_customer_lookup_enabled: bool = Field(
        default=False,
        alias="SHOPIFY_CUSTOMER_LOOKUP_ENABLED",
    )
    shopify_customer_mcp_endpoint: str = Field(default="", alias="SHOPIFY_CUSTOMER_MCP_ENDPOINT")
    shopify_customer_mcp_token: str = Field(default="", alias="SHOPIFY_CUSTOMER_MCP_TOKEN")
    shopify_customer_timeout_seconds: int = Field(default=20, alias="SHOPIFY_CUSTOMER_TIMEOUT_SECONDS")
    # Legacy-compatible timeout name used in auto_reply_bridge
    shopify_customer_mcp_timeout: int = Field(default=0, alias="SHOPIFY_CUSTOMER_MCP_TIMEOUT")
    # Optional OAuth refresh settings (used only when MCP token is absent)
    shopify_customer_oauth_token_endpoint: str = Field(
        default="",
        alias="SHOPIFY_CUSTOMER_OAUTH_TOKEN_ENDPOINT",
    )
    shopify_customer_oauth_authorization_endpoint: str = Field(
        default="",
        alias="SHOPIFY_CUSTOMER_OAUTH_AUTHORIZATION_ENDPOINT",
    )
    shopify_customer_redirect_uri: str = Field(default="", alias="SHOPIFY_CUSTOMER_REDIRECT_URI")
    shopify_customer_oauth_use_pkce: bool = Field(default=True, alias="SHOPIFY_CUSTOMER_OAUTH_USE_PKCE")
    shopify_customer_client_id: str = Field(default="", alias="SHOPIFY_CUSTOMER_CLIENT_ID")
    shopify_customer_client_secret: str = Field(default="", alias="SHOPIFY_CUSTOMER_CLIENT_SECRET")
    shopify_customer_refresh_token: str = Field(default="", alias="SHOPIFY_CUSTOMER_REFRESH_TOKEN")
    shopify_customer_scopes: str = Field(
        default="openid email customer-account-api:full customer-account-mcp-api:full",
        alias="SHOPIFY_CUSTOMER_SCOPES",
    )

    # RAG / Q&A semantic retrieval
    rag_enabled: bool = Field(default=False, alias="RAG_ENABLED")
    rag_embed_model: str = Field(default="nomic-embed-text", alias="RAG_EMBED_MODEL")
    rag_vectorstore_path: str = Field(default="data/vectorstore", alias="RAG_VECTORSTORE_PATH")
    rag_collection_name: str = Field(default="qa_pairs", alias="RAG_COLLECTION_NAME")
    # score thresholds (cosine similarity 0-1)
    rag_min_score_direct: float = Field(default=0.85, alias="RAG_MIN_SCORE_DIRECT")
    rag_min_score_hint: float = Field(default=0.60, alias="RAG_MIN_SCORE_HINT")
    # path to a JSON file with Q&A pairs to seed the vectorstore on first startup
    rag_qa_autoload_path: str = Field(default="", alias="RAG_QA_AUTOLOAD_PATH")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_effective_shopify_customer_timeout(settings: Settings) -> int:
    """Return effective timeout honoring legacy and current env names."""
    if settings.shopify_customer_mcp_timeout > 0:
        return settings.shopify_customer_mcp_timeout
    return max(1, settings.shopify_customer_timeout_seconds)


def is_shopify_customer_lookup_enabled(settings: Settings) -> bool:
    """Return whether customer lookup should be considered enabled.

    Compatibility behavior:
    - explicit flag enables lookup, OR
    - presence of endpoint + direct token enables lookup (legacy-style setup)
    - endpoint + OAuth refresh credentials also enable lookup
    """
    endpoint_ok = bool(settings.shopify_customer_mcp_endpoint.strip())
    token_ok = bool(settings.shopify_customer_mcp_token.strip())
    oauth_ok = bool(
        settings.shopify_customer_client_id.strip()
        and settings.shopify_customer_client_secret.strip()
        and settings.shopify_customer_refresh_token.strip()
    )
    return bool(endpoint_ok and ((settings.shopify_customer_lookup_enabled and (token_ok or oauth_ok)) or token_ok))