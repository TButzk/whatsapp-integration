from local_ai_backend.config import Settings, is_shopify_customer_lookup_enabled
from local_ai_backend.schemas import DependencyStatus, ReadyResponse


def build_ready_response(settings: Settings) -> ReadyResponse:
    shopify_enabled = is_shopify_customer_lookup_enabled(settings)
    shopify_endpoint_ok = bool(settings.shopify_customer_mcp_endpoint.strip())
    shopify_token_ok = bool(settings.shopify_customer_mcp_token.strip())
    shopify_oauth_ok = bool(
        settings.shopify_customer_client_id.strip()
        and settings.shopify_customer_client_secret.strip()
        and settings.shopify_customer_refresh_token.strip()
    )

    dependencies = {
        "redis": DependencyStatus(
            configured=bool(settings.redis_url.strip()),
            detail=settings.redis_url,
        ),
        "postgres": DependencyStatus(
            configured=bool(settings.database_url.strip()),
            detail=settings.database_url,
        ),
        "ollama": DependencyStatus(
            configured=bool(settings.ollama_base_url.strip()),
            detail=settings.ollama_base_url,
        ),
        "shopify_customer_lookup": DependencyStatus(
            configured=(not shopify_enabled) or (shopify_endpoint_ok and (shopify_token_ok or shopify_oauth_ok)),
            detail=(
                "disabled"
                if not shopify_enabled
                else (
                    "enabled:token"
                    if (shopify_endpoint_ok and shopify_token_ok)
                    else (
                        "enabled:oauth_refresh"
                        if (shopify_endpoint_ok and shopify_oauth_ok)
                        else "enabled_missing_endpoint_or_auth"
                    )
                )
            ),
        ),
    }

    required_keys = ("redis", "postgres", "ollama")
    overall_ready = all(dependencies[key].configured for key in required_keys)
    return ReadyResponse(
        status="ready" if overall_ready else "degraded",
        service=settings.app_name,
        dependencies=dependencies,
    )