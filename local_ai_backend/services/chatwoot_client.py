import httpx

from local_ai_backend.config import Settings


class ChatwootClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http_client = http_client or httpx.Client(timeout=20.0)

    def close(self) -> None:
        self._http_client.close()

    def can_send_reply(self) -> bool:
        return bool(self._settings.chatwoot_base_url.strip() and self._settings.chatwoot_api_token.strip())

    def send_conversation_reply(self, *, account_id: int, conversation_id: int, content: str) -> dict:
        url = (
            f"{self._settings.chatwoot_base_url.rstrip('/')}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/messages"
        )
        payload = {
            "content": content,
            "message_type": "outgoing",
            "private": False,
        }
        response = self._http_client.post(
            url,
            headers={
                "api_access_token": self._settings.chatwoot_api_token,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "chatwoot_reply_failed "
                f"status={exc.response.status_code} body={exc.response.text[:300]}"
            ) from exc
        return response.json()