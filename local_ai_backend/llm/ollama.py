from typing import Any

import httpx
from pydantic import BaseModel, Field

from local_ai_backend.config import Settings


class OllamaChatResponse(BaseModel):
    model: str
    content: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


class OllamaClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._http_client = http_client or httpx.Client(timeout=settings.llm_timeout_seconds)

    def close(self) -> None:
        self._http_client.close()

    def chat(
        self,
        *,
        system_prompt: str,
        user_message: str,
        use_smart_model: bool = False,
    ) -> OllamaChatResponse:
        model = (
            self._settings.ollama_model_smart
            if use_smart_model
            else self._settings.ollama_model_fast
        )
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "temperature": self._settings.llm_temperature,
                "num_predict": self._settings.llm_max_tokens,
            },
        }
        response = self._http_client.post(
            f"{self._settings.ollama_base_url.rstrip('/')}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message") or {}
        content = str(message.get("content") or "").strip()
        return OllamaChatResponse(model=model, content=content, raw_response=body)