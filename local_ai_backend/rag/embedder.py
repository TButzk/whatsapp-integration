"""Embed text strings via Ollama's /api/embed endpoint."""

import httpx

from local_ai_backend.config import Settings


class Embedder:
    """Thin wrapper around Ollama's embedding endpoint.

    Keeps a persistent httpx.Client for connection reuse.
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.rag_embed_model
        self._timeout = float(settings.llm_timeout_seconds)
        self._http = http_client or httpx.Client(timeout=self._timeout)

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*.

        Raises ``httpx.HTTPStatusError`` on non-2xx responses.
        """
        response = self._http.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        # Ollama /api/embed response: {"embeddings": [[...]]}
        raw = data.get("embeddings") or data.get("embedding")
        if not raw:
            raise ValueError(f"No embeddings in Ollama response: {data}")
        first = raw[0]
        return first if isinstance(first, list) else raw
