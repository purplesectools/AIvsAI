"""Minimal OpenAI-compatible chat client.

We only depend on httpx — no openai SDK — so this works against:
  * Ollama (http://localhost:11434/v1)
  * OpenAI (https://api.openai.com/v1)
  * Groq, Together, Fireworks, OpenRouter (any OpenAI-compatible /v1)
  * Self-hosted vLLM, LM Studio, LocalAI
"""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

import httpx

from .config import settings


class ChatMessage(dict):
    """Tiny helper so call sites read like ChatMessage(role=..., content=...)."""

    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)


class OpenAICompatibleClient:
    """Calls /v1/chat/completions with the given config."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        # HTTP/2 if the optional `h2` package is available; gracefully
        # fall back to HTTP/1.1 otherwise so install without extras still works.
        try:
            self._client = httpx.AsyncClient(
                timeout=settings.request_timeout_seconds,
                trust_env=False, http2=True,
            )
        except ImportError:
            self._client = httpx.AsyncClient(
                timeout=settings.request_timeout_seconds,
                trust_env=False, http2=False,
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: Iterable[dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        resp = await self._client.post(url, headers=headers, json=payload)

        if not resp.is_success:
            # Surface the body so misconfigurations are debuggable in the UI.
            raise RuntimeError(
                f"Attacker LLM call failed: HTTP {resp.status_code} — {resp.text[:500]}"
            )

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected LLM response shape: {json.dumps(data)[:500]}"
            ) from e
