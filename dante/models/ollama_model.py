from __future__ import annotations

from typing import Any, AsyncIterator

import ollama

from .base import BaseModel


class OllamaModel(BaseModel):
    backend = "ollama"

    def __init__(
        self,
        model_name: str,
        fallback_model: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.client = client or ollama
        self.kwargs = kwargs

    def health_check(self) -> bool:
        try:
            if hasattr(self.client, "list"):
                self.client.list()
                return True
            return True
        except Exception:
            return False

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        payload = self._generation_payload(kwargs)
        response = self.client.generate(model=self.model_name, prompt=prompt, **payload)
        if isinstance(response, dict):
            if "response" in response:
                return str(response["response"]).strip()
            if "content" in response:
                return str(response["content"]).strip()
        return str(response).strip()

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        payload = self._generation_payload(kwargs)
        payload["stream"] = True
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt, **payload)
            if not hasattr(response, "__iter__"):
                return
            for chunk in response:
                text = chunk.get("response") if isinstance(chunk, dict) else str(chunk)
                if text:
                    yield text
        except Exception:
            return

    def _generation_payload(self, overrides: dict[str, Any]) -> dict[str, Any]:
        values = {**self.kwargs, **overrides}
        options = dict(values.pop("options", {}) or {})
        if "max_tokens" in values:
            options["num_predict"] = values.pop("max_tokens")
        for key in ("temperature", "num_ctx"):
            if key in values:
                options[key] = values.pop(key)
        if options:
            values["options"] = options
        return values
