from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .ollama_model import OllamaModel

logger = logging.getLogger(__name__)


class MambaModel(OllamaModel):
    def __init__(self, model_name: str, fallback_model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, fallback_model=fallback_model, **kwargs)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            result = await super().generate(prompt, **kwargs)
            if result:
                return result
        except Exception as exc:  # pragma: no cover - defensive fallback path
            logger.warning("Mamba falhou para %s; tentando fallback %s. Erro: %s", self.model_name, self.fallback_model, exc)

        if not self.fallback_model:
            raise RuntimeError(f"Mamba model {self.model_name} failed and no fallback was configured.")

        fallback = OllamaModel(model_name=self.fallback_model, client=self.client)
        logger.warning("Usando fallback Qwen para o prompt do System 1.")
        return await fallback.generate(prompt, **kwargs)

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        try:
            async for chunk in super().stream(prompt, **kwargs):
                yield chunk
                return
        except Exception as exc:  # pragma: no cover - defensive fallback path
            logger.warning("Streaming do Mamba falhou para %s; erro: %s", self.model_name, exc)

        if not self.fallback_model:
            return

        fallback = OllamaModel(model_name=self.fallback_model, client=self.client)
        async for chunk in fallback.stream(prompt, **kwargs):
            yield chunk
