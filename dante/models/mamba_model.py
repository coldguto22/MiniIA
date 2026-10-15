from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .checkpoint import CheckpointError, ModelCheckpoint
from .ollama_model import OllamaModel

logger = logging.getLogger(__name__)


class StateNotSupportedError(RuntimeError):
    """Raised when the configured backend does not return a serializable state."""


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

    async def generate_with_state(
        self,
        prompt: str,
        *,
        session_id: str,
        checkpoint: ModelCheckpoint | None = None,
        **kwargs: Any,
    ) -> tuple[str, ModelCheckpoint]:
        """Generate with Ollama's serializable context continuation.

        Ollama context is an experimental continuation mechanism, not direct
        access to a Mamba recurrent state. The distinction is retained in the
        checkpoint state_type for later backend replacement.
        """
        if checkpoint is not None:
            self._validate_checkpoint(checkpoint, session_id)
            context = checkpoint.payload
        else:
            context = None

        payload = {**self.kwargs, **kwargs}
        if context is not None:
            payload["context"] = context

        try:
            response = self.client.generate(model=self.model_name, prompt=prompt, **payload)
        except Exception as exc:
            raise StateNotSupportedError(f"stateful generation failed for {self.model_name}") from exc

        text = self._response_field(response, "response")
        updated_context = self._response_field(response, "context")
        if not isinstance(updated_context, list) or not all(isinstance(token, int) for token in updated_context):
            raise StateNotSupportedError("Ollama did not return a serializable context")

        updated_checkpoint = ModelCheckpoint(
            model_name=self.model_name,
            backend="ollama",
            session_id=session_id,
            state_type="ollama-context",
            payload=updated_context,
            metadata={"experimental": True, "state_scope": "prompt-continuation"},
        )
        return text.strip(), updated_checkpoint

    def _validate_checkpoint(self, checkpoint: ModelCheckpoint, session_id: str) -> None:
        if checkpoint.model_name != self.model_name:
            raise CheckpointError("checkpoint model does not match the Mamba model")
        if checkpoint.backend != "ollama" or checkpoint.state_type != "ollama-context":
            raise CheckpointError("checkpoint is not an Ollama context checkpoint")
        if checkpoint.session_id != session_id:
            raise CheckpointError("checkpoint session does not match the requested session")
        if not isinstance(checkpoint.payload, list) or not all(isinstance(token, int) for token in checkpoint.payload):
            raise CheckpointError("checkpoint context must be a list of integers")

    @staticmethod
    def _response_field(response: Any, field_name: str) -> Any:
        if isinstance(response, dict):
            return response.get(field_name)
        return getattr(response, field_name, None)

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
