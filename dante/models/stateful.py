"""Backend-neutral stateful generation session."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .checkpoint import CheckpointError, ModelCheckpoint, load_checkpoint, save_checkpoint


class StatefulModel(Protocol):
    model_name: str
    backend: str

    async def generate_with_state(
        self,
        prompt: str,
        *,
        session_id: str,
        checkpoint: ModelCheckpoint | None = None,
        **kwargs: Any,
    ) -> tuple[str, ModelCheckpoint]:
        ...

    def validate_checkpoint(self, checkpoint: ModelCheckpoint, session_id: str) -> None:
        ...


class SessionStateError(RuntimeError):
    """Raised when a stateful session cannot continue safely."""


class ModelSession:
    """Owns one isolated stateful model trajectory."""

    def __init__(self, model: StatefulModel, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.model = model
        self.session_id = session_id
        self._checkpoint: ModelCheckpoint | None = None

    @property
    def checkpoint(self) -> ModelCheckpoint | None:
        return self._checkpoint

    def reset(self) -> None:
        """Discard the in-memory trajectory and start from an empty state."""
        self._checkpoint = None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        try:
            response, checkpoint = await self.model.generate_with_state(
                prompt,
                session_id=self.session_id,
                checkpoint=self._checkpoint,
                **kwargs,
            )
        except (CheckpointError, ValueError):
            raise
        except RuntimeError as exc:
            raise SessionStateError("stateful generation failed") from exc
        self._validate_checkpoint(checkpoint)
        self._checkpoint = checkpoint
        return response

    def save(self, path: str | Path) -> None:
        if self._checkpoint is None:
            raise SessionStateError("cannot save a session without generated state")
        save_checkpoint(path, self._checkpoint)

    def restore(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(
            path,
            expected_model_name=self.model.model_name,
            expected_backend=self.model.backend,
            expected_session_id=self.session_id,
        )
        self._validate_checkpoint(checkpoint)
        self._checkpoint = checkpoint

    def _validate_checkpoint(self, checkpoint: ModelCheckpoint) -> None:
        if checkpoint.session_id != self.session_id:
            raise CheckpointError("checkpoint session does not match the requested session")
        self.model.validate_checkpoint(checkpoint, self.session_id)
