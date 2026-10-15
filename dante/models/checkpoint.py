"""Versioned, isolated checkpoints for model state payloads."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is invalid or incompatible."""


@dataclass(frozen=True)
class ModelCheckpoint:
    model_name: str
    backend: str
    session_id: str
    state_type: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "version": CHECKPOINT_VERSION,
            "model_name": self.model_name,
            "backend": self.backend,
            "session_id": self.session_id,
            "state_type": self.state_type,
            "payload": self.payload,
            "metadata": self.metadata,
        }
        body["payload_sha256"] = _payload_digest(body["payload"])
        return body

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCheckpoint":
        if data.get("version") != CHECKPOINT_VERSION:
            raise CheckpointError("unsupported checkpoint version")

        required = ("model_name", "backend", "session_id", "state_type", "payload")
        if any(not data.get(field_name) for field_name in required):
            raise CheckpointError("checkpoint is missing required fields")

        if data.get("payload_sha256") != _payload_digest(data["payload"]):
            raise CheckpointError("checkpoint payload checksum does not match")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise CheckpointError("checkpoint metadata must be an object")

        return cls(
            model_name=str(data["model_name"]),
            backend=str(data["backend"]),
            session_id=str(data["session_id"]),
            state_type=str(data["state_type"]),
            payload=data["payload"],
            metadata=metadata,
        )


def save_checkpoint(path: str | Path, checkpoint: ModelCheckpoint) -> None:
    """Save a checkpoint atomically so interrupted writes do not replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    expected_model_name: str | None = None,
    expected_backend: str | None = None,
    expected_session_id: str | None = None,
) -> ModelCheckpoint:
    """Load and validate a checkpoint against the requested execution context."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("could not read checkpoint") from exc

    if not isinstance(data, dict):
        raise CheckpointError("checkpoint root must be an object")

    checkpoint = ModelCheckpoint.from_dict(data)
    if expected_model_name and checkpoint.model_name != expected_model_name:
        raise CheckpointError("checkpoint model does not match the requested model")
    if expected_backend and checkpoint.backend != expected_backend:
        raise CheckpointError("checkpoint backend does not match the requested backend")
    if expected_session_id and checkpoint.session_id != expected_session_id:
        raise CheckpointError("checkpoint session does not match the requested session")
    return checkpoint


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()