"""Persistência local e atômica dos estados internos do Dante."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_state(path: str | Path, factory: Any, state_type: Any) -> Any:
    """Carrega um estado JSON ou retorna seu valor inicial se ainda não existe."""
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return factory()
        return state_type.from_dict(payload)
    except (OSError, ValueError, TypeError, KeyError):
        return factory()


def save_state(path: str | Path, state: Any) -> None:
    """Grava o estado via arquivo temporário para evitar JSON parcialmente escrito."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
