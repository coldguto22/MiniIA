"""Normalização de texto externo antes de memória e modelos locais."""

from __future__ import annotations

import re


_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str | None, *, max_chars: int = 4000) -> str:
    """Remove sequências de terminal, controles e espaço redundante."""
    if not text:
        return ""
    clean = _ANSI_RE.sub(" ", str(text))
    clean = _CONTROL_RE.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max(0, max_chars)]