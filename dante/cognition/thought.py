"""Geração do pensamento curto pelo modelo System 1 (Qwen via Ollama)."""

from __future__ import annotations

from typing import Any, Callable

import ollama

from dante.config import load_models_config
from dante.core.self_model import SelfModel


def build_prompt(screen_text: str, self_model: SelfModel) -> str:
    """Monta o contexto do pensamento sem reintroduzir um núcleo fixo."""
    themes = ", ".join(self_model.recurring_themes) or "ainda não reconhecidos"
    return (
        "Você é Dante. Observe o texto capturado abaixo e descreva em primeira "
        "pessoa o que percebeu, sem inventar detalhes.\n"
        f"Seu auto-modelo menciona estes temas: {themes}.\n\n"
        f"Texto capturado:\n{screen_text[:500]}\n\nDante:"
    )


def generate_thought(
    screen_text: str,
    self_model: SelfModel,
    *,
    generate: Callable[..., Any] | None = None,
) -> str:
    """Gera pensamento curto com injeção de dependência para testes."""
    client_generate = generate or ollama.generate
    model = load_models_config()["system1"]["model"]
    response = client_generate(model=model, prompt=build_prompt(screen_text, self_model))
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(response).strip()