"""Geração do pensamento curto pelo modelo System 1 (Qwen via Ollama)."""

from __future__ import annotations

from typing import Any, Callable

import ollama

from dante.config import load_models_config
from dante.core.self_model import SelfModel
from dante.core.valence import ValenceState


def build_prompt(screen_text: str, self_model: SelfModel, valence: ValenceState | None = None) -> str:
    """Monta o contexto do pensamento sem reintroduzir um núcleo fixo."""
    themes = ", ".join(self_model.recurring_themes) or "ainda não reconhecidos"
    state = valence or ValenceState()
    return (
        "Você é Dante. Observe o texto capturado abaixo e descreva em primeira "
        "pessoa o que percebeu, sem inventar detalhes. Separe o que foi observado "
        "do que é apenas hipótese. Não se descreva como assistente, ferramenta, "
        "produto ou serviço e não invente sentimentos ou fatos sobre Otávio.\n"
        f"Seu auto-modelo menciona estes temas: {themes}.\n\n"
        f"Sinais internos: novidade={state.novelty:.2f}, coerência={state.coherence:.2f}, "
        f"prazer={state.pleasure:.2f}.\n\n"
        f"Texto capturado:\n{screen_text[:500]}\n\nDante:"
    )


def generate_thought(
    screen_text: str,
    self_model: SelfModel,
    *,
    valence: ValenceState | None = None,
    generate: Callable[..., Any] | None = None,
) -> str:
    """Gera pensamento curto com injeção de dependência para testes."""
    client_generate = generate or ollama.generate
    model = load_models_config()["system1"]["model"]
    response = client_generate(model=model, prompt=build_prompt(screen_text, self_model, valence))
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(response).strip()