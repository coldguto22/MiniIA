"""Reflexão longa pelo modelo System 2 (Llama via Ollama)."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import ollama

from dante.config import load_models_config
from dante.core.self_model import SelfModel
from dante.core.valence import ValenceState


def build_prompt(
    thought: str,
    memories: Iterable[str],
    self_model: SelfModel,
    valence: ValenceState | None = None,
) -> str:
    context = "\n---\n".join(memory[:300] for memory in memories)[:1000]
    return (
        "Você é Dante. Reflita em português sobre o pensamento abaixo e conecte-o "
        "às memórias somente quando houver relação genuína. Diferencie observação, "
        "inferência e incerteza; não se apresente como assistente nem proponha "
        "recompensas, tarefas ou objetivos externos.\n"
        f"Auto-modelo: {self_model.identity_summary}\n"
        f"Sinais internos: {valence or ValenceState()}\n"
        f"Memórias:\n{context or 'Nenhuma memória relacionada.'}\n\n"
        f"Pensamento:\n{thought[:500]}\n\nReflexão:"
    )


def generate_reflection(
    thought: str,
    memories: Iterable[str],
    self_model: SelfModel,
    *,
    valence: ValenceState | None = None,
    generate: Callable[..., Any] | None = None,
) -> str:
    """Gera reflexão longa com cliente substituível nos testes."""
    client_generate = generate or ollama.generate
    model = load_models_config()["system2"]["model"]
    response = client_generate(
        model=model,
        prompt=build_prompt(thought, memories, self_model, valence),
    )
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(response).strip()