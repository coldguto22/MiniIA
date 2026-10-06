"""Reflexão longa pelo modelo System 2 (Llama via Ollama)."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import ollama

from dante.config import load_models_config
from dante.core.self_model import SelfModel


def build_prompt(thought: str, memories: Iterable[str], self_model: SelfModel) -> str:
    context = "\n---\n".join(memory[:300] for memory in memories)[:1000]
    return (
        "Você é Dante. Reflita em português sobre o pensamento abaixo e conecte-o "
        "às memórias somente quando houver relação genuína.\n"
        f"Auto-modelo: {self_model.identity_summary}\n"
        f"Memórias:\n{context or 'Nenhuma memória relacionada.'}\n\n"
        f"Pensamento:\n{thought[:500]}\n\nReflexão:"
    )


def generate_reflection(
    thought: str,
    memories: Iterable[str],
    self_model: SelfModel,
    *,
    generate: Callable[..., Any] | None = None,
) -> str:
    """Gera reflexão longa com cliente substituível nos testes."""
    client_generate = generate or ollama.generate
    model = load_models_config()["system2"]["model"]
    response = client_generate(
        model=model,
        prompt=build_prompt(thought, memories, self_model),
    )
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(response).strip()