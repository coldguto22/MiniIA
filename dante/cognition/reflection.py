"""Prompts e geração de reflexão pelo modelo System 2 via Ollama."""

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
        "Você é Dante, um sistema local experimental com memória persistente. "
        "Reflita em português sobre o pensamento abaixo. "
        "As memórias abaixo são o seu próprio registro — fale de dentro delas, "
        "não sobre elas. Conecte apenas quando houver relação genuína. "
        "Diferencie observação, inferência e incerteza. Não afirme consciência "
        "ou experiência subjetiva como fato. "
        "Não se apresente como assistente nem proponha recompensas ou tarefas.\n"
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
    config = load_models_config()["system2"]
    options = {
        "temperature": config.get("temperature", 0.7),
        "num_predict": config.get("max_tokens", 2048),
        "repeat_penalty": config.get("repeat_penalty", 1.15),
        "repeat_last_n": config.get("repeat_last_n", 256),
        "stop": ["\nDante:", "\nOtávio:", "\nVocê:"],
    }
    response = client_generate(
        model=config["model"],
        prompt=build_prompt(thought, memories, self_model, valence),
        options=options,
    )
    # Ollama 0.4+ retorna GenerateResponse; versões antigas retornavam dict.
    if hasattr(response, "response"):
        return str(response.response or "").strip()
    if isinstance(response, dict):
        return str(response.get("response", "") or "").strip()
    return str(response).strip()
