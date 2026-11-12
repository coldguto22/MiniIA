import asyncio
import os

import pytest

import ollama
from dante.config import build_model_for, load_models_config
from dante.core.self_model import SelfModel
from dante.cognition import reflection, thought


def _installed_model_names() -> set[str]:
    try:
        response = ollama.list()
    except Exception as exc:
        pytest.skip(f"Ollama indisponível: {exc}")
    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", [])
    names = set()
    for item in models:
        name = (
            item.get("name")
            if isinstance(item, dict)
            else getattr(item, "model", getattr(item, "name", None))
        )
        if name:
            names.add(str(name))
    return names


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.parametrize("role", ["system1", "system2"])
def test_modelos_configurados_geram_resposta_curta_no_ollama(role):
    if os.getenv("RUN_OLLAMA_TESTS", "0") != "1":
        pytest.skip("Defina RUN_OLLAMA_TESTS=1 para executar modelos locais")

    model_name = load_models_config()[role]["model"]
    if model_name not in _installed_model_names():
        pytest.skip(f"Modelo {model_name} não está instalado no Ollama")

    try:
        model = build_model_for(role)
        text = asyncio.run(
            model.generate(
                "Responda em português com uma única frase curta sobre como "
                "uma memória pode mudar a interpretação de uma observação.",
                max_tokens=64,
                temperature=0.1,
            )
        )
    except Exception as exc:
        pytest.fail(f"Ollama falhou ao gerar com {role} ({model_name}): {exc}")

    assert len(text.strip()) >= 10


@pytest.mark.integration
@pytest.mark.ollama
def test_prompts_modulares_de_pensamento_e_reflexao_com_ollama(monkeypatch):
    if os.getenv("RUN_OLLAMA_TESTS", "0") != "1":
        pytest.skip("Defina RUN_OLLAMA_TESTS=1 para executar modelos locais")

    installed = _installed_model_names()
    model_name = "llama3.2:latest"
    if model_name not in installed:
        pytest.skip(f"Modelo {model_name} não está instalado no Ollama")

    config = {
        "system1": {"model": model_name, "max_tokens": 96, "temperature": 0.1},
        "system2": {"model": model_name, "max_tokens": 96, "temperature": 0.1},
    }
    monkeypatch.setattr(thought, "load_models_config", lambda: config)
    monkeypatch.setattr(reflection, "load_models_config", lambda: config)

    generated_thought = thought.generate_thought(
        "VS Code aberto com um arquivo Python que define uma função de teste.",
        SelfModel(),
    )
    generated_reflection = reflection.generate_reflection(
        generated_thought,
        ["Em outra observação, Dante revisou testes automatizados em Python."],
        SelfModel(),
    )

    assert len(generated_thought) >= 10
    assert len(generated_reflection) >= 10
