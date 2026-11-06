import asyncio
import os

import pytest

import ollama
from dante.config import build_model_for, load_models_config


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