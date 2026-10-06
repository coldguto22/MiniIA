import os

import pytest

import ollama


def _has_model(model_name: str) -> bool:
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
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if name:
            names.add(str(name))
    return model_name in names


@pytest.mark.integration
@pytest.mark.ollama
def test_llama_gera_reflexao_curta_quando_habilitado():
    if os.getenv("RUN_OLLAMA_TESTS", "0") != "1":
        pytest.skip("Defina RUN_OLLAMA_TESTS=1 para executar o Llama local")
    if not _has_model("llama3.1:8b"):
        pytest.skip("Modelo llama3.1:8b não está instalado no Ollama")

    try:
        response = ollama.generate(
            model="llama3.1:8b",
            prompt=(
                "Responda em português com uma única frase curta sobre como "
                "uma memória pode mudar a interpretação de uma observação."
            ),
            options={"temperature": 0.1, "num_predict": 64},
        )
    except Exception as exc:
        pytest.skip(f"Llama local indisponível durante geração: {exc}")

    text = response.get("response", "") if isinstance(response, dict) else str(response)
    assert len(text.strip()) >= 10