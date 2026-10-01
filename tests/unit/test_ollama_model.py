import asyncio

import pytest

from dante.models.ollama_model import OllamaModel


class FakeClient:
    def list(self):
        return {"models": [{"name": "fake"}]}

    def generate(self, model, prompt, **kwargs):
        return {"response": f"response::{model}::{prompt}"}


@pytest.mark.unit
def test_generate_returns_string():
    model = OllamaModel(model_name="llama3.1:8b", client=FakeClient())
    result = asyncio.run(model.generate("Olá, teste."))
    assert isinstance(result, str)
    assert 'response::llama3.1:8b::Olá, teste.' in result


@pytest.mark.unit
def test_health_check_true_when_client_lists_models():
    model = OllamaModel(model_name="llama3.1:8b", client=FakeClient())
    assert model.health_check() is True
