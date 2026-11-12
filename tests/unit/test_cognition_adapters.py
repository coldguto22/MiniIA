import pytest

from dante.cognition.reflection import generate_reflection
from dante.cognition.thought import build_prompt as build_thought_prompt
from dante.cognition.thought import generate_thought
from dante.core.self_model import SelfModel
from legacy import cerebro as legacy_cerebro


@pytest.mark.unit
def test_thought_adapter_uses_system1_configuration_and_self_model():
    calls = []
    model = SelfModel(recurring_themes=["memória"])

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return {"response": "pensamento curto"}

    assert generate_thought("texto da tela", model, generate=fake_generate) == "pensamento curto"
    assert calls[0]["model"] == "llama3.2:latest"
    assert calls[0]["options"]["num_predict"] == 128
    assert "memória" in calls[0]["prompt"]


@pytest.mark.unit
def test_thought_prompt_separates_observation_from_unverified_claims():
    prompt = build_thought_prompt("VS Code com Python", SelfModel())

    assert "Diferencie observação, inferência e incerteza" in prompt
    assert "Não afirme consciência" in prompt
    assert "VS Code com Python" in prompt


@pytest.mark.unit
def test_legacy_cerebro_routes_through_configured_system1(monkeypatch):
    calls = []

    def fake_thought(text, self_model):
        calls.append((text, self_model))
        return "pensamento por System 1"

    monkeypatch.setattr(legacy_cerebro, "generate_thought", fake_thought)

    assert legacy_cerebro.pensar("texto OCR") == "pensamento por System 1"
    assert calls[0][0] == "texto OCR"
    assert isinstance(calls[0][1], SelfModel)


@pytest.mark.unit
def test_reflection_adapter_uses_llama_configuration():
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return {"response": "reflexão longa"}

    assert generate_reflection(
        "pensamento", ["memória relacionada"], SelfModel(), generate=fake_generate
    ) == "reflexão longa"
    assert calls[0]["model"] == "llama3.1:8b"
    assert calls[0]["options"]["num_predict"] == 2048
