import pytest

from dante.cognition.reflection import generate_reflection
from dante.cognition.thought import generate_thought
from dante.core.self_model import SelfModel


@pytest.mark.unit
def test_thought_adapter_uses_qwen_configuration_and_self_model():
    calls = []
    model = SelfModel(recurring_themes=["memória"])

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return {"response": "pensamento curto"}

    assert generate_thought("texto da tela", model, generate=fake_generate) == "pensamento curto"
    assert calls[0]["model"] == "qwen2.5:3b"
    assert "memória" in calls[0]["prompt"]


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