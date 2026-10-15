import asyncio

import pytest

from dante.models.mamba_model import MambaModel
from dante.models.checkpoint import CheckpointError


class FakeClient:
    def list(self):
        return {"models": [{"name": "fake"}]}

    def generate(self, model, prompt, **kwargs):
        if model == "hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF":
            raise RuntimeError("model offline")
        return {"response": f"fallback:{model}:{prompt}"}


class StatefulFakeClient(FakeClient):
    def __init__(self):
        self.contexts = []

    def generate(self, model, prompt, **kwargs):
        context = kwargs.get("context", [])
        self.contexts.append(context)
        return {"response": f"stateful:{prompt}", "context": [*context, len(prompt)]}


@pytest.mark.unit
def test_generate_returns_string_for_fallback():
    model = MambaModel(
        model_name="hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF",
        fallback_model="qwen2.5:3b",
        client=FakeClient(),
    )
    result = asyncio.run(model.generate("Olá, teste."))
    assert isinstance(result, str)
    assert 'fallback:qwen2.5:3b:Olá, teste.' in result


@pytest.mark.unit
def test_health_check():
    model = MambaModel(
        model_name="hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF",
        fallback_model="qwen2.5:3b",
        client=FakeClient(),
    )
    assert model.health_check() is True


@pytest.mark.unit
def test_generate_with_state_round_trips_ollama_context():
    client = StatefulFakeClient()
    model = MambaModel(model_name="test-mamba", client=client)

    first_response, checkpoint = asyncio.run(
        model.generate_with_state("primeiro", session_id="session-a")
    )
    second_response, next_checkpoint = asyncio.run(
        model.generate_with_state("segundo", session_id="session-a", checkpoint=checkpoint)
    )

    assert first_response == "stateful:primeiro"
    assert second_response == "stateful:segundo"
    assert checkpoint.state_type == "ollama-context"
    assert client.contexts == [[], [8]]
    assert next_checkpoint.payload == [8, 7]


@pytest.mark.unit
def test_generate_with_state_rejects_checkpoint_from_other_session():
    client = StatefulFakeClient()
    model = MambaModel(model_name="test-mamba", client=client)
    _, checkpoint = asyncio.run(model.generate_with_state("primeiro", session_id="session-a"))

    with pytest.raises(CheckpointError, match="session"):
        asyncio.run(model.generate_with_state("segundo", session_id="session-b", checkpoint=checkpoint))
