import asyncio

import pytest

from dante.models.checkpoint import CheckpointError
from dante.models.mamba_model import MambaModel


class FakeClient:
    def list(self):
        return {"models": [{"name": "fake"}]}

    def generate(self, model, prompt, **kwargs):
        if model == "hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF":
            raise RuntimeError("model offline")
        return {"response": f"fallback:{model}:{prompt}"}


class StatefulFakeClient(FakeClient):
    def generate(self, model, prompt, **kwargs):
        context = kwargs.get("context", [])
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
def test_generate_with_state_round_trips_context_and_validates_session():
    model = MambaModel(model_name="test-mamba", client=StatefulFakeClient())
    first, checkpoint = asyncio.run(model.generate_with_state("primeiro", session_id="session-a"))
    second, updated = asyncio.run(
        model.generate_with_state("segundo", session_id="session-a", checkpoint=checkpoint)
    )

    assert first == "stateful:primeiro"
    assert second == "stateful:segundo"
    assert updated.payload == [8, 7]
    with pytest.raises(CheckpointError, match="session"):
        asyncio.run(model.generate_with_state("terceiro", session_id="session-b", checkpoint=checkpoint))
