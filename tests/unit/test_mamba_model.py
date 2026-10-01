import asyncio

import pytest

from dante.models.mamba_model import MambaModel


class FakeClient:
    def list(self):
        return {"models": [{"name": "fake"}]}

    def generate(self, model, prompt, **kwargs):
        if model == "hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF":
            raise RuntimeError("model offline")
        return {"response": f"fallback:{model}:{prompt}"}


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
