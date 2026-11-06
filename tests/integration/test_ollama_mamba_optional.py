import asyncio
import os

import ollama
import pytest

from dante.models.mamba_model import MambaModel
from dante.models.stateful import ModelSession


@pytest.mark.integration
@pytest.mark.ollama
def test_mamba_stateful_generation_real_ollama(run_ollama_tests_enabled):
    if not run_ollama_tests_enabled:
        pytest.skip("Defina RUN_OLLAMA_TESTS=1 para executar testes com Ollama real")

    model_name = os.getenv(
        "DANTE_MAMBA_TEST_MODEL",
        "hf.co/bartowski/mamba-2.8b-hf-GGUF:Q4_K_M",
    )
    model = MambaModel(
        model_name=model_name,
        client=ollama,
        max_tokens=32,
        temperature=0.1,
    )
    session = ModelSession(model, "ollama-mamba-integration")

    try:
        first = asyncio.run(session.generate("Responda brevemente: MAMBA_OK"))
        second = asyncio.run(session.generate("Responda brevemente: ESTADO_OK"))
    except Exception as exc:
        pytest.skip(f"Modelo Mamba indisponível ou incompatível: {exc}")

    assert first.strip()
    assert second.strip()
    assert session.checkpoint is not None
    assert session.checkpoint.state_type == "ollama-context"
    assert session.checkpoint.payload
