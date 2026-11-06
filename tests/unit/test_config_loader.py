import pytest

from dante.config import build_model_for, load_models_config
from dante.models.ollama_model import OllamaModel


@pytest.mark.unit
def test_load_models_config_returns_expected_values():
    config = load_models_config()
    assert config["system1"]["provider"] == "ollama"
    assert config["system1"]["fallback"] == "qwen2.5:3b"
    assert config["system2"]["model"] == "llama3.1:8b"


@pytest.mark.unit
def test_build_model_for_honors_provider_and_does_not_duplicate_options(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "system1:\n"
        "  provider: mamba\n"
        "  model: test-mamba\n"
        "  fallback: qwen\n"
        "system2:\n"
        "  provider: ollama\n"
        "  model: test-llama\n",
        encoding="utf-8",
    )

    from dante.models.mamba_model import MambaModel

    assert isinstance(build_model_for("system1", config_path), MambaModel)
    assert isinstance(build_model_for("system2", config_path), OllamaModel)
