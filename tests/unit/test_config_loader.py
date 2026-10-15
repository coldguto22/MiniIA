import pytest

from dante.config import load_models_config, load_runtime_config
from dante.models.mamba_model import MambaModel
from dante.models.ollama_model import OllamaModel


@pytest.mark.unit
def test_load_models_config_returns_expected_values():
    config = load_models_config()
    assert config["system1"]["provider"] == "ollama"
    assert config["system1"]["fallback"] == "qwen2.5:3b"
    assert config["system2"]["model"] == "qwen2.5:7b"


@pytest.mark.unit
def test_load_runtime_config_returns_intrinsic_state_parameters():
    config = load_runtime_config()
    assert config["silence"]["min_energy"] == 0.2
    assert config["self_model"]["min_diary_entries"] == 20


@pytest.mark.unit
def test_build_model_for_uses_provider_specific_adapter(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "system1:\n"
        "  provider: mamba\n"
        "  model: test-mamba\n"
        "  fallback: qwen2.5:3b\n"
        "system2:\n"
        "  provider: ollama\n"
        "  model: qwen2.5:7b\n",
        encoding="utf-8",
    )

    from dante.config import build_model_for

    assert isinstance(build_model_for("system1", config_path), MambaModel)
    assert isinstance(build_model_for("system2", config_path), OllamaModel)
