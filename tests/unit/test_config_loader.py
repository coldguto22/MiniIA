import pytest

from dante.config import load_models_config, load_runtime_config


@pytest.mark.unit
def test_load_models_config_returns_expected_values():
    config = load_models_config()
    assert config["system1"]["provider"] == "ollama"
    assert config["system1"]["fallback"] == "qwen2.5:3b"
    assert config["system2"]["model"] == "llama3.1:8b"


@pytest.mark.unit
def test_load_runtime_config_returns_intrinsic_state_parameters():
    config = load_runtime_config()
    assert config["silence"]["min_energy"] == 0.2
    assert config["self_model"]["min_diary_entries"] == 20
