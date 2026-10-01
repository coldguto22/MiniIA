import pytest

from dante.config import load_models_config


@pytest.mark.unit
def test_load_models_config_returns_expected_values():
    config = load_models_config()
    assert config["system1"]["provider"] == "ollama"
    assert config["system1"]["fallback"] == "qwen2.5:3b"
    assert config["system2"]["model"] == "llama3.1:8b"
