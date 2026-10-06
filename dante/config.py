from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is added in requirements
    yaml = None

from .models.ollama_model import OllamaModel

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "system1": {
        "provider": "ollama",
        "model": "qwen2.5:3b",
        "fallback": None,
        "max_tokens": 512,
        "temperature": 0.7,
    },
    "system2": {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "max_tokens": 2048,
        "temperature": 0.3,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_models_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config for the model layers.

    If the config is unavailable or the YAML dependency is missing, returns a
    sensible default set that still keeps the project runnable in tests.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    config_path = Path(config_path)

    if not config_path.exists() or yaml is None:
        return deepcopy(DEFAULT_MODEL_CONFIG)

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        return deepcopy(DEFAULT_MODEL_CONFIG)

    return _deep_merge(DEFAULT_MODEL_CONFIG, loaded)


def build_model_for(role: str, config_path: str | Path | None = None):
    config = load_models_config(config_path=config_path)
    if role not in ("system1", "system2"):
        raise ValueError(f"Papel de modelo desconhecido: {role}")
    role_config = config[role]
    model_name = role_config.get("model")
    fallback = role_config.get("fallback")
    options = {key: value for key, value in role_config.items() if key not in {"model", "fallback"}}
    return OllamaModel(model_name=model_name, fallback_model=fallback, **options)
