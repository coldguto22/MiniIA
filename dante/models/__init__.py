"""Implementações de modelos; backends são importados sob demanda."""

__all__ = ["BaseModel", "OllamaModel"]


def __getattr__(name: str):
    if name == "BaseModel":
        from .base import BaseModel

        return BaseModel
    if name == "OllamaModel":
        from .ollama_model import OllamaModel

        return OllamaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
