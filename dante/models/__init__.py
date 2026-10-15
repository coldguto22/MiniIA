"""Implementações de modelos; backends são importados sob demanda."""

__all__ = ["BaseModel", "MambaModel", "OllamaModel"]


def __getattr__(name: str):
    if name == "BaseModel":
        from .base import BaseModel

        return BaseModel
    if name == "OllamaModel":
        from .ollama_model import OllamaModel

        return OllamaModel
    if name == "MambaModel":
        from .mamba_model import MambaModel

        return MambaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
