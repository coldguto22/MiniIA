"""Pacote modular do Dante, sem carregar backends de modelo ao importar."""

__all__ = ["load_models_config", "build_model_for"]


def __getattr__(name: str):
    """Carrega utilitários de modelo somente quando algum chamador os solicita."""
    if name in __all__:
        from .config import build_model_for, load_models_config

        return {"build_model_for": build_model_for, "load_models_config": load_models_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
