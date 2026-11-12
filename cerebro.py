"""Compatibilidade para o gerador de pensamento legado."""

from legacy.cerebro import pensar

__all__ = ["pensar"]


if __name__ == "__main__":
    print(pensar("Teste de compatibilidade do módulo legado."))