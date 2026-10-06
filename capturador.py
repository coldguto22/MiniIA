"""Compatibilidade para o capturador anterior à refatoração."""

from legacy.capturador import capturar_e_extrair_texto

__all__ = ["capturar_e_extrair_texto"]


if __name__ == "__main__":
    print(capturar_e_extrair_texto())