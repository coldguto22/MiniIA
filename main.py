"""Compatibilidade para o ciclo único anterior à refatoração."""

from legacy.main import main

__all__ = ["main"]


if __name__ == "__main__":
    main()