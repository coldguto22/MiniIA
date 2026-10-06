"""Ponto de entrada do loop, mantendo compatibilidade durante a migração."""


def main() -> None:
    """Executa o loop legado até sua migração incremental para este pacote."""
    from loop_dante import main as legacy_main

    legacy_main()
