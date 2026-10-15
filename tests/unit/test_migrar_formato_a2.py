import pytest

from scripts.migrar_formato_a2 import (
    parse_diario,
    parse_observacao_passiva,
    parse_pesquisa,
    parse_reflexao,
)


@pytest.mark.unit
def test_parse_observacao_passiva_separates_substance_and_ocr():
    thought, observation = parse_observacao_passiva(
        "Observação: tela com texto\nPensamento: percebi uma mudança"
    )

    assert thought == "percebi uma mudança"
    assert observation == "tela com texto"


@pytest.mark.unit
def test_parse_reflexao_separates_origin():
    reflection, thought = parse_reflexao(
        "Reflexão: isso me parece familiar\n(Baseado em: pensei sobre memória)"
    )

    assert reflection == "isso me parece familiar"
    assert thought == "pensei sobre memória"


@pytest.mark.unit
def test_parse_diario_removes_legacy_prefix():
    assert parse_diario("Diário: Hoje observei em silêncio.") == "Hoje observei em silêncio."


@pytest.mark.unit
def test_parse_pesquisa_separates_query_and_learning():
    learning, query = parse_pesquisa("Pesquisa: Mamba SSM\nAprendizado: estado recorrente")

    assert learning == "estado recorrente"
    assert query == "Mamba SSM"
