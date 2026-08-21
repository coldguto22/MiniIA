import pytest

import asas


@pytest.mark.unit
def test_formatar_resultados_vazio():
    assert asas.formatar_resultados([]) == "Nenhum resultado encontrado."


@pytest.mark.unit
def test_formatar_resultados_com_dados():
    resultados = [
        {
            "title": "Teste",
            "url": "https://exemplo.com",
            "snippet": "Resumo curto",
        }
    ]

    texto = asas.formatar_resultados(resultados)
    assert "1. Teste" in texto
    assert "Resumo curto" in texto
    assert "https://exemplo.com" in texto


@pytest.mark.unit
def test_pesquisar_com_ddgs_mock(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def text(self, query, max_results=3):
            yield {
                "title": f"Resultado para {query}",
                "href": "https://exemplo.com/resultado",
                "body": "Snippet de teste",
            }

    monkeypatch.setattr(asas, "DDGS", FakeDDGS)

    resultados = asas.pesquisar("curiosidade dante", max_results=1)
    assert len(resultados) == 1
    assert resultados[0]["title"].startswith("Resultado para")
    assert resultados[0]["url"] == "https://exemplo.com/resultado"
