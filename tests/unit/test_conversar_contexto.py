import pytest

import conversar


@pytest.mark.unit
def test_buscar_contexto_filtra_por_threshold(monkeypatch):
    class FakeColecao:
        def query(self, **kwargs):
            return {
                "documents": [["doc_relevante", "doc_distante"]],
                "distances": [[0.2, 0.95]],
                "metadatas": [[{}, {}]],
            }

    monkeypatch.setattr(conversar, "colecao", FakeColecao())
    monkeypatch.setattr(conversar, "gerar_embedding", lambda _: [0.1, 0.2, 0.3])

    memorias = conversar.buscar_contexto("pergunta", top_n=2, threshold=0.7)
    assert memorias == ["doc_relevante"]


@pytest.mark.unit
def test_buscar_contexto_sem_distancias_retorna_docs(monkeypatch):
    class FakeColecao:
        def query(self, **kwargs):
            return {
                "documents": [["doc_a", "doc_b"]],
                "metadatas": [[{}, {}]],
            }

    monkeypatch.setattr(conversar, "colecao", FakeColecao())
    monkeypatch.setattr(conversar, "gerar_embedding", lambda _: [0.3, 0.2, 0.1])

    memorias = conversar.buscar_contexto("pergunta", top_n=2, threshold=0.7)
    assert memorias == ["doc_a", "doc_b"]
