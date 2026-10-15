import pytest

import conversar
from dante.core.homeostasis import HomeostasisState
from dante.core.relationship import RelationshipModel
from dante.core.valence import ValenceState


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


@pytest.mark.unit
def test_conversation_prompt_requires_evidence_and_disagreement():
    prompt = conversar.build_conversation_prompt(
        "Você concorda comigo?",
        "Memória 1: uma observação anterior",
        "Guto: uma fala anterior",
        RelationshipModel(recent_topics=["consciência"]),
        ValenceState(novelty=0.8, coherence=0.3),
        HomeostasisState(energy=0.7, boredom=0.9),
    )

    assert "não concorde por" in prompt
    assert "não afirme consciência" in prompt.casefold()
    assert "Memória 1" in prompt
    assert "novidade=0.80" in prompt
