import pytest


@pytest.mark.integration
def test_persistencia_e_leitura_em_diretorio_temporario(temp_collection):
    documento = "Observacao: teste de persistencia isolada\nPensamento: sem tocar banco real"
    temp_collection.add(
        documents=[documento],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"tipo": "observacao_passiva"}],
        ids=["id_teste_1"],
    )

    resultado = temp_collection.peek(limit=10)
    assert "documents" in resultado
    assert len(resultado["documents"]) >= 1
    assert documento in resultado["documents"]


@pytest.mark.integration
def test_query_por_embedding_em_colecao_temporaria(temp_collection):
    documento_alvo = "Sou Dante com memoria persistente em teste"
    documento_outro = "Texto distante para consulta vetorial"

    temp_collection.add(
        documents=[documento_alvo, documento_outro],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"tipo": "alvo"}, {"tipo": "outro"}],
        ids=["doc_alvo", "doc_outro"],
    )

    resultados = temp_collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=1,
        include=["documents", "distances"],
    )

    assert resultados["documents"]
    assert resultados["documents"][0][0] == documento_alvo
    assert resultados["distances"][0][0] <= resultados["distances"][0][-1]
