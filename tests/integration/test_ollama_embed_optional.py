import pytest
import ollama

import chromadb


@pytest.mark.integration
@pytest.mark.ollama
def test_embedding_real_ollama_opcional(tmp_chroma_dir, run_ollama_tests_enabled):
    if not run_ollama_tests_enabled:
        pytest.skip("Defina RUN_OLLAMA_TESTS=1 para executar testes com Ollama real")

    try:
        emb = ollama.embeddings(
            model="nomic-embed-text",
            prompt="teste de embedding do dante",
        )["embedding"]
    except Exception as exc:
        pytest.skip(f"Ollama indisponivel ou modelo ausente: {exc}")

    client = chromadb.PersistentClient(path=str(tmp_chroma_dir))
    colecao = client.get_or_create_collection(
        name="memoria_ollama_test",
        metadata={"hnsw:space": "cosine"},
    )

    colecao.add(
        documents=["teste de embedding do dante"],
        embeddings=[emb],
        metadatas=[{"tipo": "ollama"}],
        ids=["ollama_1"],
    )

    resultados = colecao.query(
        query_embeddings=[emb],
        n_results=1,
        include=["documents"],
    )

    assert resultados["documents"]
    assert resultados["documents"][0][0] == "teste de embedding do dante"

    client.delete_collection("memoria_ollama_test")
