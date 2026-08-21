import os
import uuid

import chromadb
import pytest


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    return tmp_path / "chroma_db_test"


@pytest.fixture
def temp_collection(tmp_chroma_dir):
    client = chromadb.PersistentClient(path=str(tmp_chroma_dir))
    collection_name = f"memoria_test_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    yield collection
    client.delete_collection(collection_name)


@pytest.fixture
def fake_embedding():
    def _embed(texto):
        tamanho = max(1, len(texto))
        return [float(tamanho % 7), 0.25, 0.75]

    return _embed


@pytest.fixture
def run_ollama_tests_enabled():
    return os.getenv("RUN_OLLAMA_TESTS", "0") == "1"
