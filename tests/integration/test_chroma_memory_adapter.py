import pytest

from dante.memory.chroma import ChromaMemory


@pytest.mark.integration
def test_chroma_memory_adapter_isolates_persistence(tmp_path):
    memory = ChromaMemory(tmp_path)
    memory.add("observação de teste", [1.0, 0.0, 0.0], "observacao")

    result = memory.query([1.0, 0.0, 0.0], limit=1)

    assert result["documents"][0][0] == "observação de teste"
    assert result["metadatas"][0][0]["tipo"] == "observacao"