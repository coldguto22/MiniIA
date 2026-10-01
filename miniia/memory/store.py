import os
from typing import Optional

try:
    import chromadb
except Exception:  # pragma: no cover - environment guard
    chromadb = None

DEFAULT_COLLECTION_NAME = "memoria_da_ia"


class _FallbackCollection:
    def __init__(self):
        self._docs = []

    def add(self, **kwargs):
        docs = kwargs.get("documents", [])
        self._docs.extend(docs)
        return None

    def query(self, **kwargs):
        docs = self._docs[:]
        query_embeddings = kwargs.get("query_embeddings") or []
        if not docs:
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
        if len(query_embeddings) == 0:
            return {"documents": [docs], "distances": [[0.0] * len(docs)], "metadatas": [[{} for _ in docs]]}
        return {"documents": [docs], "distances": [[0.0] * len(docs)], "metadatas": [[{} for _ in docs]]}


def get_persist_dir(base_dir: Optional[str] = None) -> str:
    root = base_dir if base_dir is not None else os.getcwd()
    return os.path.join(root, "chroma_db")


def get_or_create_memory_collection(
    base_dir: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    if chromadb is None:
        return _FallbackCollection()
    try:
        persist_dir = get_persist_dir(base_dir=base_dir)
        client = chromadb.PersistentClient(path=persist_dir)
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except BaseException:  # pragma: no cover - runtime guard for problematic local installs
        return _FallbackCollection()
