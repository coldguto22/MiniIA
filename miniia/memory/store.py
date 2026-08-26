import os
from typing import Optional

import chromadb

DEFAULT_COLLECTION_NAME = "memoria_da_ia"


def get_persist_dir(base_dir: Optional[str] = None) -> str:
    root = base_dir if base_dir is not None else os.getcwd()
    return os.path.join(root, "chroma_db")


def get_or_create_memory_collection(
    base_dir: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    persist_dir = get_persist_dir(base_dir=base_dir)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
