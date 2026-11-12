"""Adaptador pequeno para a memória persistente ChromaDB."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb


class ChromaMemory:
    """Isola criação, gravação e consulta da coleção de Dante."""

    def __init__(self, path: str | Path, collection_name: str = "memoria_da_ia") -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, document: str, embedding: list[float], kind: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[{"tipo": kind, "timestamp": timestamp}],
            ids=[f"{kind}_{timestamp}"],
        )

    def query(self, embedding: list[float], *, limit: int = 5) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )