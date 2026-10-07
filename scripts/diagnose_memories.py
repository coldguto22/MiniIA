"""Inspeção somente leitura da coleção ChromaDB do Dante."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from scripts.common import embed, memory_collection


def _metadata_value(metadata: dict[str, Any] | None, key: str) -> str:
    return str((metadata or {}).get(key, "-"))


def _print_inventory(collection, limit: int) -> None:
    total = collection.count()
    print(f"Coleção: memoria_da_ia")
    print(f"Documentos: {total}")
    if total == 0:
        return

    all_metadata = collection.get(include=["metadatas"]).get("metadatas") or []
    all_types = Counter(_metadata_value(meta, "tipo") for meta in all_metadata)
    all_sources = Counter(_metadata_value(meta, "fonte") for meta in all_metadata)
    print(f"Tipos no banco: {dict(all_types)}")
    print(f"Fontes no banco: {dict(all_sources)}")

    data = collection.get(
        limit=min(limit, total),
        include=["documents", "metadatas", "embeddings"],
    )
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    embeddings = data.get("embeddings")
    if embeddings is None:
        embeddings = []
    dimensions = sorted({len(vector) for vector in embeddings if vector is not None})

    print(f"Dimensões dos vetores amostrados: {dimensions or ['não retornadas']}")
    print("\nAmostras:")
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        print(
            f"[{index + 1}] tipo={_metadata_value(metadata, 'tipo')} "
            f"fonte={_metadata_value(metadata, 'fonte')}\n"
            f"    {' '.join(document.split())[:240]}"
        )


def _print_query(collection, query: str, top: int, threshold: float) -> None:
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=top,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    print(f"\nConsulta: {query}")
    print(f"Limiar exibido: distância <= {threshold}")
    if not documents:
        print("Nenhuma memória retornada.")
        return
    for index, document in enumerate(documents):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) else {}
        marker = "OK" if distance is None or distance <= threshold else "FORA"
        print(
            f"[{marker}] distância={distance} "
            f"tipo={_metadata_value(metadata, 'tipo')} "
            f"fonte={_metadata_value(metadata, 'fonte')}\n"
            f"    {' '.join(document.split())[:500]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Amostras do inventário")
    parser.add_argument("--query", help="Texto para consulta semântica real")
    parser.add_argument("--top", type=int, default=5, help="Quantidade de resultados da consulta")
    parser.add_argument("--threshold", type=float, default=0.7, help="Distância máxima destacada como OK")
    args = parser.parse_args()

    collection = memory_collection()
    _print_inventory(collection, max(1, args.limit))
    if args.query:
        _print_query(collection, args.query, max(1, args.top), args.threshold)


if __name__ == "__main__":
    main()