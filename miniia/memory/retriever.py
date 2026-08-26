from typing import List


def query_relevant_documents(collection, query_embedding, top_n: int, threshold: float) -> List[str]:
    resultados = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    memorias = []
    if resultados and resultados.get("documents") and resultados["documents"][0]:
        docs = resultados["documents"][0]
        dists = resultados.get("distances", [[]])
        dists = dists[0] if dists and len(dists) > 0 else []

        for i, doc in enumerate(docs):
            if i < len(dists):
                if dists[i] <= threshold:
                    memorias.append(doc)
            else:
                memorias.append(doc)

    return memorias
