import json
import os
import shutil
from datetime import datetime

import chromadb
import numpy as np

from scripts.common import project_root

PERSIST_DIR = os.path.join(project_root(), "chroma_db")


def main():
    old_client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        old_collection = old_client.get_collection(name="memoria_da_ia")
        all_data = old_collection.get(include=["documents", "metadatas", "embeddings"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        embeddings = all_data.get("embeddings", [])
        print(f"Encontrados {len(documents)} documentos.")
    except Exception as e:
        print(f"Erro ao acessar colecao antiga: {e}")
        documents = []
        metadatas = []
        embeddings = []

    backup = []
    for i, doc in enumerate(documents):
        emb = embeddings[i] if i < len(embeddings) else None
        if emb is not None and isinstance(emb, np.ndarray):
            emb = emb.tolist()
        backup.append(
            {
                "document": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "embedding": emb,
            }
        )

    backup_path = os.path.join(project_root(), "memorias_backup.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)
    print(f"Backup salvo em memorias_backup.json com {len(backup)} itens.")

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    print("Pasta chroma_db removida.")

    new_client = chromadb.PersistentClient(path=PERSIST_DIR)
    new_collection = new_client.get_or_create_collection(
        name="memoria_da_ia",
        metadata={"hnsw:space": "cosine"},
    )

    if backup:
        ids = []
        docs = []
        metas = []
        embs = []
        for i, item in enumerate(backup):
            doc_id = f"recovered_{datetime.now().isoformat()}_{i}"
            ids.append(doc_id)
            docs.append(item["document"])
            metas.append(item.get("metadata", {}))
            emb = item.get("embedding")
            if emb is not None:
                embs.append(emb)

        if embs and len(embs) == len(docs):
            new_collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            new_collection.add(ids=ids, documents=docs, metadatas=metas)
            print("Embeddings nao foram restaurados; sera necessario regenerar.")
        print(f"{len(docs)} documentos reinseridos.")
    else:
        print("Nenhum dado para reinserir. A colecao esta vazia.")


if __name__ == "__main__":
    main()
