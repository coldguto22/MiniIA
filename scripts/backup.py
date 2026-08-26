import json
import os

from scripts.common import memory_collection


def main(output_file="backup_pre_limpeza.json"):
    colecao = memory_collection()
    todos = colecao.get(include=["documents", "metadatas", "embeddings"])

    backup = []
    for tid, doc, meta, emb in zip(
        todos["ids"],
        todos["documents"],
        todos["metadatas"],
        todos.get("embeddings", []),
    ):
        backup.append(
            {
                "id": tid,
                "doc": doc,
                "meta": meta,
                "emb": emb.tolist() if hasattr(emb, "tolist") else emb,
            }
        )

    output_path = os.path.join(os.getcwd(), output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)
    print(f"Backup salvo com {len(backup)} documentos em {output_file}.")


if __name__ == "__main__":
    main()
