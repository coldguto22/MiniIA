from scripts.common import memory_collection


def main(limit=20):
    colecao = memory_collection()
    resultado = colecao.get(include=["documents", "metadatas"])
    registros = list(
        zip(
            resultado.get("ids") or [],
            resultado.get("documents") or [],
            resultado.get("metadatas") or [],
        )
    )
    registros.sort(key=lambda registro: (registro[2] or {}).get("timestamp", ""), reverse=True)

    print(f"Total de documentos na colecao: {colecao.count()}\n")
    print("=" * 80)
    for memoria_id, doc, meta in registros[:limit]:
        meta = meta or {}
        print(f"ID: {memoria_id}")
        print(f"Tipo: {meta.get('tipo', 'desconhecido')}")
        print(f"Data: {meta.get('timestamp', 'sem data')[:19]}")
        print(f"Truncado: {meta.get('truncado', False)}")
        print(f"Conteudo (primeiros 200 caracteres): {doc[:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
