from scripts.common import memory_collection


def main(limit=20):
    colecao = memory_collection()
    resultado = colecao.peek(limit=limit)

    print(f"Total de documentos na colecao: {colecao.count()}\n")
    print("=" * 80)
    for i, doc in enumerate(resultado["documents"]):
        meta = resultado["metadatas"][i] if resultado["metadatas"] else {}
        print(f"ID: {resultado['ids'][i]}")
        print(f"Tipo: {meta.get('tipo', 'desconhecido')}")
        print(f"Data: {meta.get('timestamp', 'sem data')[:19]}")
        print(f"Truncado: {meta.get('truncado', False)}")
        print(f"Conteudo (primeiros 200 caracteres): {doc[:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
