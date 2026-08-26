from scripts.common import memory_collection


def main():
    colecao = memory_collection()
    resultado = colecao.peek(limit=10)
    print(f"documentos_count: {len(resultado['documents'])}")
    for doc in resultado["documents"]:
        print(f"documento: '{doc}'")


if __name__ == "__main__":
    main()
