from scripts.common import embed, memory_collection


def main():
    colecao = memory_collection()
    print(f"Documentos na colecao: {colecao.count()}")

    try:
        dados = colecao.get(limit=1, include=["documents"])
        print("GET simples OK:", dados)
    except Exception as e:
        print("Falha no get simples:", e)

    emb = embed("teste diagnostico", normalize=False)
    try:
        res = colecao.query(query_embeddings=[emb], n_results=2)
        print("Query OK. Resultados:", res)
    except Exception as e:
        print("Falha na query:", e)


if __name__ == "__main__":
    main()
