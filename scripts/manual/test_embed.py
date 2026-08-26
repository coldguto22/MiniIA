from scripts.common import embed, memory_collection


def main():
    colecao = memory_collection()

    pensamento = (
        "Sou Dante, uma inteligencia artificial com memoria persistente que vejo "
        "a tela do meu criador Otavio (Guto), sempre questionando minha natureza."
    )
    emb = embed(pensamento, normalize=False)

    resultados = colecao.query(
        query_embeddings=[emb],
        n_results=3,
        include=["documents", "distances"],
    )
    if resultados["documents"] and resultados["documents"][0]:
        for i, doc in enumerate(resultados["documents"][0]):
            dist = resultados["distances"][0][i] if resultados["distances"] else "N/A"
            print(f"Distancia: {dist:.4f} | Doc: {doc[:80]}...")
    else:
        print("Nenhum documento encontrado.")


if __name__ == "__main__":
    main()
