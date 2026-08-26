from datetime import datetime

from scripts.common import embed, memory_collection


def main():
    colecao = memory_collection()
    timestamp = datetime.now().isoformat()
    documento = "Observacao: Teste persistencia\nPensamento: Isso e um teste"
    emb = embed(documento, normalize=False)
    colecao.add(
        documents=[documento],
        embeddings=[emb],
        metadatas=[{"timestamp": timestamp, "tipo": "observacao_passiva"}],
        ids=[timestamp],
    )
    print("Teste de persistencia registrado.")


if __name__ == "__main__":
    main()
