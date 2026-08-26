# memoria.py
from datetime import datetime
import os

from miniia.memory.embedder import generate_embedding
from miniia.memory.store import get_or_create_memory_collection

BASE_DIR = os.path.dirname(__file__)
colecao = get_or_create_memory_collection(base_dir=BASE_DIR)

def gerar_embedding(texto):
    """Gera embedding usando o modelo nomic-embed-text via Ollama."""
    return generate_embedding(texto=texto, normalize=False)

def registrar_lembranca(observacao, pensamento):
    """Armazena uma nova memória com embedding."""
    timestamp = datetime.now().isoformat()
    documento = f"Observação: {observacao}\nPensamento: {pensamento}"
    emb = gerar_embedding(documento)
    colecao.add(
        documents=[documento],
        embeddings=[emb],
        metadatas=[{"timestamp": timestamp, "tipo": "observacao_passiva"}],
        ids=[timestamp]
    )
    print(f"🧠 Lembrança registrada: {pensamento[:80]}...")

def recordar(quantidade=10):
    resultado = colecao.peek(limit=quantidade)
    return resultado

if __name__ == "__main__":
    registrar_lembranca("Tela inicial do Windows", "Ambiente de trabalho padrão.")
    print("--- ÚLTIMAS LEMBRANÇAS ---")
    lembrancas = recordar()
    for doc in lembrancas['documents']:
        print(doc)
        print("-------------------------------")