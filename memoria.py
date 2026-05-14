# memoria.py
import chromadb
from chromadb.config import Settings
from datetime import datetime
import os
import ollama

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Criar ou acessar coleção com métrica cosine
colecao = client.get_or_create_collection(
    name="memoria_da_ia",
    metadata={"hnsw:space": "cosine"}
)

def gerar_embedding(texto):
    """Gera embedding usando o modelo nomic-embed-text via Ollama."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    return response['embedding']

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