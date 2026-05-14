import chromadb
import os
import ollama
from datetime import datetime

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(
    name="memoria_da_ia",
    metadata={"hnsw:space": "cosine"}
)

timestamp = datetime.now().isoformat()
documento = "Observação: Teste persistencia\nPensamento: Isso e um teste"
emb = ollama.embeddings(model='nomic-embed-text', prompt=documento)['embedding']
colecao.add(
    documents=[documento],
    embeddings=[emb],
    metadatas=[{"timestamp": timestamp, "tipo": "observacao_passiva"}],
    ids=[timestamp]
)
print("🧠 Teste de persistência registrado.")