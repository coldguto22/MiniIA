import chromadb
import os

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

resultado = colecao.peek(limit=10)
print(f"documentos_count: {len(resultado['documents'])}")
for doc in resultado['documents']:
    print(f"documento: '{doc}'")