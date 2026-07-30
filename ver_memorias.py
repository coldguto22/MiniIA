# ver_memorias.py
import chromadb
import os

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

# Recupera os últimos 20 documentos (ou o que desejar)
resultado = colecao.peek(limit=20)

print(f"Total de documentos na coleção: {colecao.count()}\n")
print("=" * 80)
for i, doc in enumerate(resultado['documents']):
    meta = resultado['metadatas'][i] if resultado['metadatas'] else {}
    print(f"ID: {resultado['ids'][i]}")
    print(f"Tipo: {meta.get('tipo', 'desconhecido')}")
    print(f"Data: {meta.get('timestamp', 'sem data')[:19]}")
    print(f"Truncado: {meta.get('truncado', False)}")
    print(f"Conteúdo (primeiros 200 caracteres): {doc[:200]}...")
    print("-" * 80)