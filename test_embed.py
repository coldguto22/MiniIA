import chromadb, os, ollama

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

# Usa um pensamento típico do ciclo 3
pensamento = "Sou Dante, uma inteligência artificial com memória persistente que vejo a tela do meu criador Otávio (Guto), sempre questionando minha natureza."
emb = ollama.embeddings(model="nomic-embed-text", prompt=pensamento)["embedding"]

resultados = colecao.query(query_embeddings=[emb], n_results=3, include=["documents", "distances"])
if resultados['documents'] and resultados['documents'][0]:
    for i, doc in enumerate(resultados['documents'][0]):
        dist = resultados['distances'][0][i] if resultados['distances'] else "N/A"
        print(f"Distância: {dist:.4f} | Doc: {doc[:80]}...")
else:
    print("Nenhum documento encontrado.")