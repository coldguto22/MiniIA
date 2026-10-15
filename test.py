import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("memoria_da_ia")
r = col.get(where={"tipo": "observacao_passiva"}, include=["documents", "metadatas"], limit=3)
for doc, meta in zip(r["documents"], r["metadatas"]):
    print("="*80)
    print(f"ID: {meta.get('timestamp', '?')}")
    print(f"repeticoes: {meta.get('repeticoes', '?')}")
    print(f"--- DOCUMENTO COMPLETO (não truncado) ---")
    print(doc)
    print()