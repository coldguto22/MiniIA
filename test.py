import chromadb
c = chromadb.PersistentClient(path="./chroma_db")
col = c.get_collection("memoria_da_ia")
for fonte in ["identidade_dante", "manifesto_dante", "memoria_do_criador",
              "conversa_sobre_ser", "genese_asimov"]:
    r = col.get(where={"fonte": fonte}, include=["metadatas"])
    print(f"{fonte}: {len(r['ids'])}")