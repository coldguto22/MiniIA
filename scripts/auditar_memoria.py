"""Auditoria do banco de memórias de Dante, sem apagar nada.

Usa embeddings do ChromaDB para detectar duplicatas semanticamente, em vez de
comparar textos caractere por caractere.
"""
import os
import json
from collections import Counter

import chromadb
import numpy as np

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection("memoria_da_ia")

todos = colecao.get(include=["documents", "metadatas", "embeddings"])
docs = todos.get("documents", []) or []
metas = todos.get("metadatas", []) or []
ids = todos.get("ids", []) or []
embs = todos.get("embeddings")

print(f"Total de memórias: {len(docs)}\n")

# Distribuição
tipos = Counter(m.get("tipo", "sem_tipo") for m in metas)
fontes = Counter(m.get("fonte", "sem_fonte") for m in metas)
print("Por tipo:")
for t, n in tipos.most_common():
    print(f"  {t}: {n}")
print("\nPor fonte:")
for f, n in fontes.most_common():
    print(f"  {f}: {n}")

if embs is None or len(embs) != len(docs):
    print("\nAVISO: embeddings não vieram na resposta. Algumas memórias não têm embedding.")
    print("Não é possível fazer comparação semântica completa.")
    raise SystemExit(0)

# Converte para matriz numpy e normaliza (por segurança)
mat = np.array(embs, dtype=np.float32)
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat = mat / norms

# Similaridade cosseno em bloco: mat @ mat.T é a matriz de similaridades
print("\nCalculando matriz de similaridades (isso leva segundos)...")
sim = mat @ mat.T

# Zera diagonal e parte superior para evitar contar pares duas vezes
np.fill_diagonal(sim, 0.0)

# Encontra pares com similaridade >= 0.85
threshold = 0.93
pares = []
n = len(docs)
for i in range(n):
    # Pega apenas j > i para não duplicar
    linha = sim[i, i+1:]
    candidatos = np.where(linha >= threshold)[0]
    for offset in candidatos:
        j = i + 1 + offset
        pares.append({
            "id_a": ids[i],
            "id_b": ids[j],
            "ratio": float(linha[offset]),
            "texto_a": docs[i][:120],
            "texto_b": docs[j][:120],
            "tipo_a": metas[i].get("tipo", "?"),
            "tipo_b": metas[j].get("tipo", "?"),
            "fonte_a": metas[i].get("fonte", "?"),
            "fonte_b": metas[j].get("fonte", "?"),
        })

print(f"\nEncontrados {len(pares)} pares com similaridade >= {threshold}.")
print(f"Amostra (10 primeiros):\n")
for p in pares[:10]:
    print(f"  [{p['ratio']:.3f}] {p['tipo_a']} × {p['tipo_b']}")
    print(f"        A: {p['texto_a']}")
    print(f"        B: {p['texto_b']}\n")

with open("auditoria_duplicatas.json", "w", encoding="utf-8") as f:
    json.dump(pares, f, ensure_ascii=False, indent=2)
print(f"Lista completa salva em auditoria_duplicatas.json ({len(pares)} pares)")