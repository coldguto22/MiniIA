# scripts/diag_memorias.py
"""Diagnóstico: conta por tipo e procura observações específicas."""
import chromadb
from collections import Counter

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("memoria_da_ia")

r = col.get(include=["documents", "metadatas"])
ids = r["ids"]
docs = r["documents"]
metas = r["metadatas"]

print(f"Total: {len(ids)}\n")

print("Por tipo:")
tipos = Counter(m.get("tipo", "?") for m in metas)
for t, c in tipos.most_common():
    print(f"  {t}: {c}")

# Estatísticas de repeticoes em observacoes
obs_reps = [m.get("repeticoes", 1) for m in metas if m.get("tipo") == "observacao_passiva"]
if obs_reps:
    print(f"\nObservações passivas: {len(obs_reps)}")
    print(f"  Média de repetições: {sum(obs_reps)/len(obs_reps):.1f}")
    print(f"  Máx: {max(obs_reps)}")
    print(f"  Total implícito de capturas: {sum(obs_reps)}")

# Busca textual
print("\n=== OCORRÊNCIAS: 'football' ou 'manager' ===")
found = 0
for mid, doc, meta in zip(ids, docs, metas):
    low = (doc or "").lower()
    if "football" in low or "manager" in low or "colômbia" in low or "brasil" in low:
        found += 1
        print(f"\n[{meta.get('tipo')}] {mid[:50]} | repeticoes={meta.get('repeticoes', '?')}")
        print(f"  {(doc or '')[:250]}...")
        if found >= 10:
            print(f"\n  ... (mais resultados truncados)")
            break
print(f"\nTotal encontrado: {found}")

# Amostra de observações
print("\n=== AMOSTRA: 10 observacoes_passiva ===")
count = 0
for mid, doc, meta in zip(ids, docs, metas):
    if meta.get("tipo") == "observacao_passiva":
        count += 1
        print(f"\n[{mid[:50]}] repeticoes={meta.get('repeticoes', '?')}")
        print(f"  {(doc or '')[:250]}...")
        if count >= 10:
            break