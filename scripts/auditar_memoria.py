"""Auditoria do banco de memórias de Dante, sem apagar nada.

Duas análises:
  1. Pares de duplicatas semânticas (similaridade cosseno >= threshold)
  2. "Lixo" — recusas de assistente, diálogos curtos, diários vazios

Ambos os relatórios são salvos em JSON. A limpeza é feita por script separado.
"""
import os
import json
from collections import Counter

import chromadb
import numpy as np

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection("memoria_da_ia")

# --- Configuração ---
SIMILARITY_THRESHOLD = 0.93
MIN_CHARS_RESPOSTA = 80
MIN_CHARS_DIARIO = 60
MIN_CHARS_OBSERVACAO = 80

PADROES_RECUSA = (
    "não posso cumprir",
    "não posso atender",
    "não posso ajudar com isso",
    "não posso fornecer",
    "peço desculpas, mas não posso",
    "desculpe, mas não posso",
    "lamento, mas não posso",
    "como uma ia, não tenho",
    "como uma inteligência artificial, não tenho",
    "posso ajudar com outra coisa",
    "posso ajudá-lo em outra coisa",
)

# --- Carregar banco ---
todos = colecao.get(include=["documents", "metadatas", "embeddings"])
docs = todos.get("documents", []) or []
metas = todos.get("metadatas", []) or []
ids = todos.get("ids", []) or []
embs = todos.get("embeddings")

print(f"Total de memórias: {len(docs)}\n")

tipos = Counter(m.get("tipo", "sem_tipo") for m in metas)
fontes = Counter(m.get("fonte", "sem_fonte") for m in metas)
print("Por tipo:")
for t, n in tipos.most_common():
    print(f"  {t}: {n}")
print("\nPor fonte:")
for f, n in fontes.most_common():
    print(f"  {f}: {n}")


# --- Análise 1: Lixo ---
def _parece_recusa(texto: str) -> bool:
    t = (texto or "").casefold()
    return any(p in t for p in PADROES_RECUSA)


def _eh_lixo(doc: str, meta: dict) -> tuple[bool, str]:
    """Detecta memórias que não deveriam estar no banco."""
    doc = doc or ""
    tipo = meta.get("tipo", "")

    # Recusa de assistente — independente do tipo
    if _parece_recusa(doc):
        return True, "recusa_de_assistente"

    if tipo == "diario":
        # Remove prefixo "Diário: " se houver
        conteudo = doc.replace("Diário:", "").strip()
        if len(conteudo) < MIN_CHARS_DIARIO:
            return True, "diario_curto"

    if tipo == "dialogo":
        # Extrai a resposta do formato "Pergunta: ...\nResposta: ..."
        if "Resposta:" in doc:
            resposta = doc.split("Resposta:", 1)[1].strip()
            if len(resposta) < MIN_CHARS_RESPOSTA:
                return True, "dialogo_resposta_curta"
        else:
            # Sem o formato esperado — pode ser lixo
            if len(doc) < MIN_CHARS_RESPOSTA:
                return True, "dialogo_malformado"

    if tipo == "observacao_passiva" and len(doc) < MIN_CHARS_OBSERVACAO:
        return True, "observacao_curta"

    return False, ""


print("\n\n=== ANÁLISE DE LIXO ===")
lixo = []
motivos = Counter()
for mid, doc, meta in zip(ids, docs, metas):
    meta = meta or {}
    eh, motivo = _eh_lixo(doc, meta)
    if eh:
        motivos[motivo] += 1
        lixo.append({
            "id": mid,
            "motivo": motivo,
            "tipo": meta.get("tipo", "?"),
            "fonte": meta.get("fonte", "?"),
            "preview": (doc or "")[:150],
        })

print(f"Total de lixo detectado: {len(lixo)}")
for motivo, n in motivos.most_common():
    print(f"  {motivo}: {n}")

print("\nAmostra (5 primeiros):")
for item in lixo[:5]:
    print(f"  [{item['motivo']}] {item['tipo']}/{item['fonte']}")
    print(f"    {item['preview']}\n")

with open("para_limpar.json", "w", encoding="utf-8") as f:
    json.dump(lixo, f, ensure_ascii=False, indent=2)
print(f"Lista de lixo salva em para_limpar.json ({len(lixo)} itens)")


# --- Análise 2: Duplicatas semânticas ---
if embs is None or len(embs) != len(docs):
    print("\nAVISO: embeddings não vieram na resposta. Pulando análise de duplicatas.")
    raise SystemExit(0)

mat = np.array(embs, dtype=np.float32)
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat = mat / norms

print(f"\n\n=== ANÁLISE DE DUPLICATAS (threshold {SIMILARITY_THRESHOLD}) ===")
print("Calculando matriz de similaridades...")
sim = mat @ mat.T
np.fill_diagonal(sim, 0.0)

pares = []
n = len(docs)
for i in range(n):
    linha = sim[i, i + 1:]
    candidatos = np.where(linha >= SIMILARITY_THRESHOLD)[0]
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

print(f"Encontrados {len(pares)} pares com similaridade >= {SIMILARITY_THRESHOLD}.")
print(f"Amostra (5 primeiros):\n")
for p in pares[:5]:
    print(f"  [{p['ratio']:.3f}] {p['tipo_a']} × {p['tipo_b']}")
    print(f"        A: {p['texto_a']}")
    print(f"        B: {p['texto_b']}\n")

with open("auditoria_duplicatas.json", "w", encoding="utf-8") as f:
    json.dump(pares, f, ensure_ascii=False, indent=2)
print(f"Lista completa salva em auditoria_duplicatas.json ({len(pares)} pares)")


# --- Resumo ---
print("\n\n=== RESUMO ===")
print(f"Total de memórias: {len(docs)}")
print(f"Lixo detectado: {len(lixo)}")
print(f"Pares duplicados: {len(pares)}")
print(f"\nPróximos passos:")
print(f"  1. Revisar para_limpar.json")
print(f"  2. Rodar script de limpeza (a criar) para deletar IDs de para_limpar.json")
print(f"  3. Revisar auditoria_duplicatas.json e deletar duplicatas residuais")