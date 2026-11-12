# scripts/limpar_memoria_v2.py
"""
Limpeza de memória com duas frentes:
  1. Deleta lixo detectado em para_limpar.json
  2. Deduplica cluster por cluster, protegendo fundacionais

Faz backup do diretório do ChromaDB antes de qualquer alteração.
"""
import os
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import chromadb

# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma_db"
BACKUP_DIR = BASE_DIR / "backups"
PARA_LIMPAR = Path("para_limpar.json")
DUPLICATAS = Path("auditoria_duplicatas.json")

TIPOS_PROTEGIDOS = {"conhecimento_fundacional", "conhecimento_pessoal"}
# ============================================================


def log(msg):
    print(msg, flush=True)


def backup_diretorio():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"chroma_db_prelimpeza_{ts}"
    shutil.copytree(PERSIST_DIR, dest)
    log(f"[BACKUP] {dest}")
    return dest


def carregar_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encontrar_clusters(pares):
    """Union-find simples para agrupar IDs conectados por duplicatas."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pares:
        for k in (p["id_a"], p["id_b"]):
            if k not in parent:
                parent[k] = k
        union(p["id_a"], p["id_b"])

    grupos = defaultdict(list)
    for k in parent:
        grupos[find(k)].append(k)
    return list(grupos.values())


def main():
    log("=" * 60)
    log("LIMPEZA DE MEMÓRIA — DANTE v2")
    log("=" * 60)

    # 1. Backup
    log("\n[1/6] Backup...")
    backup_diretorio()

    # 2. Carregar bancos
    log("\n[2/6] Carregando dados...")
    lixo = carregar_json(PARA_LIMPAR, [])
    pares = carregar_json(DUPLICATAS, [])
    log(f"  Lixo a deletar: {len(lixo)}")
    log(f"  Pares de duplicatas: {len(pares)}")

    if not lixo and not pares:
        log("Nada para fazer. Verifique se você rodou o auditar_memoria.py.")
        sys.exit(0)

    # 3. Conectar
    log("\n[3/6] Conectando ao ChromaDB...")
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    colecao = client.get_collection("memoria_da_ia")
    log(f"  Total atual: {colecao.count()}")

    # 4. Montar plano de deleção
    log("\n[4/6] Montando plano...")
    ids_deletar = set()
    motivos = defaultdict(int)

    # 4a. Lixo direto
    for item in lixo:
        ids_deletar.add(item["id"])
        motivos[item["motivo"]] += 1

    # 4b. Duplicatas — cluster por cluster
    clusters = encontrar_clusters(pares)
    log(f"  Clusters de duplicatas: {len(clusters)}")

    # Precisa pegar metadados dos IDs para saber tipo e timestamp
    todos_ids = set()
    for c in clusters:
        todos_ids.update(c)

    # Pega metadados só dos IDs envolvidos em clusters
    if todos_ids:
        info = colecao.get(
            ids=list(todos_ids),
            include=["metadatas"],
        )
        id_to_meta = {
            mid: (meta or {})
            for mid, meta in zip(info["ids"], info["metadatas"])
        }
    else:
        id_to_meta = {}

    for cluster in clusters:
        # Remove IDs que já estão marcados como lixo
        vivos = [i for i in cluster if i not in ids_deletar]
        if len(vivos) <= 1:
            continue

        # Prioridade 1: se houver fundacional, mantém esse
        fundacionais = [
            i for i in vivos
            if id_to_meta.get(i, {}).get("tipo") in TIPOS_PROTEGIDOS
        ]
        if fundacionais:
            # Mantém o primeiro fundacional (mais antigo, por ordem de get)
            a_manter = fundacionais[0]
            motivo = "duplicata_de_fundacional"
        else:
            # Mantém o mais antigo por timestamp
            def _ts(i):
                return id_to_meta.get(i, {}).get("timestamp", "")
            a_manter = min(vivos, key=_ts)
            motivo = "duplicata"

        for i in vivos:
            if i != a_manter:
                ids_deletar.add(i)
                motivos[motivo] += 1

    # 5. Report
    log("\n[5/6] Resumo do plano de deleção")
    log(f"  Total a deletar: {len(ids_deletar)}")
    for m, n in sorted(motivos.items()):
        log(f"    {m}: {n}")

    total_atual = colecao.count()
    log(f"  Banco atual: {total_atual}")
    log(f"  Banco após limpeza: {total_atual - len(ids_deletar)}")

    # Grava plano para auditoria
    plano_path = BACKUP_DIR / f"plano_delecao_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(plano_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_deletar": len(ids_deletar),
            "motivos": dict(motivos),
            "ids": sorted(ids_deletar),
        }, f, ensure_ascii=False, indent=2)
    log(f"  Plano salvo: {plano_path}")

    # 6. Confirmar
    log("\n[6/6] Confirmação")
    log("  Digite 'SIM' para deletar.")
    log("  Digite 'dry' para simular sem alterar.")
    resp = input("  > ").strip()

    if resp.lower() == "dry":
        log("  Simulação. Nada alterado.")
        return
    if resp != "SIM":
        log("  Cancelado.")
        return

    # Deleta em lotes
    ids_lista = list(ids_deletar)
    batch = 500
    for i in range(0, len(ids_lista), batch):
        chunk = ids_lista[i:i + batch]
        colecao.delete(ids=chunk)
        log(f"  Deletados {min(i + batch, len(ids_lista))}/{len(ids_lista)}")

    log(f"\n[DONE] Total final: {colecao.count()}")


if __name__ == "__main__":
    main()