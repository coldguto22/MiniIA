# scripts/limpar_memoria.py
"""
Limpeza do banco de memória do Dante.

Estratégia:
  - PROTEGE: conhecimento_fundacional, conhecimento_pessoal, dialogo,
             pesquisa_autonoma, e diarios recentes
  - DELETA:  reflexao, observacao_passiva, conhecimento_funadcional (typo)

Backup duplo: diretório inteiro + dump JSON, antes de qualquer alteração.
"""

import chromadb
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter

# ============================================================
# CONFIGURAÇÃO — ajuste apenas se necessário
# ============================================================
DB_PATH = Path("./chroma_db")
COLLECTION_NAME = None          # None = auto-detecta se houver só uma coleção
BACKUP_DIR = Path("./backups")

# Tipos que NUNCA serão deletados
TIPOS_PROTEGIDOS = {
    "conhecimento_fundacional",
    "conhecimento_pessoal",
    "dialogo",
    "pesquisa_autonoma",
}

# Tipos que serão deletados sempre
TIPOS_DELETADOS = {
    "reflexao",
    "observacao_passiva",
    "conhecimento_funadcional",   # typo: duplicata das fundacionais
}

# Diários: quantos dias para trás manter?
#   None = mantém todos
#   0    = deleta todos
#   N    = mantém apenas os últimos N dias
MANTER_DIARIOS_DIAS = 120

# Campos possíveis de timestamp nos metadados
CAMPOS_TIMESTAMP = ("timestamp", "data", "criado_em", "created_at", "date")
# ============================================================


def log(msg):
    print(msg, flush=True)


def get_meta_field(meta, keys, default=None):
    for k in keys:
        if meta and k in meta and meta[k]:
            return meta[k]
    return default


def parse_ts(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        for fmt in (None, "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                if fmt is None:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def backup_diretorio(db_path: Path, backup_dir: Path) -> Path:
    if not db_path.exists():
        raise FileNotFoundError(f"Diretório ChromaDB não encontrado: {db_path}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"chroma_db_{ts}"
    shutil.copytree(db_path, dest)
    log(f"[BACKUP] Diretório copiado: {dest}")
    return dest


def get_all_paginated(collection, batch_size=500):
    total = collection.count()
    ids, docs, metas = [], [], []
    offset = 0
    while offset < total:
        r = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas", "documents"],
        )
        ids.extend(r["ids"])
        docs.extend(r.get("documents") or [None] * len(r["ids"]))
        metas.extend(r.get("metadatas") or [None] * len(r["ids"]))
        offset += batch_size
    return ids, docs, metas


def backup_json(ids, docs, metas, backup_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"memorias_dump_{ts}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(
            {"ids": ids, "documents": docs, "metadatas": metas},
            f, ensure_ascii=False, indent=2,
        )
    log(f"[BACKUP] JSON exportado: {dest} ({len(ids)} memórias)")
    return dest


def decidir(tipo: str, meta: dict, agora: datetime):
    """Retorna 'manter' ou 'deletar'."""
    if tipo in TIPOS_DELETADOS:
        return "deletar"
    if tipo in TIPOS_PROTEGIDOS:
        return "manter"
    if tipo == "diario":
        if MANTER_DIARIOS_DIAS is None:
            return "manter"
        if MANTER_DIARIOS_DIAS == 0:
            return "deletar"
        ts_raw = get_meta_field(meta, CAMPOS_TIMESTAMP)
        ts = parse_ts(ts_raw)
        if ts is None:
            return "manter"  # sem timestamp confiável -> conservador
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return "manter" if (agora - ts) <= timedelta(days=MANTER_DIARIOS_DIAS) else "deletar"
    # tipos desconhecidos -> conservador
    return "manter"


def main():
    log("=" * 64)
    log("LIMPEZA DE MEMÓRIA — DANTE")
    log("=" * 64)

    # 1. Backup do diretório
    log("\n[1/5] Backup do diretório...")
    backup_diretorio(DB_PATH, BACKUP_DIR)

    # 2. Conectar
    log("\n[2/5] Conectando ao ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    raw_collections = client.list_collections()
    names = [getattr(c, "name", c) for c in raw_collections]
    log(f"  Coleções encontradas: {names}")

    if COLLECTION_NAME:
        col_name = COLLECTION_NAME
    elif len(names) == 1:
        col_name = names[0]
    else:
        log("  ERRO: múltiplas coleções. Ajuste COLLECTION_NAME.")
        sys.exit(1)

    collection = client.get_collection(name=col_name)
    log(f"  Usando: {col_name}")
    log(f"  Total atual: {collection.count()}")

    # 3. Dump JSON
    log("\n[3/5] Carregando e exportando todas as memórias...")
    ids, docs, metas = get_all_paginated(collection)
    log(f"  Carregadas: {len(ids)}")
    backup_json(ids, docs, metas, BACKUP_DIR)

    # 4. Classificar
    log("\n[4/5] Classificando...")
    agora = datetime.now(timezone.utc)
    tipo_counter = Counter()
    plano = {"manter": [], "deletar": []}
    detalhes_deletar = []

    for mid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        tipo = meta.get("tipo", "sem_tipo")
        tipo_counter[tipo] += 1
        destino = decidir(tipo, meta, agora)
        plano[destino].append(mid)
        if destino == "deletar":
            detalhes_deletar.append({
                "id": mid,
                "tipo": tipo,
                "preview": (doc or "")[:120],
                "metadata": meta,
            })

    log("\nDistribuição por tipo:")
    for t, c in tipo_counter.most_common():
        log(f"  {t}: {c}")

    log(f"\nRESULTADO:")
    log(f"  Manter:  {len(plano['manter'])}")
    log(f"  Deletar: {len(plano['deletar'])}")

    plano_path = BACKUP_DIR / f"plano_limpeza_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(plano_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "resumo": {
                    "manter": len(plano["manter"]),
                    "deletar": len(plano["deletar"]),
                    "por_tipo": dict(tipo_counter),
                },
                "para_manter": plano["manter"],
                "para_deletar": detalhes_deletar,
            },
            f, ensure_ascii=False, indent=2,
        )
    log(f"\n  Plano completo salvo em: {plano_path}")
    log("  Revise esse arquivo antes de prosseguir, se quiser.")

    # 5. Confirmar e deletar
    log("\n[5/5] Confirmação")
    log(f"  Digite 'SIM' para deletar {len(plano['deletar'])} memórias.")
    log(f"  Digite 'dry' para simular sem alterar nada.")
    resp = input("  > ").strip()

    if resp.lower() == "dry":
        log("  Simulação. Nada foi alterado.")
        return
    if resp != "SIM":
        log("  Cancelado. Nada foi alterado.")
        return

    batch = 500
    ids_deletar = plano["deletar"]
    total = len(ids_deletar)
    for i in range(0, total, batch):
        chunk = ids_deletar[i:i + batch]
        collection.delete(ids=chunk)
        log(f"  Deletados {min(i + batch, total)}/{total}")

    log(f"\n[DONE] Total final: {collection.count()}")
    log(f"[DONE] Backup em: {BACKUP_DIR}")


if __name__ == "__main__":
    main()