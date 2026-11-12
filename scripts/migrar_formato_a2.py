"""Migra memórias legadas para o formato A2 com backup e revisão prévia."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import ollama

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma_db"
BACKUP_DIR = BASE_DIR / "backups"
COLLECTION_NAME = "memoria_da_ia"
EMBEDDING_MODEL = "nomic-embed-text"


def gerar_embedding(texto: str) -> list[float]:
    import numpy as np

    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=texto)
    embedding = np.array(response["embedding"])
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    return embedding.tolist()


def parse_observacao_passiva(documento: str) -> tuple[str, str]:
    if "Pensamento:" not in documento:
        return documento.strip(), ""
    pre, after = documento.split("Pensamento:", 1)
    observacao = pre.strip()
    if observacao.startswith("Observação:"):
        observacao = observacao[len("Observação:"):].strip()
    return after.strip(), observacao


def parse_reflexao(documento: str) -> tuple[str, str]:
    pensamento_origem = ""
    reflexao = documento.strip()
    if "(Baseado em:" in reflexao:
        reflexao, origem = reflexao.split("(Baseado em:", 1)
        pensamento_origem = origem.rstrip(")").strip()
    if reflexao.strip().startswith("Reflexão:"):
        reflexao = reflexao.strip()[len("Reflexão:"):].strip()
    return reflexao, pensamento_origem


def parse_diario(documento: str) -> str:
    documento = documento.strip()
    if documento.startswith("Diário:"):
        return documento[len("Diário:"):].strip()
    return documento


def parse_pesquisa(documento: str) -> tuple[str, str]:
    if "Aprendizado:" not in documento:
        return documento.strip(), ""
    pre, after = documento.split("Aprendizado:", 1)
    query = pre.strip()
    if query.startswith("Pesquisa:"):
        query = query[len("Pesquisa:"):].strip()
    return after.strip(), query


def _build_plan_from_records(records: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    plan: list[dict[str, Any]] = []
    counts: Counter = Counter()

    for memory_id, documento, metadata in zip(
        records.get("ids", []),
        records.get("documents", []) or [],
        records.get("metadatas", []) or [],
    ):
        metadata = dict(metadata or {})
        kind = metadata.get("tipo", "")
        extras: dict[str, str] = {}

        if kind == "observacao_passiva":
            new_document, observation = parse_observacao_passiva(documento or "")
            extras["observacao_bruta"] = observation[:2000]
        elif kind == "reflexao":
            new_document, thought = parse_reflexao(documento or "")
            extras["pensamento_origem"] = thought[:500]
        elif kind == "diario":
            new_document = parse_diario(documento or "")
        elif kind == "pesquisa_autonoma":
            new_document, query = parse_pesquisa(documento or "")
            extras["query_pesquisa"] = query[:200]
        else:
            continue

        if not new_document:
            continue
        plan.append(
            {
                "id": memory_id,
                "tipo": kind,
                "metadata": metadata,
                "novo": new_document,
                "extras": extras,
            }
        )
        counts[kind] += 1

    return plan, counts


def _build_plan(collection) -> tuple[list[dict[str, Any]], Counter]:
    records = collection.get(include=["documents", "metadatas"])
    return _build_plan_from_records(records)


def _print_plan(plan: list[dict[str, Any]], counts: Counter) -> None:
    print(f"A migrar: {len(plan)}")
    for kind, count in counts.most_common():
        print(f"  {kind}: {count}")
    print("\nAmostra (3 primeiros):")
    for item in plan[:3]:
        print(f"\n  [{item['tipo']}] ID: {item['id'][:60]}")
        print(f"  Novo documento: {item['novo'][:140]}")
        for key, value in item["extras"].items():
            print(f"  {key}: {value[:100]}")


def _create_backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_DIR / f"chroma_db_premigracao_{timestamp}"
    shutil.copytree(PERSIST_DIR, destination)
    return destination


def _migrate(collection, plan: list[dict[str, Any]], *, update_existing: bool) -> None:
    for index, item in enumerate(plan, 1):
        embedding = gerar_embedding(item["novo"])
        metadata = dict(item["metadata"])
        for key, value in item["extras"].items():
            if value:
                metadata[key] = value
        operation = collection.update if update_existing else collection.add
        operation(ids=[item["id"]], documents=[item["novo"]], embeddings=[embedding], metadatas=[metadata])
        if index % 20 == 0 or index == len(plan):
            print(f"  {index}/{len(plan)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exibe a amostra sem criar backup ou alterar o banco",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        help="reconstrói uma base nova a partir de um dump JSON, sem abrir o banco legado",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "chroma_db_a2_recovered",
        help="diretório da base nova quando --source-json é usado",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirma a operação após a revisão do plano; só funciona com --source-json",
    )
    args = parser.parse_args()

    if args.source_json:
        records = json.loads(args.source_json.read_text(encoding="utf-8"))
        plan, counts = _build_plan_from_records(records)
        _print_plan(plan, counts)
        if args.dry_run:
            print("\nSimulação concluída. Nada alterado.")
            return
        print("\nDigite SIM para criar a base recuperada, qualquer outra coisa para cancelar:")
        confirmed = args.yes or input("> ").strip() == "SIM"
        if not confirmed:
            print("Cancelado.")
            return
        if args.yes:
            print("Confirmação explícita recebida via --yes.")
        if args.output_dir.exists():
            raise SystemExit(f"Diretório de saída já existe: {args.output_dir}")
        new_client = chromadb.PersistentClient(path=str(args.output_dir))
        new_collection = new_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        _migrate(new_collection, plan, update_existing=False)
        print(f"[DONE] Base recuperada em {args.output_dir}; total: {new_collection.count()}")
        return

    if not PERSIST_DIR.exists():
        raise SystemExit(f"Banco não encontrado: {PERSIST_DIR}")

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    plan, counts = _build_plan(collection)
    _print_plan(plan, counts)

    if args.dry_run:
        print("\nSimulação concluída. Nada alterado.")
        return

    print("\nDigite SIM para criar o backup e migrar, qualquer outra coisa para cancelar:")
    if input("> ").strip() != "SIM":
        print("Cancelado.")
        return

    backup = _create_backup()
    print(f"[BACKUP] {backup}")
    print("Migrando...")
    _migrate(collection, plan, update_existing=True)
    print(f"[DONE] Total final: {collection.count()}")


if __name__ == "__main__":
    main()
