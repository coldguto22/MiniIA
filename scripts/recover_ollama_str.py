"""Recupera pensamentos e reflexões salvos como str(response) do Ollama.

Extrai o campo `response='...'` real, reembeda e reescreve o documento.
Backup automático antes de alterar.
"""
import ast
import re
import shutil
from datetime import datetime
from pathlib import Path

import chromadb
import ollama

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma_db"
BACKUP_DIR = BASE_DIR / "backups"
EMBEDDING_MODEL = "nomic-embed-text"

# Match response='...' ou response="..." antes de , word= ou )
_RESPONSE_RE = re.compile(
    r"response=(['\"])((?:\\.|(?!\1).)*)\1(?=,\s*\w+\s*=|\s*\))",
    re.DOTALL,
)


def extract_ollama_response(texto: str) -> str | None:
    """Extrai response='...' de um str(obj) do Ollama.

    Não usa regex porque o padrão depende de contexto (vírgula, espaço,
    parêntese) que varia entre versões do Ollama. Em vez disso, localiza
    o campo response=, pega a aspa de abertura, e varre até a aspa de
    fechamento respeitando escapes.
    """
    idx = texto.find("response=")
    if idx == -1:
        return None
    i = idx + len("response=")
    if i >= len(texto):
        return None
    quote = texto[i]
    if quote not in ("'", '"'):
        return None
    i += 1
    start = i
    while i < len(texto):
        c = texto[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            raw = texto[start:i]
            try:
                # ast.literal_eval lida com \n, \', \\ e outras sequências
                return ast.literal_eval(quote + raw + quote).strip()
            except (ValueError, SyntaxError):
                return raw.strip()
        i += 1
    return None


def gerar_embedding(texto):
    import numpy as np
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=texto)
    emb = np.array(resp["embedding"])
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"chroma_db_prerecovery_{ts}"
    shutil.copytree(PERSIST_DIR, dest)
    print(f"[BACKUP] {dest}")

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    col = client.get_collection("memoria_da_ia")

    r = col.get(include=["documents", "metadatas"])
    ids, docs, metas = r["ids"], r["documents"], r["metadatas"]

    plan = []
    skipped = 0

    for mid, doc, meta in zip(ids, docs, metas):
        meta = dict(meta or {})
        tipo = meta.get("tipo", "")
        doc = doc or ""

        if tipo == "observacao_passiva":
            # Esperado: "Observação: [OCR]\nPensamento: str(obj)"
            if "Pensamento:" not in doc:
                skipped += 1
                continue
            pre, after = doc.split("Pensamento:", 1)
            obs = pre.strip()
            if obs.startswith("Observação:"):
                obs = obs[len("Observação:"):].strip()
            real = extract_ollama_response(after)
            if not real:
                skipped += 1
                continue
            plan.append({
                "id": mid, "tipo": tipo, "novo": real,
                "extras": {"observacao_bruta": obs[:2000]},
            })

        elif tipo == "reflexao":
            text = doc
            pensamento_origem = ""
            if "(Baseado em:" in text:
                text, origem = text.split("(Baseado em:", 1)
                pensamento_origem = origem.rstrip(")").strip()
            if text.strip().startswith("Reflexão:"):
                text = text.strip()[len("Reflexão:"):].strip()
            real = extract_ollama_response(text)
            if not real:
                # Não era str(obj), mantém o texto como está
                real = text.strip()
                if len(real) < 20:
                    skipped += 1
                    continue
            plan.append({
                "id": mid, "tipo": tipo, "novo": real,
                "extras": {"pensamento_origem": pensamento_origem[:500]},
            })

    print(f"\nA recuperar: {len(plan)}")
    print(f"Ignorados (não batiam com o padrão): {skipped}")
    print("\nAmostra (3 primeiros):")
    for item in plan[:3]:
        print(f"\n  [{item['tipo']}] ID: {item['id'][:60]}")
        print(f"  Novo doc: {item['novo'][:200]}...")
        for k, v in item["extras"].items():
            print(f"  {k}: {v[:100]}...")

    print("\nDigite SIM para recuperar, dry para simular:")
    resp = input("> ").strip()

    if resp.lower() == "dry":
        print("Simulação. Nada alterado.")
        return
    if resp != "SIM":
        print("Cancelado.")
        return

    for i, item in enumerate(plan, 1):
        try:
            new_emb = gerar_embedding(item["novo"])
        except Exception as e:
            print(f"  [{i}] Erro no embedding: {e}")
            continue

        existing = col.get(ids=[item["id"]], include=["metadatas"])
        old_meta = dict((existing["metadatas"] or [{}])[0] or {})
        for k, v in item["extras"].items():
            if v:
                old_meta[k] = v

        col.update(
            ids=[item["id"]],
            documents=[item["novo"]],
            embeddings=[new_emb],
            metadatas=[old_meta],
        )
        if i % 20 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)}")

    print(f"\n[DONE] Total final: {col.count()}")


if __name__ == "__main__":
    main()