import ollama
import memoria
from datetime import datetime
import textwrap
import sys

MAX_CHUNK = 500


def chunk_text(text, max_chars=MAX_CHUNK):
    # Primeiro por parágrafos
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            # divide em fatias simples
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])
    if not chunks:
        # fallback: split por tamanho bruto
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
    return chunks


def get_embedding(text):
    try:
        resp = ollama.embeddings(model='nomic-embed-text', prompt=text)
        # tentar extrair embedding de formatos comuns
        if isinstance(resp, dict) and 'embedding' in resp:
            return resp['embedding']
        if isinstance(resp, list) and len(resp) > 0:
            first = resp[0]
            if isinstance(first, dict) and 'embedding' in first:
                return first['embedding']
            return first
        return resp
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None


def main():
    print("Cole o texto (termine com uma linha contendo apenas 'END'):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == 'END':
            break
        lines.append(line)

    texto = '\n'.join(lines).strip()
    if not texto:
        print("Nenhum texto recebido. Abortando.")
        sys.exit(0)

    fonte = input("Fonte (ex: conversa_sobre_ser): ").strip() or "manual"
    tipo = input("Tipo (ex: conhecimento_fundacional): ").strip() or "conhecimento_fundacional"

    chunks = chunk_text(texto)
    embeddings = []
    metadatas = []
    timestamp = datetime.now().isoformat()
    print(f"Gerando embeddings para {len(chunks)} chunk(s)...")
    for i, c in enumerate(chunks):
        emb = get_embedding(c)
        embeddings.append(emb)
        metadatas.append({"fonte": fonte, "tipo": tipo, "timestamp": timestamp, "chunk_index": i})
    # Se algum embedding falhou (None), não passemos a lista incompleta ao ChromaDB
    if any(e is None for e in embeddings):
        print("Alguns embeddings falharam; armazenando sem embeddings.")
        memoria.store_chunks(chunks, metadatas, embeddings=None)
    else:
        memoria.store_chunks(chunks, metadatas, embeddings=embeddings)
    print(f"Concluído: {len(chunks)} chunk(s) armazenado(s).")


if __name__ == '__main__':
    main()
