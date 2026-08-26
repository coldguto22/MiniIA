from datetime import datetime

from scripts.common import embed, memory_collection

colecao = memory_collection()


def dividir_texto(texto, tamanho_max=500):
    """Divide o texto em chunks menores, tentando quebrar em paragrafos ou frases."""
    paragrafos = texto.split("\n")
    chunks = []
    atual = ""
    for p in paragrafos:
        if len(atual) + len(p) < tamanho_max:
            atual += p + "\n"
        else:
            if atual.strip():
                chunks.append(atual.strip())
            atual = p + "\n"
    if atual.strip():
        chunks.append(atual.strip())

    final_chunks = []
    for c in chunks:
        while len(c) > tamanho_max:
            ponto = c.rfind(" ", 0, tamanho_max)
            if ponto == -1:
                ponto = tamanho_max
            final_chunks.append(c[:ponto].strip())
            c = c[ponto:].strip()
        if c:
            final_chunks.append(c)
    return final_chunks


def main():
    print("Cole o texto (termine com uma linha contendo apenas 'END'):")
    linhas = []
    while True:
        linha = input()
        if linha.strip() == "END":
            break
        linhas.append(linha)
    texto_completo = "\n".join(linhas).strip()

    if not texto_completo:
        print("Nenhum texto fornecido.")
        return

    fonte = input("Fonte (ex: conversa_sobre_ser): ").strip()
    tipo = input("Tipo (ex: conhecimento_fundacional): ").strip()

    chunks = dividir_texto(texto_completo)
    print(f"Gerando embeddings para {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks):
        emb = embed(chunk)
        timestamp = datetime.now().isoformat()
        chunk_id = f"{timestamp}_{i}"
        colecao.add(
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{
                "fonte": fonte,
                "tipo": tipo,
                "timestamp": timestamp,
                "chunk_index": i,
            }],
            ids=[chunk_id],
        )

    print(f"{len(chunks)} chunk(s) armazenado(s) na colecao 'memoria_da_ia'.")


if __name__ == "__main__":
    main()
