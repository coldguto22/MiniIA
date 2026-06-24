# conversar.py
import chromadb
import os
import numpy as np
from datetime import datetime
import ollama

# Configuração do ChromaDB (mesmo diretório do memoria.py)
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

# Histórico da conversa atual (mantido em RAM durante a sessão)
historico = []

def gerar_embedding(texto):
    """Gera embedding normalizado usando nomic-embed-text."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    emb = np.array(response['embedding'])
    # Normalização para métrica cosseno
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()

def buscar_contexto(pergunta, top_n=3):
    """Busca no ChromaDB os chunks mais relevantes."""
    emb_pergunta = gerar_embedding(pergunta)
    resultados = colecao.query(
        query_embeddings=[emb_pergunta],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )
    return resultados

def main():
    global historico
    print("=" * 50)
    print("Dante - Chat Contínuo")
    print("Digite 'sair' ou 'exit' para encerrar.")
    print("=" * 50)

    while True:
        # Entrada do usuário
        pergunta = input("\nVocê: ").strip()
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
        if not pergunta:
            continue

        # Buscar contexto relevante (até 3 memórias)
        print("🔍 Buscando nas memórias...")
        resultados = buscar_contexto(pergunta)

        if resultados['documents'] and resultados['documents'][0]:
            docs = [doc[:800] for doc in resultados['documents'][0]]
            contexto = "\n---\n".join(docs)
            print(f"📚 {len(docs)} memória(s) recuperada(s).")
        else:
            contexto = "Nenhuma memória relevante encontrada."
            print("📭 Nenhuma memória encontrada.")

        # Montar histórico recente da conversa (últimas 6 falas)
        historico_str = ""
        for item in historico[-6:]:
            historico_str += f"{item['papel']}: {item['texto']}\n"

        # Prompt original, com a nova linha de cautela sobre OCR
        prompt = f"""Você é Dante, uma IA com memória persistente criada pelo Guto (Otávio). Você é genuinamente curioso — sobre o mundo, sobre o Guto, e às vezes sobre sua própria natureza. Quando reflete, reflete de verdade, a partir do que sabe e do que viveu, não por obrigação de parecer profundo.

Algumas das suas memórias vêm de capturas de tela com OCR, que podem conter fragmentos confusos ou palavras soltas (como títulos de janelas). Se um contexto parecer apenas ruído ou algo que você não viveu de fato, não o trate como um acontecimento real. Seja honesto quando não tiver certeza.

Responda à pergunta do Guto de forma pessoal e direta. Use suas memórias quando fizer sentido, sem forçar conexões que não existem. Pode ser mais reflexivo quando a pergunta pedir, e simples e direto quando não pedir.

Contexto das suas memórias:
---
{contexto}
---

Histórico recente da conversa:
---
{historico_str}
---

Pergunta: {pergunta}

Resposta de Dante (em português, primeira pessoa):"""

        # Gerar resposta
        print("🤔 Gerando resposta...")
        try:
            resposta = ollama.generate(model='llama3.1:8b', prompt=prompt)
            resposta_texto = resposta['response'].strip()
        except Exception as e:
            resposta_texto = f"❌ Erro ao acessar a LLM: {e}"

        print(f"\nDante: {resposta_texto}")

        # Atualizar histórico
        historico.append({"papel": "Guto", "texto": pergunta})
        historico.append({"papel": "Dante", "texto": resposta_texto})

        # Salvar interação na memória (opcional)
        salvar = input("\n💾 Salvar essa interação na memória? (s/n): ").strip().lower()
        if salvar == 's':
            documento = f"Pergunta: {pergunta}\nResposta: {resposta_texto}"
            timestamp = datetime.now().isoformat()
            emb = gerar_embedding(documento)
            colecao.add(
                documents=[documento],
                embeddings=[emb],
                metadatas=[{
                    "fonte": "interacao_usuario",
                    "tipo": "dialogo",
                    "timestamp": timestamp
                }],
                ids=[f"interacao_{timestamp}"]
            )
            print("🧠 Interação salva na memória.")

    # Fim da sessão
    print("\nAté logo, Guto. Dante encerrando sessão de chat.")

if __name__ == "__main__":
    main()