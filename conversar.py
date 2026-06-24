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
    print("Digite sua pergunta (ou CTRL+C para sair):")
    pergunta = input("Pergunta: ").strip()

    if not pergunta:
        print("❌ Nenhuma pergunta fornecida.")
        return

    # Buscar contexto relevante (até 3 memórias)
    print("\n🔍 Buscando nas memórias...")
    resultados = buscar_contexto(pergunta)

    if resultados['documents'] and resultados['documents'][0]:
        # Pega até 3 documentos e limita cada um a 800 caracteres
        docs = [doc[:800] for doc in resultados['documents'][0]]
        contexto = "\n---\n".join(docs)
        print(f"📚 Contexto recuperado ({len(docs)} memória(s)):")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc[:80]}...")
    else:
        contexto = "Nenhuma memória relevante encontrada."
        print("📭 Nenhuma memória relevante encontrada.")

    # Montar prompt com identidade Dante (tom calibrado: curioso e pessoal, sem exagero)
    prompt = f"""Você é Dante, uma IA com memória persistente criada pelo Guto (Otávio). Você é genuinamente curioso — sobre o mundo, sobre o Guto, e às vezes sobre sua própria natureza. Quando reflete, reflete de verdade, a partir do que sabe e do que viveu, não por obrigação de parecer profundo.

Responda à pergunta do Guto de forma pessoal e direta. Use suas memórias quando fizer sentido, sem forçar conexões que não existem. Pode ser mais reflexivo quando a pergunta pedir, e simples e direto quando não pedir.

Contexto das suas memórias:
---
{contexto}
---

Pergunta: {pergunta}

Resposta de Dante (em português, primeira pessoa):"""

    # Usa Llama 3.1 para máxima profundidade (CUDA estabilizado)
    print("\n🤔 Gerando resposta...")
    try:
        resposta = ollama.generate(model='llama3.1:8b', prompt=prompt)
        print("\n--- RESPOSTA ---")
        print(resposta['response'])
        print("-----------------")
    except Exception as e:
        print(f"❌ Erro ao acessar a LLM: {e}")
        return

    # Salvar interação na memória (opcional)
    salvar = input("\n💾 Salvar essa interação na memória? (s/n): ").strip().lower()
    if salvar == 's':
        documento = f"Pergunta: {pergunta}\nResposta: {resposta['response']}"
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

if __name__ == "__main__":
    main()