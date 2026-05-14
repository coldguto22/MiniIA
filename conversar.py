# conversar.py
import chromadb
import os
from datetime import datetime
import ollama

# Configuração do ChromaDB (mesmo diretório do memoria.py)
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

def gerar_embedding(texto):
    """Gera embedding usando o modelo nomic-embed-text via Ollama."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    return response['embedding']

def buscar_contexto(pergunta, top_n=3):
    """Busca no ChromaDB os chunks mais relevantes para a pergunta."""
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

    # Buscar contexto relevante
    print("\n🔍 Buscando nas memórias...")
    resultados = buscar_contexto(pergunta)
    
    if resultados['documents'] and resultados['documents'][0]:
        contexto = "\n---\n".join(resultados['documents'][0])
        print(f"📚 Contexto recuperado ({len(resultados['documents'][0])} memórias):")
        for doc in resultados['documents'][0]:
            print(f"  - {doc[:100]}...")
    else:
        contexto = "Nenhuma memória relevante encontrada."
        print("📭 Nenhuma memória relevante encontrada.")

    # Montar prompt aumentado
    prompt = f"""Você é uma IA com acesso às suas próprias memórias e anotações.

Contexto recuperado das suas memórias:
---
{contexto}
---

Pergunta: {pergunta}

Responda com base no contexto acima. Se o contexto não for relevante, responda com seu conhecimento geral."""

    # Enviar para a LLM
    print("\n🤔 Gerando resposta...")
    try:
        resposta = ollama.generate(model='qwen2.5:3b', prompt=prompt)
        print("\n--- RESPOSTA ---")
        print(resposta['response'])
        print("-----------------")
    except Exception as e:
        print(f"❌ Erro ao acessar a LLM: {e}")
        return

    # Perguntar se quer salvar a interação na memória
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