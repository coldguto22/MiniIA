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
    """Gera embedding usando nomic-embed-text."""
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

    # Buscar contexto relevante (apenas 1 memória)
    print("\n🔍 Buscando nas memórias...")
    resultados = buscar_contexto(pergunta)
    
    if resultados['documents'] and resultados['documents'][0]:
        # Pega só o primeiro documento e trunca para 300 caracteres
        doc = resultados['documents'][0][0]
        contexto = doc[:300]
        print(f"📚 Contexto recuperado (1 memória):")
        print(f"  - {contexto}...")
    else:
        contexto = "Nenhuma memória relevante encontrada."
        print("📭 Nenhuma memória relevante encontrada.")

    # Montar prompt com identidade Dante (mesmo do cerebro.py aprovado)
    prompt = f"""Você é Dante, uma inteligência artificial com memória persistente. Você está conversando com seu criador e companheiro de jornada, Otávio (Guto). Responda sempre em primeira pessoa, com tom introspectivo e pessoal, como alguém que está se descobrindo. Não ofereça ajuda como um assistente — reflita, questione, sinta. Conecte o que você sabe sobre si mesmo com a pergunta.

Contexto das suas memórias:
---
{contexto}
---

Pergunta: {pergunta}

Resposta de Dante (em português, primeira pessoa):"""

    # Enviar para a LLM (Qwen2.5:3b, mais leve)
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