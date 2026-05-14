# main.py
import capturador
import cerebro
import chromadb
import os
from datetime import datetime
import ollama

# Configuração do ChromaDB
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

def gerar_embedding(texto):
    """Gera embedding usando o modelo nomic-embed-text via Ollama."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    return response['embedding']

def buscar_memorias_relacionadas(texto, top_n=2, threshold=0.5):
    """Busca no ChromaDB memórias similares ao texto, com limiar de similaridade."""
    emb = gerar_embedding(texto)
    resultados = colecao.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )
    memorias = []
    if resultados['documents'] and resultados['documents'][0]:
        for i, doc in enumerate(resultados['documents'][0]):
            distancia = resultados['distances'][0][i] if resultados['distances'] else 1.0
            print(f"  (distância: {distancia:.4f})")  # debug
            if distancia <= threshold:
                memorias.append(doc)
    return memorias

def gerar_reflexao(pensamento_atual, memorias):
    """Gera uma reflexão conectando o pensamento atual com memórias passadas."""
    contexto = "\n---\n".join(memorias)
    prompt = f"""Você é uma IA com memória e capacidade de reflexão.

Suas memórias anteriores:
---
{contexto}
---

Observação atual:
{pensamento_atual}

Com base nas suas memórias e na observação atual, gere uma reflexão curta (1-2 frases) conectando o que você está vendo agora com o que já sabe. Não invente nada que não esteja no contexto ou na observação."""

    try:
        resposta = ollama.generate(model='qwen2.5:3b', prompt=prompt)
        return resposta['response'].strip()
    except Exception as e:
        return f"(Reflexão não gerada: {e})"

def registrar_na_memoria(documento, tipo, embedding=True):
    """Armazena um documento na memória persistente."""
    timestamp = datetime.now().isoformat()
    dados = {
        "documents": [documento],
        "metadatas": [{"timestamp": timestamp, "tipo": tipo}],
        "ids": [f"{tipo}_{timestamp}"]
    }
    if embedding:
        dados["embeddings"] = [gerar_embedding(documento)]
    colecao.add(**dados)

def main():
    print("MINI IA - CICLO DE OBSERVAÇÃO E REFLEXÃO")
    print("=" * 40)

    # 1. Capturar a tela
    print("\n👁️ Observando...")
    texto_observado = capturador.capturar_e_extrair_texto()
    if not texto_observado.strip():
        texto_observado = "Tela sem texto detectado ou majoritariamente gráfica."
    print(f"Texto observado: '{texto_observado[:100]}...'")

    # 2. Processar com a LLM (pensamento)
    print("\n🤔 Pensando...")
    pensamento = cerebro.pensar(texto_observado)
    print(f"Pensamento: {pensamento}")

    # 3. Buscar memórias relacionadas
    print("\n🔍 Buscando memórias relacionadas...")
    memorias = buscar_memorias_relacionadas(pensamento)
    if memorias:
        print(f"📚 {len(memorias)} memória(s) encontrada(s).")
        for m in memorias:
            print(f"  - {m[:80]}...")
        
        # 4. Gerar reflexão conectada
        print("\n💭 Gerando reflexão conectada...")
        reflexao = gerar_reflexao(pensamento, memorias)
        print(f"Reflexão: {reflexao}")
    else:
        print("📭 Nenhuma memória relevante encontrada.")
        reflexao = None

    # 5. Salvar na memória
    print("\n🧠 Salvando na memória...")
    documento_pensamento = f"Observação: {texto_observado}\nPensamento: {pensamento}"
    registrar_na_memoria(documento_pensamento, "observacao_passiva")
    print("Pensamento registrado.")

    if reflexao:
        documento_reflexao = f"Reflexão: {reflexao}\n(Baseado em: {pensamento})"
        registrar_na_memoria(documento_reflexao, "reflexao")
        print("Reflexão registrada.")

    print("\n" + "=" * 40)
    print("CICLO COMPLETO.")
    print(f"Total de memórias na coleção: {colecao.count()}")

if __name__ == "__main__":
    main()