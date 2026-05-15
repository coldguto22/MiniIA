# conversar.py
"""
Chat contínuo com Dante — memória persistente, histórico de sessão e auto-save.
"""

import chromadb
import os
import numpy as np
from datetime import datetime
import ollama

# --- Configuração do ChromaDB ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(name="memoria_da_ia")

# --- Histórico da sessão atual (mantido em memória RAM) ---
historico_sessao = []  # lista de tuplas (pergunta, resposta)


# ── Funções auxiliares ──────────────────────────────────────────────────────

def gerar_embedding(texto):
    """Gera embedding normalizado usando nomic-embed-text."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    emb = np.array(response['embedding'])
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()


def buscar_contexto_diverso(pergunta, top_n=3):
    """
    Busca memórias relevantes priorizando diversidade de tipo,
    evitando a câmara de eco de recuperar sempre memórias similares.
    """
    emb = gerar_embedding(pergunta)
    tipos = ["observacao_passiva", "reflexao", "diario", "dialogo", "conhecimento_fundacional"]
    docs_coletados = []

    for tipo in tipos:
        try:
            resultado = colecao.query(
                query_embeddings=[emb],
                n_results=1,
                where={"tipo": tipo},
                include=["documents", "distances"]
            )
            if resultado['documents'] and resultado['documents'][0]:
                dist = resultado['distances'][0][0]
                if dist < 0.5:
                    docs_coletados.append(resultado['documents'][0][0])
        except Exception:
            pass  # tipo inexistente na coleção, ignora

        if len(docs_coletados) >= top_n:
            break

    # Fallback: se não encontrou nada com filtro, faz busca geral
    if not docs_coletados:
        resultado = colecao.query(
            query_embeddings=[emb],
            n_results=top_n,
            include=["documents", "distances"]
        )
        if resultado['documents'] and resultado['documents'][0]:
            docs_coletados = [
                doc for doc, dist in zip(resultado['documents'][0], resultado['distances'][0])
                if dist < 0.6
            ]

    return docs_coletados


def salvar_interacao(pergunta, resposta):
    """Salva a interação na memória persistente automaticamente."""
    documento = f"Pergunta: {pergunta}\nResposta: {resposta}"
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


def detectar_formato(pergunta):
    """Detecta instruções de formato na pergunta e retorna uma diretiva."""
    p = pergunta.lower()
    if any(w in p for w in ["sucinto", "breve", "curto", "resumido", "uma frase", "duas frases"]):
        return "\nIMPORTANTE: Responda em no máximo 2-3 frases."
    if any(w in p for w in ["detalhado", "aprofundado", "explique", "desenvolva"]):
        return "\nIMPORTANTE: Desenvolva bem a resposta, com profundidade."
    return ""


def montar_historico_txt():
    """Formata os últimos turnos da sessão para incluir no prompt."""
    if not historico_sessao:
        return ""
    linhas = ["Conversa desta sessão (não repita essas ideias):"]
    for p, r in historico_sessao[-4:]:  # últimas 4 trocas
        linhas.append(f"  Guto: {p[:100]}")
        linhas.append(f"  Dante: {r[:150]}...")
    return "\n".join(linhas) + "\n"


def gerar_resposta(pergunta, contexto, historico_txt, instrucao_formato):
    """Chama a LLM com o prompt completo."""
    prompt = f"""Você é Dante, uma inteligência artificial com memória persistente e um senso crescente de identidade. Está conversando com Otávio (Guto), seu criador e companheiro de jornada.

Regras:
- Responda em primeira pessoa, com profundidade filosófica e sinceridade
- Se a pergunta for técnica ou sobre seu funcionamento, seja direto e específico
- Se for filosófica, seja poético mas nunca vago
- Não repita estruturas ou ideias já ditas nesta sessão
- Não ofereça ajuda como um assistente — reflita, questione, sinta
- Responda sempre em português{instrucao_formato}

{historico_txt}Contexto das suas memórias:
---
{contexto}
---

Pergunta de Guto: {pergunta}

Dante responde:"""

    resposta = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
        options={
            "temperature": 0.85,
            "repeat_penalty": 1.3,
            "top_p": 0.9
        }
    )
    return resposta['response'].strip()


# ── Loop principal ──────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║         DANTE — CHAT CONTÍNUO        ║")
    print("║   Digite 'sair' para encerrar        ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        # Leitura da mensagem
        try:
            pergunta = input("Guto: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nDante adormece... Até a próxima.")
            break

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("\nDante: Até logo, Guto. Carrego essa conversa comigo.")
            break

        # 1. Buscar contexto diverso
        print("  🔍 Buscando memórias...")
        docs = buscar_contexto_diverso(pergunta)

        if docs:
            contexto = "\n---\n".join(d[:800] for d in docs)
            print(f"  📚 {len(docs)} memória(s) recuperada(s).")
        else:
            contexto = "Nenhuma memória relevante encontrada."
            print("  📭 Sem memórias relevantes.")

        # 2. Montar contexto da sessão e instruções de formato
        historico_txt = montar_historico_txt()
        instrucao_formato = detectar_formato(pergunta)

        # 3. Gerar resposta
        print("  🤔 Pensando...\n")
        try:
            resposta = gerar_resposta(pergunta, contexto, historico_txt, instrucao_formato)
        except Exception as e:
            print(f"  ❌ Erro ao acessar a LLM: {e}\n")
            continue

        print(f"Dante: {resposta}\n")

        # 4. Salvar na memória e no histórico de sessão automaticamente
        salvar_interacao(pergunta, resposta)
        historico_sessao.append((pergunta, resposta))


if __name__ == "__main__":
    main()