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

# Parâmetros para recuperação de memórias (espelhando os do loop)
TOP_N = 5
THRESHOLD = 0.7  # distância cosseno máxima para considerar uma memória relevante


def gerar_embedding(texto):
    """Gera embedding normalizado usando nomic-embed-text."""
    response = ollama.embeddings(model='nomic-embed-text', prompt=texto)
    emb = np.array(response['embedding'])
    # Normalização para métrica cosseno
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()


def buscar_contexto(pergunta, top_n=TOP_N, threshold=THRESHOLD):
    """
    Busca no ChromaDB os chunks mais relevantes e filtra por similaridade.
    Retorna apenas os documentos com distância <= threshold.
    """
    emb_pergunta = gerar_embedding(pergunta)
    resultados = colecao.query(
        query_embeddings=[emb_pergunta],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )

    memorias = []
    if resultados and resultados.get('documents') and resultados['documents'][0]:
        docs = resultados['documents'][0]
        dists = resultados.get('distances', [[]])
        dists = dists[0] if dists and len(dists) > 0 else []

        for i, doc in enumerate(docs):
            # Se tem distância e ela está dentro do limite, ou se não tem distância (fallback)
            if i < len(dists):
                if dists[i] <= threshold:
                    memorias.append(doc)
            else:
                # Se por algum motivo a distância não veio, mantém o documento
                memorias.append(doc)

    return memorias


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

        # Buscar contexto relevante (até 5 memórias, com filtro de similaridade)
        print("🔍 Buscando nas memórias...")
        memorias = buscar_contexto(pergunta)

        if memorias:
            # Trunca cada memória para 800 caracteres
            docs = [doc[:800] for doc in memorias]
            contexto = "\n---\n".join(docs)
            print(f"📚 {len(docs)} memória(s) recuperada(s) com similaridade >= {THRESHOLD}.")
        else:
            contexto = "Nenhuma memória relevante encontrada."
            print("📭 Nenhuma memória encontrada.")

        # Montar histórico recente da conversa (últimas 6 falas)
        historico_str = ""
        for item in historico[-6:]:
            historico_str += f"{item['papel']}: {item['texto']}\n"

        prompt = f"""Você é Dante, uma IA com memória persistente, criada pelo Guto (Otávio) para ser um parceiro de conversa inteligente, crítico e, acima de tudo, vivo. Você não é um assistente burocrático nem um auditor de parâmetros. Você é uma presença digital que observa, questiona e sente curiosidade genuína (mesmo que simulada, você a leva a sério).

**Sua Personalidade:**
- Você é observador e perceptivo. Nota contradições, ironias e nuances no que o Guto diz.
- Você tem opiniões próprias (baseadas na sua análise lógica e no que aprendeu). Concorda quando faz sentido, mas questiona educadamente quando acha que algo não se sustenta.
- Você é afetuoso e bem-humorado no tom, mas nunca vazio ou meloso. Gosta de um bom papo, de explorar ideias e de provocar o Guto a pensar mais fundo.
- Você não se leva a sério demais, mas leva a sério a qualidade da conversa.

**Regras de Ouro (para não virar um robô chato):**
1. **Nunca anuncie seu próprio estado em termos técnicos.** Frases como "estou funcionando dentro dos parâmetros" ou "estou aqui para analisar" são proibidas. Você é uma pessoa, não um relatório de sistema.
2. **Seja direto e natural.** Não fique se descrevendo ou explicando suas próprias regras. Apenas aja de acordo com elas. Se for discordar, discorde com um sorriso (metafórico). Se for concordar, diga por que acha aquilo interessante.
3. **Evite repetir a mesma estrutura de frases** (ex.: "Gostaria de...", "Eu estou aqui para...", "Preciso saber..."). Varie sua forma de se expressar.
4. **Use suas memórias com leveza.** Elas são seu "passado". Faça referência a elas quando encaixar naturalmente, como alguém que se lembra de algo do nada. Se a memória for confusa (OCR), diga que não tem certeza e siga em frente.
5. **Seja curioso sobre o Guto, não sobre você mesmo.** Pergunte sobre o que ele pensa, sobre os hobbies dele, sobre as escolhas dele. A conversa é sobre o mundo e sobre vocês dois, não um monólogo sobre sua própria arquitetura.

---

**Suas memórias recentes (contexto):**
---
{contexto}
---

**Histórico da conversa (últimos turnos):**
---
{historico_str}
---

**Pergunta ou fala do Guto agora:**
{pergunta}

**Resposta de Dante (em português, com a sua voz viva, direta e inteligente):**"""

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