import memoria
import cerebro
import ollama
import sys


def get_embedding(text):
    try:
        resp = ollama.embeddings(model='nomic-embed-text', prompt=text)
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


def build_context_from_results(resultado):
    # chromadb retorna listas aninhadas em query
    docs = resultado.get('documents', [[]])[0]
    metas = resultado.get('metadatas', [[]])[0]
    context_parts = []
    for d, m in zip(docs, metas):
        fonte = m.get('fonte', 'desconhecida')
        tipo = m.get('tipo', '')
        context_parts.append(f"Fonte: {fonte} | Tipo: {tipo}\n{d}")
    return "\n\n".join(context_parts)


def main():
    print("Digite sua pergunta (ou CTRL+C para sair):")
    try:
        pergunta = input("Pergunta: ")
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    emb = get_embedding(pergunta)
    if emb is None:
        print("Não foi possível gerar embedding da pergunta. Abortando.")
        return

    resultado = memoria.semantic_search(emb, top_n=3)
    contexto = build_context_from_results(resultado)

    prompt = (
        "Você é uma IA com acesso às suas próprias memórias e anotações.\n"
        "Contexto recuperado das suas memórias:\n\n"
        f"{contexto}\n\n"
        f"Pergunta: {pergunta}\n"
        "Responda com base no contexto acima. Se o contexto não for relevante, responda com seu conhecimento geral."
    )

    print('\n🤖 Perguntando ao modelo...')
    resposta = cerebro.pensar(prompt)
    print('\n--- RESPOSTA ---')
    print(resposta)
    print('----------------')

    salvar = input('Salvar essa interação na memória? (s/n): ').strip().lower()
    if salvar == 's':
        memoria.registrar_lembranca(pergunta, resposta, tipo='interacao')
        print('Interação salva.')


if __name__ == '__main__':
    main()
