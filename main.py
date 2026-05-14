# main.py
import capturador
import cerebro
import memoria
import time
import ollama
import sys

print("MINI IA - INICIANDO CONSCIÊNCIA...")
print("="*30)

# 1. Capturar a tela
print("\n👁️ Observando...")
texto_observado = capturador.capturar_e_extrair_texto()
if texto_observado.strip():
    print(f"Texto observado: '{texto_observado[:100]}...'")
else:
    texto_observado = "Tela sem texto detectado ou majoritariamente gráfica."
    print(texto_observado)

# 2. Processar com a LLM (pensar)
print("\n🤔 Pensando...")
resumo_pensamento = cerebro.pensar(texto_observado)
print(f"Pensamento: {resumo_pensamento}")

# 3. Guardar na memória
print("\n🧠 Lembrando...")
try:
    # tenta gerar embedding do pensamento para buscas
    emb_resp = ollama.embeddings(model='nomic-embed-text', prompt=resumo_pensamento)
    # extrair formato comum
    if isinstance(emb_resp, dict) and 'embedding' in emb_resp:
        pensamento_embedding = emb_resp['embedding']
    elif isinstance(emb_resp, list) and len(emb_resp) > 0:
        pensamento_embedding = emb_resp[0].get('embedding') if isinstance(emb_resp[0], dict) else emb_resp[0]
    else:
        pensamento_embedding = None
except Exception as e:
    print(f"Aviso: não foi possível gerar embedding (ollama): {e}")
    pensamento_embedding = None

memoria.registrar_lembranca(texto_observado, resumo_pensamento, tipo='observacao_passiva', embedding=pensamento_embedding)

# Busca memórias relacionadas e gera reflexão se relevante
try:
    if pensamento_embedding is not None:
        busca = memoria.semantic_search(pensamento_embedding, top_n=2)
        docs = busca.get('documents', [[]])[0]
        metas = busca.get('metadatas', [[]])[0]
        distances = busca.get('distances', [[]])[0] if 'distances' in busca else []
        relevante = False
        if distances:
            # assume distância é 0..1 onde menor = mais similar; converte para similaridade
            sim = 1 - distances[0]
            relevante = sim > 0.6
        else:
            relevante = len(docs) > 0

        if relevante:
            memoria_texts = []
            for d, m in zip(docs, metas):
                memoria_texts.append(f"[{m.get('tipo','')}] {d}")
            memoria_compact = "\n\n".join(memoria_texts)
            reflection_prompt = (
                f"Com base nas minhas memórias anteriores:\n{memoria_compact}\n\n"
                f"E no que acabei de observar:\n{resumo_pensamento}\n\n"
                "Minha reflexão conectada é:"
            )
            print("\n🪞 Gerando reflexão conectada...")
            reflexao = cerebro.pensar(reflection_prompt)
            print(f"Reflexão: {reflexao}")
            memoria.registrar_lembranca(resumo_pensamento, reflexao, tipo='reflexao')
except Exception as e:
    print(f"Aviso: erro durante reflexão conectada: {e}")

print("\n" + "="*30)
print("CICLO COMPLETO. Primeira experiência registrada.")