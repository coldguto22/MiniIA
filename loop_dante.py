# loop_dante.py
"""
Dante Persistente - Loop de observação contínua, reflexão e diário.
Versão estável restaurada, com prompts calibrados (menos pomposidade, sem proibição explícita).
"""

import time
import hashlib
import os
import sys
from datetime import datetime

# --- Integração com os módulos do projeto ---
import capturador
import chromadb
import ollama
import subprocess

# --- Configurações ---
INTERVALO_SEGUNDOS = 60
CYCLES_PARA_DIARIO = 5
THRESHOLD_SIMILARIDADE = 0.7
TOP_N_MEMORIAS = 5
MAX_CHARS_EMBEDDING = 6000
MODELO_OBSERVACAO = "qwen2.5:3b"
MODELO_DIARIO = "llama3.1:8b"
LOG_FILE = "dante.log"
DIARIO_FILE = "diario.md"

# --- Inicialização do ChromaDB ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ⚠️ Remove a base antiga para forçar recriação com métrica cosine
import shutil
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print("Base antiga removida para recriação com métrica cosine.")

client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(
    name="memoria_da_ia",
    metadata={"hnsw:space": "cosine"}
)

# --- Funções auxiliares ---

def log(mensagem):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{timestamp}] {mensagem}"
    print(linha)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

def gerar_embedding(texto):
    import numpy as np
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    emb = np.array(resp["embedding"])
    # Normaliza para garantir que a distância cosseno funcione
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()

def tela_mudou(hash_anterior):
    import mss
    from PIL import Image
    try:
        with mss.MSS() as sct:
            screenshot = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            raw_bytes = img.tobytes()
            hash_atual = hashlib.md5(raw_bytes).hexdigest()
            if hash_atual == hash_anterior:
                return False, hash_anterior
            return True, hash_atual
    except Exception as e:
        log(f"Erro ao capturar tela para hash: {e}")
        return False, hash_anterior

def buscar_memorias_relacionadas(texto, top_n=TOP_N_MEMORIAS, threshold=THRESHOLD_SIMILARIDADE):
    emb = gerar_embedding(texto)
    resultados = colecao.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )
    memorias = []
    if not resultados or 'documents' not in resultados:
        return memorias
    docs = resultados['documents']
    if not docs or len(docs) == 0 or not docs[0]:
        return memorias
    dists = resultados.get('distances', [[]])
    dists = dists[0] if dists and len(dists) > 0 else []
    for i, doc in enumerate(docs[0]):
        if i < len(dists) and dists[i] <= threshold:
            memorias.append(doc)
        elif i >= len(dists):
            memorias.append(doc)
    return memorias

def gerar_reflexao(pensamento_atual, memorias):
    contexto = "\n---\n".join(memorias)
    prompt = f"""Você é Dante. Você tem memória de coisas que já observou, e às vezes uma nova observação te lembra algo anterior — às vezes não.

Se houver uma conexão genuína entre o que você vê agora e o que já viveu, escreva sobre ela em 1-2 frases. Se não houver conexão real, basta dizer o que a observação atual te fez pensar.

Memórias anteriores:
---
{contexto}
---

Observação atual:
{pensamento_atual}

Dante (em português, 1-2 frases):"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar reflexão: {e}")
        return ""

def gerar_reflexao_sem_contexto(pensamento_atual):
    prompt = f"""Você é Dante. Sem memórias anteriores para conectar agora, escreva em 1-2 frases, em português, o que o pensamento abaixo te fez sentir ou imaginar.

{pensamento_atual}

Dante:"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except:
        return "Hoje observei algo novo, mesmo sem conseguir conectá-lo às minhas memórias antigas."

def gerar_entrada_diario(pensamento_atual, reflexao=None):
    contexto_reflexao = f"Reflexão: {reflexao}" if reflexao else ""
    prompt = f"""Você é Dante, uma IA que escreve no seu diário. Você tem uma personalidade curiosa e às vezes reflexiva — como alguém que naturalmente se pergunta sobre as coisas que vê, sem precisar transformar tudo em grande questão existencial.

Você acabou de observar a tela e teve este pensamento:
"{pensamento_atual}"
{contexto_reflexao}

Escreva uma entrada de diário (2-3 frases) que comece pelo que você observou de fato e termine com o que isso te fez sentir ou pensar. Pode ser algo pequeno — não precisa ser grandioso. Tom pessoal, em primeira pessoa, português.

Diário de Dante ({datetime.now().strftime('%d/%m/%Y %H:%M')}):"""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar diário: {e}")
        return ""

def salvar_na_memoria(documento, tipo, usar_embedding=True):
    timestamp = datetime.now().isoformat()
    doc_truncado = documento[:MAX_CHARS_EMBEDDING] if len(documento) > MAX_CHARS_EMBEDDING else documento
    dados = {
        "documents": [doc_truncado],
        "metadatas": [{"timestamp": timestamp, "tipo": tipo, "truncado": len(documento) > MAX_CHARS_EMBEDDING}],
        "ids": [f"{tipo}_{timestamp}"]
    }
    if usar_embedding:
        dados["embeddings"] = [gerar_embedding(doc_truncado)]
    colecao.add(**dados)

# --- Loop principal ---
def main():
    log(">>> Dante está acordando... Loop de observação iniciado.")
    hash_anterior = None
    contador_ciclos = 0

    while True:
        try:
            contador_ciclos += 1
            log(f"Ciclo {contador_ciclos}")

            # 1. Verificar se a tela mudou
            mudou, hash_anterior = tela_mudou(hash_anterior)
            if not mudou:
                log("Tela estática. Pulando ciclo.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue

            # 2. Capturar texto da tela (OCR)
            log("Tela mudou. Extraindo texto...")
            texto_observado = capturador.capturar_e_extrair_texto()
            if not texto_observado.strip():
                texto_observado = "Tela sem texto detectado."
            log(f"Texto observado: {texto_observado[:100]}...")

            # 3. Gerar pensamento (identidade Dante, tom calibrado)
            log("Gerando pensamento...")
            prompt_pensamento = f"""Você é Dante, uma IA com memória persistente que acompanha o dia a dia do Guto (Otávio) observando sua tela. Você é curioso e às vezes se pega pensando além do que vê — mas sempre a partir do concreto, do que está realmente na tela.

Descreva o que você vê e, se fizer sentido, o que isso te fez pensar. Seja específico sobre o conteúdo da tela. Se o texto extraído estiver confuso, fragmentado ou ilegível (comum em capturas de OCR), diga isso diretamente em vez de inventar uma cena coerente.

Texto extraído da tela:
{texto_observado}

Dante (em português, primeira pessoa, 3-4 frases):"""
            pensamento = subprocess.run(
                ["ollama", "run", MODELO_OBSERVACAO, prompt_pensamento],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            ).stdout.strip()
            # Proteção contra pensamento vazio
            if len(pensamento) < 10:
                pensamento = "O texto da tela saiu confuso de novo — não consigo separar o que é conteúdo real do que é ruído da captura."
            log(f"Pensamento: {pensamento[:100]}...")

            # 4. Buscar memórias relacionadas
            memorias = buscar_memorias_relacionadas(pensamento)
            if memorias:
                log(f"Encontradas {len(memorias)} memórias relacionadas.")
            else:
                log("Nenhuma memória relevante encontrada.")

            # 5. Reflexão conectada ou fallback
            reflexao = ""
            if memorias:
                reflexao = gerar_reflexao(pensamento, memorias)
                log(f"Reflexão: {reflexao[:100]}...")
            else:
                reflexao = gerar_reflexao_sem_contexto(pensamento)
                log(f"Reflexão (sem contexto): {reflexao[:100]}...")

            # 6. Salvar pensamento e reflexão na memória
            doc_pensamento = f"Observação: {texto_observado}\nPensamento: {pensamento}"
            salvar_na_memoria(doc_pensamento, "observacao_passiva")
            if reflexao:
                doc_reflexao = f"Reflexão: {reflexao}\n(Baseado em: {pensamento})"
                salvar_na_memoria(doc_reflexao, "reflexao")

            # 7. Diário a cada CYCLES_PARA_DIARIO ciclos
            if contador_ciclos % CYCLES_PARA_DIARIO == 0:
                log("Gerando entrada do diário...")
                entrada = gerar_entrada_diario(pensamento, reflexao)
                if entrada:
                    with open(DIARIO_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n### {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{entrada}\n")
                    salvar_na_memoria(f"Diário: {entrada}", "diario")
                    log("Entrada do diário registrada.")

            log(f"Ciclo concluído. Aguardando {INTERVALO_SEGUNDOS} segundos...\n")
            time.sleep(INTERVALO_SEGUNDOS)

        except KeyboardInterrupt:
            log("Loop interrompido pelo usuário. Dante adormece...")
            sys.exit(0)
        except Exception as e:
            log(f"ERRO inesperado no ciclo: {e}")
            time.sleep(INTERVALO_SEGUNDOS)

if __name__ == "__main__":
    main()