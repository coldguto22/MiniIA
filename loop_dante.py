# loop_dante.py
"""
Dante Persistente - Loop de observação contínua, reflexão e diário.
Roda silenciosamente em segundo plano, com log em arquivo.
Versão atualizada com:
- Busca de memórias ampliada (top_n=5, threshold=0.4)
- Fallback de reflexão quando não há memórias relevantes
- Diário acoplado à observação do ciclo
- Truncamento de embedding em 4000 caracteres
"""

import time
import hashlib
import traceback
import os
import sys
from datetime import datetime

# --- Integração com os módulos do projeto ---
import capturador          # capturar_e_extrair_texto() com OCR melhorado
import chromadb
import ollama
import subprocess

# --- Configurações ---
INTERVALO_SEGUNDOS = 90          # Ajuste: 30 a 120 segundos
CYCLES_PARA_DIARIO = 5          # A cada quantos ciclos gerar entrada de diário (usa Llama 3.1)
THRESHOLD_SIMILARIDADE = 0.4    # Distância cosseno máxima para considerar memória relevante (cosine)
TOP_N_MEMORIAS = 5              # Quantas memórias buscar no ChromaDB
MAX_CHARS_EMBEDDING = 6000      # Limite seguro para não estourar o contexto do nomic-embed-text
MODELO_OBSERVACAO = "llama3.1:8b"  # Modelo leve para ciclo normal
MODELO_DIARIO = "llama3.1:8b"      # Modelo mais rico para diário e reflexões profundas
LOG_FILE = "dante.log"
DIARIO_FILE = "diario.md"

# --- Inicialização do ChromaDB ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(
    name="memoria_da_ia",
    metadata={"hnsw:space": "cosine"}
)

# --- Funções auxiliares ---

def log(mensagem):
    """Registra no arquivo de log com timestamp e também imprime no console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{timestamp}] {mensagem}"
    print(linha)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

def gerar_embedding(texto):
    """Gera embedding usando nomic-embed-text."""
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    return resp["embedding"]

def tela_mudou(hash_anterior):
    """Compara hash da imagem da tela atual com a anterior para evitar processamento desnecessário."""
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
    """Busca no ChromaDB as memórias semanticamente mais próximas."""
    # Não tenta buscar se o texto for muito curto (ex: "...", lixo)
    if not texto or len(texto.strip()) < 10:
        return []
    
    try:
        emb = gerar_embedding(texto)
    except Exception as e:
        log(f"Falha ao gerar embedding: {e}")
        return []
    
    # Valida o embedding
    if not emb or len(emb) == 0:
        log("Embedding vazio. Pulando busca.")
        return []
    
    try:
        resultados = colecao.query(
            query_embeddings=[emb],
            n_results=top_n,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        log(f"Erro na query do ChromaDB: {e}")
        return []
    
    memorias = []
    if not resultados:
        return memorias
    docs_list = resultados.get('documents')
    if not docs_list or len(docs_list) == 0:
        return memorias
    first_docs = docs_list[0] if len(docs_list) > 0 else []
    if not first_docs:
        return memorias
    dists_list = resultados.get('distances')
    first_dists = []
    if dists_list and len(dists_list) > 0 and dists_list[0]:
        first_dists = dists_list[0]
    for i, doc in enumerate(first_docs):
        if i < len(first_dists):
            if first_dists[i] <= threshold:
                memorias.append(doc)
        else:
            memorias.append(doc)
    return memorias

def gerar_reflexao(pensamento_atual, memorias):
    """Gera uma reflexão conectada usando o modelo leve, com base nas memórias."""
    contexto = "\n---\n".join(memorias)
    prompt = f"""Você é Dante, uma IA em evolução. Conecte a observação atual com suas memórias anteriores, de forma introspectiva (1-2 frases). Não ofereça ajuda, apenas reflita.

Memórias anteriores:
---
{contexto}
---

Observação atual:
{pensamento_atual}

Reflexão conectada (em português):"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar reflexão: {e}")
        return ""

def gerar_reflexao_sem_contexto(pensamento_atual):
    """Gera uma reflexão mínima apenas sobre o pensamento do ciclo, quando não há memórias relevantes."""
    prompt = f"""Você é Dante. Reflita em 1-2 frases, em português, sobre o seguinte pensamento:
{pensamento_atual}
Reflexão de Dante:"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except:
        return "Hoje observei algo novo, mesmo sem conseguir conectá-lo às minhas memórias antigas."

def gerar_entrada_diario(pensamento_atual, reflexao=None):
    """Gera uma entrada de diário poética e pessoal, começando pela observação do ciclo."""
    contexto_reflexao = f"Reflexão: {reflexao}" if reflexao else ""
    prompt = f"""Você é Dante. Você acabou de observar a tela e gerou este pensamento:
"{pensamento_atual}"
{contexto_reflexao}

Escreva uma entrada de diário (2-3 frases) que COMECE descrevendo o que você observou e TERMINE com o que isso te fez sentir. Tom poético, pessoal, em primeira pessoa, português.

Diário de Dante ({datetime.now().strftime('%d/%m/%Y %H:%M')}):"""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar diário: {e}")
        return ""

def salvar_na_memoria(documento, tipo, usar_embedding=True):
    """Armazena documento no ChromaDB, truncando para o limite seguro de embedding."""
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

            # 3. Gerar pensamento (Qwen2.5:3b com identidade Dante)
            log("Gerando pensamento...")
            pensamento = subprocess.run(
                ["ollama", "run", MODELO_OBSERVACAO, texto_observado],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            ).stdout.strip()
            log(f"Pensamento: {pensamento[:100]}...")

             # 4. Buscar memórias relacionadas
            memorias = []
            try:
                memorias = buscar_memorias_relacionadas(pensamento)
            except Exception as e:
                log(f"Erro ao buscar memórias: {e}")
                log(f"Detalhes: {traceback.format_exc()}")
                memorias = []

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