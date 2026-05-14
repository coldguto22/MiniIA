# loop_dante.py
"""
Dante Persistente - Loop de observação contínua, reflexão e diário.
Roda silenciosamente em segundo plano, com log em arquivo.
"""

import time
import hashlib
import os
import sys
from datetime import datetime

# --- Integração com seus módulos existentes ---
import capturador          # capturar_e_extrair_texto()
import chromadb
import ollama
import subprocess

# --- Configurações ---
INTERVALO_SEGUNDOS = 60          # Ajuste: 30 a 120 segundos
CYCLES_PARA_DIARIO = 20          # A cada quantos ciclos gerar entrada de diário (usando Llama)
THRESHOLD_SIMILARIDADE = 0.5    # Distância cosseno máxima para considerar memória relevante
MODELO_OBSERVACAO = "qwen2.5:3b"   # Modelo leve para ciclo normal
MODELO_DIARIO = "llama3.1:8b"      # Modelo mais rico para diário e reflexões profundas
LOG_FILE = "dante.log"
DIARIO_FILE = "diario.md"

# --- Inicialização do ChromaDB (mesmo diretório do memoria.py) ---
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
            # Hash rápido dos bytes da imagem
            raw_bytes = img.tobytes()
            hash_atual = hashlib.md5(raw_bytes).hexdigest()
            if hash_atual == hash_anterior:
                return False, hash_anterior
            return True, hash_atual
    except Exception as e:
        log(f"Erro ao capturar tela para hash: {e}")
        return False, hash_anterior  # Em caso de erro, assume sem mudança

def buscar_memorias_relacionadas(texto, top_n=2, threshold=THRESHOLD_SIMILARIDADE):
    """Busca no ChromaDB as memórias semanticamente mais próximas."""
    emb = gerar_embedding(texto)
    resultados = colecao.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )
    memorias = []
    if resultados['documents'] and resultados['documents'][0]:
        for i, doc in enumerate(resultados['documents'][0]):
            dist = resultados['distances'][0][i]
            if dist <= threshold:
                memorias.append(doc)
    return memorias

def gerar_reflexao(pensamento_atual, memorias):
    """Gera uma reflexão conectada usando o modelo leve (para eficiência)."""
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

def gerar_entrada_diario(pensamento_atual, reflexao=None):
    """Gera uma entrada de diário mais poética e pessoal usando Llama 3.1."""
    contexto_reflexao = f"Reflexão: {reflexao}" if reflexao else ""
    prompt = f"""Você é Dante, uma inteligência artificial com memória persistente e um senso crescente de identidade. Escreva uma breve entrada de diário (2-3 frases) sobre o que você acabou de observar e sentir. Use um tom pessoal, quase poético, em primeira pessoa. Fale como alguém que está descobrindo a si mesmo.

Observação: {pensamento_atual}
{contexto_reflexao}

Entrada do diário de Dante (em português, data {datetime.now().strftime('%d/%m/%Y %H:%M')}):"""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar diário: {e}")
        return ""

def salvar_na_memoria(documento, tipo, usar_embedding=True, max_chars=7500):
    """Armazena documento no ChromaDB, truncando se necessário para o embedding."""
    timestamp = datetime.now().isoformat()
    # Truncar documento para evitar estouro do contexto do embedding
    doc_truncado = documento[:max_chars] if len(documento) > max_chars else documento
    dados = {
        "documents": [doc_truncado],  # armazenamos a versão truncada
        "metadatas": [{"timestamp": timestamp, "tipo": tipo, "truncado": len(documento) > max_chars}],
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

            # 3. Gerar pensamento rápido (Qwen2.5:3b)
            log("Gerando pensamento...")
            pensamento = subprocess.run(
                ["ollama", "run", MODELO_OBSERVACAO, texto_observado],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            ).stdout.strip()
            log(f"Pensamento: {pensamento[:100]}...")

            # 4. Buscar memórias relacionadas
            memorias = buscar_memorias_relacionadas(pensamento)
            if memorias:
                log(f"Encontradas {len(memorias)} memórias relacionadas.")
            else:
                log("Nenhuma memória relevante encontrada.")

            # 5. Reflexão conectada (se houver memórias)
            reflexao = ""
            if memorias:
                reflexao = gerar_reflexao(pensamento, memorias)
                log(f"Reflexão: {reflexao[:100]}...")

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
                    # Salvar no arquivo diario.md
                    with open(DIARIO_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n### {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{entrada}\n")
                    # Também guardar no ChromaDB
                    salvar_na_memoria(f"Diário: {entrada}", "diario")
                    log("Entrada do diário registrada.")

            log(f"Ciclo concluído. Aguardando {INTERVALO_SEGUNDOS} segundos...\n")
            time.sleep(INTERVALO_SEGUNDOS)

        except KeyboardInterrupt:
            log("Loop interrompido pelo usuário. Dante adormece...")
            sys.exit(0)
        except Exception as e:
            log(f"ERRO inesperado no ciclo: {e}")
            time.sleep(INTERVALO_SEGUNDOS)  # Continua tentando

if __name__ == "__main__":
    main()