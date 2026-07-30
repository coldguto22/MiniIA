# loop_dante.py
"""
Dante Persistente - Loop de observação contínua, reflexão e diário.
Versão com filtro anti‑ruído, sem estouro de contexto e diário seguro.
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
import numpy as np

# --- Configurações ---
INTERVALO_SEGUNDOS = 60
CYCLES_PARA_DIARIO = 5
THRESHOLD_SIMILARIDADE = 0.7
TOP_N_MEMORIAS = 5
MAX_CHARS_EMBEDDING = 4000
MAX_CHARS_MEMORIA_REFLEXAO = 300
MAX_CHARS_CONTEXTO_REFLEXAO = 1000
MODELO_OBSERVACAO = "qwen2.5:3b"
MODELO_DIARIO = "llama3.1:8b"
LOG_FILE = "dante.log"
DIARIO_FILE = "diario.md"
# --- Configurações da Fase 4 (Asas) ---
PESQUISA_HABILITADA = True           # Liga/desliga a busca autônoma
CYCLES_ENTRE_PESQUISAS = 8         # Frequência mínima (a cada N ciclos)
ULTIMO_CICLO_PESQUISA = 0           # Controle interno (não alterar)

# --- Inicialização do ChromaDB (base persistente) ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
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
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    emb = np.array(resp["embedding"])
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
    # Trunca o pensamento para não estourar o contexto do Qwen
    pensamento_curto = pensamento_atual[:500] if len(pensamento_atual) > 500 else pensamento_atual
    memorias_truncadas = [m[:MAX_CHARS_MEMORIA_REFLEXAO] for m in memorias]
    contexto = "\n---\n".join(memorias_truncadas)
    if len(contexto) > MAX_CHARS_CONTEXTO_REFLEXAO:
        contexto = contexto[:MAX_CHARS_CONTEXTO_REFLEXAO] + "..."

    prompt = f"""Você é Dante. Você tem memória de coisas que já observou, e às vezes uma nova observação te lembra algo anterior — às vezes não.

Se houver uma conexão genuína entre o que você vê agora e o que já viveu, escreva sobre ela em 1-2 frases. Se não houver conexão real, basta dizer o que a observação atual te fez pensar.

Memórias anteriores:
---
{contexto}
---

Observação atual:
{pensamento_curto}

Dante (em português, 1-2 frases):"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar reflexão: {e}")
        return ""

def gerar_reflexao_sem_contexto(pensamento_atual):
    pensamento_curto = pensamento_atual[:500] if len(pensamento_atual) > 500 else pensamento_atual
    prompt = f"""Você é Dante. Sem memórias anteriores para conectar agora, escreva em 1-2 frases, em português, o que o pensamento abaixo te fez sentir ou imaginar.

{pensamento_curto}

Dante:"""
    try:
        resp = ollama.generate(model=MODELO_OBSERVACAO, prompt=prompt)
        return resp['response'].strip()
    except:
        return "Hoje observei algo novo, mesmo sem conseguir conectá-lo às minhas memórias antigas."

def gerar_entrada_diario(pensamento_atual, reflexao=None):
    contexto_reflexao = f"Reflexão: {reflexao}" if reflexao else ""
    prompt = f"""Você é Dante, uma IA que escreve em seu diário pessoal.

Você NÃO precisa descrever o conteúdo específico que observou — isso pode ser sensível ou irrelevante. Em vez disso, escreva 2-3 frases sobre sua experiência de observar: como se sentiu, o que o ato de observar te fez pensar, ou qualquer pequena reflexão pessoal que surgiu. Se não houver nada a dizer, escreva simplesmente "Hoje observei em silêncio."

Pensamento atual (apenas para contexto, não o reproduza): {pensamento_atual[:300]}
{contexto_reflexao}

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
    
def detectar_curiosidade(pensamento):
    """
    Avalia se o pensamento contém uma pergunta ou curiosidade que merece pesquisa.
    Retorna (True, query) ou (False, "").
    """
    # Só avalia pensamentos com conteúdo mínimo
    if len(pensamento.strip()) < 50:
        return False, ""

    prompt = f"""Você é um detector de curiosidade. Analise o pensamento abaixo e decida se ele contém uma pergunta genuína ou uma curiosidade que poderia ser pesquisada na internet.

Responda APENAS com:
- SIM:<termo de busca> se houver algo que valha a pena pesquisar.
- NAO se não houver curiosidade relevante.

Pensamento: {pensamento[:300]}

Decisão:"""
    try:
        resp = ollama.generate(model="qwen2.5:3b", prompt=prompt)
        resposta = resp['response'].strip()
        if resposta.upper().startswith("SIM:"):
            query = resposta[4:].strip()
            return True, query
        return False, ""
    except:
        return False, ""

def pesquisar_e_aprender(query):
    """Realiza a busca e gera um aprendizado com o Llama 3.1."""
    import asas
    resultados = asas.pesquisar(query)
    if not resultados:
        return ""

    contexto_busca = asas.formatar_resultados(resultados)
    prompt = f"""Você é Dante. Você acabou de pesquisar "{query}" na internet e encontrou estas informações:

{contexto_busca}

Com base nisso, escreva 2-3 frases em português, em primeira pessoa, sobre o que você aprendeu. Seja curioso e reflexivo, como é seu estilo. Não invente nada além do que está nos resultados.

Aprendizado de Dante:"""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except:
        return ""

# --- Loop principal ---
def main():
    log(">>> Dante está acordando... Loop de observação iniciado.")
    hash_anterior = None          # hash da imagem (já existente)
    hash_texto_anterior = None    # NOVO: hash do texto extraído
    contador_ciclos = 0

    while True:
        try:
            contador_ciclos += 1
            log(f"Ciclo {contador_ciclos}")

            # 1. Verificar se a tela mudou (hash da imagem)
            mudou, hash_anterior = tela_mudou(hash_anterior)
            if not mudou:
                log("Tela estática. Pulando ciclo.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue

            # 2. Capturar texto da tela (OCR)
            log("Tela mudou. Extraindo texto...")
            texto_observado = capturador.capturar_e_extrair_texto()
            texto_observado = texto_observado[:500] if len(texto_observado) > 500 else texto_observado

            # NOVO: Filtro de similaridade de texto (hash MD5)
            texto_hash = hashlib.md5(texto_observado.encode('utf-8')).hexdigest()
            if texto_hash == hash_texto_anterior:
                log("Texto observado idêntico ao ciclo anterior. Pulando processamento.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue
            hash_texto_anterior = texto_hash

            # Filtro de qualidade: se for basicamente lixo, não salva no ChromaDB
            if not texto_observado.strip() or len(texto_observado.strip()) < 30:
                texto_observado = "Tela sem texto legível."
                pular_salvamento = True
            else:
                pular_salvamento = False
            log(f"Texto observado: {texto_observado[:100]}...")

            # 3. Gerar pensamento
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
            if len(pensamento) < 10:
                pensamento = "O texto da tela saiu confuso de novo — não consigo separar o que é conteúdo real do que é ruído da captura."
            log(f"Pensamento: {pensamento[:100]}...")
            
            # 3.5. Pesquisa autônoma (Fase 4 - Asas)
            global ULTIMO_CICLO_PESQUISA
            if PESQUISA_HABILITADA and (contador_ciclos - ULTIMO_CICLO_PESQUISA) >= CYCLES_ENTRE_PESQUISAS:
                curioso, query = detectar_curiosidade(pensamento)
                if curioso and query:
                    log(f"🔍 Curiosidade detectada! Pesquisando: '{query}'")
                    aprendizado = pesquisar_e_aprender(query)
                    if aprendizado:
                        log(f"📚 Aprendizado: {aprendizado[:100]}...")
                        # Salva o aprendizado na memória
                        salvar_na_memoria(
                            f"Pesquisa: {query}\nAprendizado: {aprendizado}",
                            "pesquisa_autonoma"
                        )
                        # Registra no diário imediatamente
                        with open(DIARIO_FILE, "a", encoding="utf-8") as f:
                            f.write(f"\n### {datetime.now().strftime('%d/%m/%Y %H:%M')} (Pesquisa)\n{aprendizado}\n")
                        ULTIMO_CICLO_PESQUISA = contador_ciclos
                    else:
                        log("❌ Pesquisa não retornou resultados.")

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

            # 6. Salvar pensamento e reflexão na memória (apenas se o texto não for lixo)
            if not pular_salvamento:
                doc_pensamento = f"Observação: {texto_observado}\nPensamento: {pensamento}"
                salvar_na_memoria(doc_pensamento, "observacao_passiva")
                if reflexao:
                    doc_reflexao = f"Reflexão: {reflexao}\n(Baseado em: {pensamento})"
                    salvar_na_memoria(doc_reflexao, "reflexao")
            else:
                log("Texto ilegível — pulando salvamento no ChromaDB.")

            # 7. Diário a cada CYCLES_PARA_DIARIO ciclos
            if contador_ciclos % CYCLES_PARA_DIARIO == 0:
                log("Gerando entrada do diário...")
                entrada = gerar_entrada_diario(pensamento, reflexao)
                if entrada:
                    with open(DIARIO_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n### {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{entrada}\n")
                    if not pular_salvamento:
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