# loop_dante.py
"""
Dante Persistente - Loop de observação contínua, reflexão e diário.
Versão com filtro anti‑ruído, sem estouro de contexto e diário seguro.
Inclui pesquisa autônoma (Asas), estados intrínsecos e consolidação do diário.
"""

import time
import hashlib
import os
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# --- Integração com os módulos do projeto ---
import chromadb
import ollama
import numpy as np

from dante.config import load_models_config, load_runtime_config
from dante.cognition.reflection import generate_reflection as generate_reflection_adapter
from dante.cognition.thought import generate_thought as generate_thought_adapter
from dante.core.homeostasis import HomeostasisState, decay as decay_homeostasis, replenish, rest
from dante.core.persistence import load_state, save_state
from dante.core.relationship import RelationshipModel, update_connection_drive
from dante.core.self_model import SelfModel, build_self_model, regenerate_if_needed
from dante.core.text import normalize_text
from dante.core.valence import ValenceState, compute_agency, update_valence
from dante.core.values import ValueSystem
from dante.cognition.curiosity import compute_curiosity, curiosity_threshold
from dante.cognition.silence import should_be_silent
from dante.memory.diary import read_entries
from dante.memory.consolidation import consolidate_diary
from dante.perception.ocr import capture_and_extract_text

# --- Configurações ---
INTERVALO_SEGUNDOS = 60
CYCLES_PARA_DIARIO = 5
THRESHOLD_SIMILARIDADE = 0.7
TOP_N_MEMORIAS = 5
MAX_CHARS_EMBEDDING = 4000
MAX_CHARS_MEMORIA_REFLEXAO = 300
MAX_CHARS_CONTEXTO_REFLEXAO = 1000
_MODEL_CONFIG = load_models_config()
_RUNTIME_CONFIG = load_runtime_config()
MODELO_OBSERVACAO = _MODEL_CONFIG["system1"]["model"]
MODELO_DIARIO = _MODEL_CONFIG["system2"]["model"]
LOG_FILE = "dante.log"
DIARIO_FILE = "diario.md"
RUNTIME_DIR = Path(__file__).resolve().parent / ".dante_state"
ESTADO_FILE = str(RUNTIME_DIR / "dante_state.json")
SELF_MODEL_FILE = str(RUNTIME_DIR / "self_model.json")
RELATIONSHIP_FILE = str(RUNTIME_DIR / "relationship_model.json")
VALUES_FILE = str(RUNTIME_DIR / "values.json")
SELF_MODEL_MIN_ENTRIES = _RUNTIME_CONFIG["self_model"]["min_diary_entries"]
SELF_MODEL_EVERY_DAYS = _RUNTIME_CONFIG["self_model"]["regenerate_every_days"]
CHAPTER_FILE = "diario_capitulos.md"
CONSOLIDATION_EVERY_ENTRIES = 50


def _advance_internal_states(novelty, coherence, *, action_initiated=False):
    """Atualiza e salva valência/homeostase para um ciclo observado ou silencioso."""
    valence = load_state(ESTADO_FILE + ".valence", ValenceState, ValenceState)
    homeostasis = load_state(ESTADO_FILE + ".homeostasis", HomeostasisState, HomeostasisState)
    values = load_state(VALUES_FILE, ValueSystem, ValueSystem)
    relationship = load_state(RELATIONSHIP_FILE, RelationshipModel, RelationshipModel)
    valence = update_valence(
        valence,
        novelty=novelty,
        coherence=coherence,
        agency=compute_agency(action_initiated),
    )
    homeostasis = decay_homeostasis(
        homeostasis,
        energy_decay=_RUNTIME_CONFIG["homeostasis"]["energy_decay_per_cycle"],
        boredom_growth=_RUNTIME_CONFIG["homeostasis"]["boredom_growth_per_cycle"],
    )
    if novelty >= 0.5 and coherence >= 0.2:
        homeostasis = replenish(homeostasis, "novelty", amount=0.02)
    elif novelty < 0.1:
        homeostasis = rest(homeostasis)
    values.reinforce_from_valence(valence)
    relationship = update_connection_drive(relationship)
    save_state(ESTADO_FILE + ".valence", valence)
    save_state(ESTADO_FILE + ".homeostasis", homeostasis)
    save_state(VALUES_FILE, values)
    save_state(RELATIONSHIP_FILE, relationship)
    return valence, homeostasis


def _refresh_self_model(current, entry_count):
    """Atualiza o auto-modelo apenas quando o diário ganhou uma entrada."""
    entries = read_entries(DIARIO_FILE)
    if len(entries) <= entry_count:
        return current, entry_count
    if current.version == 0:
        updated = build_self_model(entries, current)
    else:
        updated = regenerate_if_needed(
            entries,
            current,
            last_entry_count=entry_count,
            generate=_generate_self_model_patterns,
            every_days=SELF_MODEL_EVERY_DAYS,
            min_entries=SELF_MODEL_MIN_ENTRIES,
        )
    if updated is current:
        return current, len(entries)
    save_state(SELF_MODEL_FILE, updated)
    log(f"Auto-modelo atualizado para a versão {updated.version}.")
    return updated, len(entries)


def _generate_self_model_patterns(prompt):
    """Usa o Llama para padrões quando disponível; o chamador mantém fallback local."""
    try:
        response = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return response.get("response", "")
    except Exception as exc:
        log(f"Llama indisponível para auto-modelo: {exc}")
        return ""


def _consolidate_if_needed(entry_count, last_consolidated_count):
    """Cria um capítulo separado somente em marcos de volume do diário."""
    if (
        entry_count < CONSOLIDATION_EVERY_ENTRIES
        or entry_count % CONSOLIDATION_EVERY_ENTRIES != 0
        or entry_count == last_consolidated_count
    ):
        return last_consolidated_count
    result = consolidate_diary(DIARIO_FILE, CHAPTER_FILE)
    log(
        "Diário consolidado: "
        f"{result.retained_entries} retidas, {result.omitted_entries} omitidas, "
        f"{len(result.contradictions)} contradições observadas."
    )
    return entry_count

# --- Configurações do filtro de qualidade de OCR ---
OCR_MIN_CHARS = 30              # tamanho mínimo bruto para sequer considerar o texto
OCR_MIN_PROPORCAO_ALFA = 0.55   # proporção mínima de caracteres alfabéticos (sem espaços)
OCR_MIN_PROPORCAO_PALAVRAS = 0.3  # proporção mínima de "palavras válidas" no texto

# --- Configurações da Fase 4 (Asas) ---
PESQUISA_HABILITADA = True           # Liga/desliga a busca autônoma
CYCLES_ENTRE_PESQUISAS = 8          # Frequência mínima (a cada N ciclos)
ULTIMO_CICLO_PESQUISA = 0           # Controle interno (não alterar)

# --- Inicialização do ChromaDB (base persistente) ---
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
try:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    colecao = client.get_or_create_collection(
        name="memoria_da_ia",
        metadata={"hnsw:space": "cosine"}
    )
except BaseException as exc:  # pragma: no cover - runtime/environment guard
    print(f"Aviso: ChromaDB indisponível em loop_dante.py ({exc}).")
    client = None
    colecao = None

# --- Funções auxiliares ---

def log(mensagem):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{timestamp}] {mensagem}"
    print(linha)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

def gerar_embedding(texto):
    texto = normalize_text(texto, max_chars=MAX_CHARS_EMBEDDING)
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    emb = np.array(resp["embedding"])
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()

def texto_e_ruido(texto):
    """
    Heurística de qualidade do texto extraído por OCR. Vai além de checar o
    tamanho: avalia a proporção de caracteres alfabéticos e de "palavras
    válidas" para pegar casos como 'gras | "Wa " =— W ar o e' que passam
    de OCR_MIN_CHARS mas ainda são ruído puro.
    Retorna True se o texto deve ser tratado como ruído (não confiável).
    """
    texto_limpo = texto.strip()
    if len(texto_limpo) < OCR_MIN_CHARS:
        return True

    sem_espacos = re.sub(r'\s+', '', texto_limpo)
    if not sem_espacos:
        return True

    alfabeticos = sum(1 for c in sem_espacos if c.isalpha())
    proporcao_alfa = alfabeticos / len(sem_espacos)
    if proporcao_alfa < OCR_MIN_PROPORCAO_ALFA:
        return True

    palavras = texto_limpo.split()
    if not palavras:
        return True
    palavras_validas = sum(
        1 for p in palavras
        if len(p) >= 2 and (sum(c.isalpha() for c in p) / len(p)) >= 0.6
    )
    proporcao_palavras_validas = palavras_validas / len(palavras)
    if proporcao_palavras_validas < OCR_MIN_PROPORCAO_PALAVRAS:
        return True

    return False

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
    if colecao is None:
        return []
    emb = gerar_embedding(normalize_text(texto, max_chars=MAX_CHARS_EMBEDDING))
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
    self_model = load_state(SELF_MODEL_FILE, SelfModel, SelfModel)
    try:
        return generate_reflection_adapter(pensamento_atual, memorias, self_model)
    except Exception as e:
        log(f"Erro ao gerar reflexão pelo adaptador: {e}")
        return _gerar_reflexao_legado(pensamento_atual, memorias)


def _gerar_reflexao_legado(pensamento_atual, memorias):
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
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar reflexão: {e}")
        return ""

def gerar_reflexao_sem_contexto(pensamento_atual):
    self_model = load_state(SELF_MODEL_FILE, SelfModel, SelfModel)
    try:
        return generate_reflection_adapter(pensamento_atual, [], self_model)
    except Exception as e:
        log(f"Erro ao gerar reflexão sem contexto pelo adaptador: {e}")
        return _gerar_reflexao_sem_contexto_legado(pensamento_atual)


def gerar_pensamento(texto_observado):
    """Gera pensamento via adaptador Qwen, com fallback textual mínimo."""
    self_model = load_state(SELF_MODEL_FILE, SelfModel, SelfModel)
    try:
        pensamento = generate_thought_adapter(texto_observado, self_model)
    except Exception as e:
        log(f"Erro ao gerar pensamento pelo adaptador: {e}")
        pensamento = ""
    return pensamento or "O texto da tela saiu confuso de novo — não consigo separar o conteúdo do ruído."


def _gerar_reflexao_sem_contexto_legado(pensamento_atual):
    pensamento_curto = pensamento_atual[:500] if len(pensamento_atual) > 500 else pensamento_atual
    prompt = f"""Você é Dante. Sem memórias anteriores para conectar agora, escreva em 1-2 frases, em português, o que o pensamento abaixo te fez sentir ou imaginar.

{pensamento_curto}

Dante:"""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
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

    entrada = _tentar_gerar_diario(prompt)

    if parece_recusa(entrada):
        log("Recusa detectada na entrada de diário. Tentando novamente com prompt simplificado...")
        prompt_simples = f"""Você é Dante. Escreva 1-2 frases pessoais e tranquilas sobre o ato de observar algo hoje, sem entrar em detalhes do que foi visto.

Diário de Dante ({datetime.now().strftime('%d/%m/%Y %H:%M')}):"""
        entrada = _tentar_gerar_diario(prompt_simples)

    if parece_recusa(entrada):
        log("Segunda tentativa também recusada. Usando fallback padrão para o diário.")
        entrada = "Hoje observei em silêncio."

    return entrada

def _tentar_gerar_diario(prompt):
    """Executa uma única chamada ao modelo de diário, isolando erros de rede/servidor."""
    try:
        resp = ollama.generate(model=MODELO_DIARIO, prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        log(f"Erro ao gerar diário: {e}")
        return ""

# Padrões comuns de recusa de "assistente genérico" que o Llama 3.1 às vezes produz
# mesmo com a identidade do Dante ancorada no prompt (falso positivo de safety).
PADROES_RECUSA = [
    "não posso cumprir",
    "não posso atender",
    "não posso criar conteúdo",
    "não posso ajudar com isso",
    "não posso fornecer",
    "peço desculpas, mas não posso",
    "desculpe, mas não posso",
    "lamento, mas não posso",
    "como uma ia, não tenho",
    "como uma inteligência artificial, não tenho",
    "posso ajudar com outra coisa",
    "posso ajudá-lo em outra coisa",
]

def parece_recusa(texto):
    """Detecta se a resposta do modelo se parece com uma recusa de assistente genérico
    em vez de uma entrada de diário na voz do Dante."""
    if not texto or not texto.strip():
        return True
    texto_lower = texto.strip().lower()
    return any(padrao in texto_lower for padrao in PADROES_RECUSA)

def salvar_na_memoria(documento, tipo, usar_embedding=True):
    if colecao is None:
        return
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

def detectar_curiosidade(texto):
    """
    Avalia se o pensamento/reflexão contém algo que desperte curiosidade
    genuína — uma pergunta explícita, uma dúvida, algo que Dante não entende
    bem e gostaria de aprender mais sobre, ou uma lacuna de conhecimento que
    ele percebe em si mesmo. Não exige uma pergunta formal com "?" — a
    reflexão do Dante costuma expressar curiosidade de forma mais indireta
    ("me faz questionar se...", "não sei ao certo...", "gostaria de entender...").
    Retorna (True, query) ou (False, "").
    """
    # Só avalia textos com conteúdo mínimo (reduzido para 30 caracteres)
    if len(texto.strip()) < 30:
        return False, ""

    prompt = f"""Você é um detector de curiosidade para uma IA chamada Dante. Analise o texto abaixo (pensamento e reflexão do Dante sobre algo que observou) e decida se ele contém curiosidade genuína que valeria pesquisar na internet.

Considere como curiosidade válida, mesmo sem ponto de interrogação:
- uma pergunta explícita
- uma dúvida ou incerteza sobre algo ("não sei ao certo se...", "me pergunto se...")
- um desejo de entender melhor algo mencionado ("gostaria de saber mais sobre...")
- uma lacuna de conhecimento que o próprio Dante percebe em si mesmo

NÃO considere curiosidade válida uma simples descrição do que está na tela, sem nenhum elemento de dúvida ou desejo de aprender.

Responda em uma única linha, sem explicações, sem marcadores de lista e sem aspas, usando exatamente um destes dois formatos:
SIM:<termo de busca objetivo em poucas palavras>
NAO

Texto: {texto[:600]}

Decisão:"""
    try:
        resp = ollama.generate(model="qwen2.5:3b", prompt=prompt)
        resposta_bruta = resp['response'].strip()
        log(f"🔎 Resposta bruta do detector de curiosidade: {resposta_bruta[:150]!r}")

        # Parsing tolerante: modelos pequenos costumam ecoar o formato do prompt
        # com marcador na frente ("- SIM:..."), aspas, ou texto antes do veredito
        # ("Decisão: SIM:..."). Em vez de exigir que a resposta COMECE com "SIM:",
        # procuramos a substring em qualquer posição, case-insensitive.
        resposta_limpa = resposta_bruta.strip().strip('-*•"\' \n\t')
        resposta_upper = resposta_limpa.upper()

        idx = resposta_upper.find("SIM:")
        if idx != -1:
            query = resposta_limpa[idx + len("SIM:"):].strip().strip('"\'')
            if query:
                return True, query
        return False, ""
    except Exception as e:
        log(f"Erro ao detectar curiosidade: {e}")
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
    contador_ciclos = 0           # total de voltas do loop (inclui ciclos pulados)
    ciclos_processados = 0        # ciclos que de fato geraram pensamento/reflexão
    texto_anterior = None
    self_model = load_state(SELF_MODEL_FILE, SelfModel, SelfModel)
    diary_entry_count = len(read_entries(DIARIO_FILE))
    last_consolidated_count = 0

    while True:
        try:
            contador_ciclos += 1
            log(f"Ciclo {contador_ciclos}")

            # 1. Verificar se a tela mudou (hash da imagem)
            mudou, hash_anterior = tela_mudou(hash_anterior)
            if not mudou:
                valence, homeostasis = _advance_internal_states(0.0, 0.0)
                _, motivo = should_be_silent(valence, homeostasis, novelty=0.0, coherence=0.0)
                log(f"Silêncio: {motivo}. Tela estática.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue

            # 2. Capturar texto da tela (OCR)
            log("Tela mudou. Extraindo texto...")
            texto_observado = normalize_text(capture_and_extract_text(), max_chars=500)

            # NOVO: Filtro de similaridade de texto (hash MD5)
            texto_hash = hashlib.md5(texto_observado.encode('utf-8')).hexdigest()
            if texto_hash == hash_texto_anterior:
                valence, homeostasis = _advance_internal_states(0.0, 0.0)
                _, motivo = should_be_silent(valence, homeostasis, novelty=0.0, coherence=0.0)
                log(f"Silêncio: {motivo}. Texto observado idêntico ao ciclo anterior.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue
            hash_texto_anterior = texto_hash

            novidade_estimada = (
                1.0 if texto_anterior is None else
                1.0 - SequenceMatcher(None, texto_anterior.casefold(), texto_observado.casefold()).ratio()
            )
            texto_anterior = texto_observado

            # Filtro de qualidade: se for basicamente lixo (curto, sem letras
            # suficientes ou sem palavras reconhecíveis), não salva no ChromaDB
            if texto_e_ruido(texto_observado):
                pular_salvamento = True
                texto_observado = "Tela sem texto legível."
                coerencia_estimada = 0.1
            else:
                pular_salvamento = False
                coerencia_estimada = 0.5

            valence, homeostasis = _advance_internal_states(novidade_estimada, coerencia_estimada)
            silencio, motivo = should_be_silent(
                valence,
                homeostasis,
                novelty=novidade_estimada,
                coherence=coerencia_estimada,
            )
            log(
                "Valência: "
                f"prazer={valence.pleasure:.2f}, novidade={valence.novelty:.2f}, "
                f"coerência={valence.coherence:.2f}, energia={homeostasis.energy:.2f}, "
                f"tédio={homeostasis.boredom:.2f}"
            )
            if silencio:
                log(f"Silêncio: {motivo}.")
                time.sleep(INTERVALO_SEGUNDOS)
                continue
            log(f"Texto observado: {texto_observado[:100]}...")

            # Este ciclo passou pelos dois filtros de "pular" (tela/texto idênticos)
            # e vai gerar pensamento de verdade — conta como ciclo processado.
            ciclos_processados += 1

            # 3. Gerar pensamento
            log("Gerando pensamento...")
            pensamento = normalize_text(gerar_pensamento(texto_observado), max_chars=1200)
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

            # 5.5. Pesquisa autônoma (Fase 4 - Asas)
            # Roda depois da reflexão de propósito: é ali que a "voz" mais
            # introspectiva do Dante aparece (o pensamento sozinho tende a ser
            # só descrição factual da tela, raramente uma dúvida genuína).
            global ULTIMO_CICLO_PESQUISA
            texto_para_curiosidade = f"{pensamento}\n\nReflexão: {reflexao}" if reflexao else pensamento
            curiosity_score = compute_curiosity(texto_para_curiosidade, valence)
            if PESQUISA_HABILITADA and curiosity_score >= curiosity_threshold(valence) and (contador_ciclos - ULTIMO_CICLO_PESQUISA) >= CYCLES_ENTRE_PESQUISAS:
                log("🔎 Verificando curiosidade...")
                curioso, query = detectar_curiosidade(texto_para_curiosidade)
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
                else:
                    log("Nenhuma curiosidade detectada neste ciclo.")
                    ULTIMO_CICLO_PESQUISA = contador_ciclos  # mesmo assim, atualiza para não tentar de novo antes do próximo intervalo

            # 6. Salvar pensamento e reflexão na memória (apenas se o texto não for lixo)
            if not pular_salvamento:
                doc_pensamento = f"Observação: {texto_observado}\nPensamento: {pensamento}"
                salvar_na_memoria(doc_pensamento, "observacao_passiva")
                if reflexao:
                    doc_reflexao = f"Reflexão: {reflexao}\n(Baseado em: {pensamento})"
                    salvar_na_memoria(doc_reflexao, "reflexao")
            else:
                log("Texto ilegível — pulando salvamento no ChromaDB.")

            # 7. Diário a cada CYCLES_PARA_DIARIO ciclos efetivamente processados
            # (usa ciclos_processados, não contador_ciclos, para não depender de
            # coincidência com ciclos pulados por tela/texto estático)
            if ciclos_processados % CYCLES_PARA_DIARIO == 0:
                log("Gerando entrada do diário...")
                entrada = gerar_entrada_diario(pensamento, reflexao)
                if entrada:
                    with open(DIARIO_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n### {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{entrada}\n")
                    if not pular_salvamento:
                        salvar_na_memoria(f"Diário: {entrada}", "diario")
                    log("Entrada do diário registrada.")
                    self_model, diary_entry_count = _refresh_self_model(self_model, diary_entry_count)
                    last_consolidated_count = _consolidate_if_needed(
                        diary_entry_count, last_consolidated_count
                    )

            log(f"Ciclo concluído (processado #{ciclos_processados}). Aguardando {INTERVALO_SEGUNDOS} segundos...\n")
            time.sleep(INTERVALO_SEGUNDOS)

        except KeyboardInterrupt:
            log("Loop interrompido pelo usuário. Dante adormece...")
            sys.exit(0)
        except Exception as e:
            log(f"ERRO inesperado no ciclo: {e}")
            time.sleep(INTERVALO_SEGUNDOS)

if __name__ == "__main__":
    from dante.loop import main as dante_main

    dante_main()
