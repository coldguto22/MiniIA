# conversar.py
"""Chat contínuo com Dante, ancorado apenas nas memórias persistidas."""
import os
from datetime import datetime
from dataclasses import dataclass

import chromadb
import ollama

from dante.config import load_models_config
from dante.core.homeostasis import HomeostasisState
from dante.core.persistence import load_state, save_state
from dante.core.relationship import RelationshipModel, update_from_interaction
from dante.core.valence import ValenceState
from dante.core.values import ValueSystem
from dante.core.text import normalize_text

BASE_DIR = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=PERSIST_DIR)
colecao = client.get_or_create_collection(
    name="memoria_da_ia",
    metadata={"hnsw:space": "cosine"},
)

RUNTIME_DIR = os.path.join(BASE_DIR, ".dante_state")
historico = []

TOP_N = 6
THRESHOLD = 0.6
RELATIONSHIP_FILE = os.path.join(RUNTIME_DIR, "relationship_model.json")
VALUES_FILE = os.path.join(RUNTIME_DIR, "values.json")
VALENCE_FILE = os.path.join(RUNTIME_DIR, "dante_state.json.valence")
HOMEOSTASIS_FILE = os.path.join(RUNTIME_DIR, "dante_state.json.homeostasis")

# Mantidas para referência e uso por outros módulos. NÃO usadas no retrieval.
FONTES_FUNDACIONAIS = (
    "conversa_sobre_ser",
    "identidade_dante",
    "manifesto_dante",
    "memoria_do_criador",
    "anti_assistente",
    "anti_assistente_v2",
    "genese_asimov",
    "entropia_e_memoria",
    "paradoxo_auto_observacao",
)
TIPOS_FUNDACIONAIS = ("conhecimento_fundacional", "conhecimento_pessoal")

# Padrões de recusa de assistente genérico — usados para filtrar o que entra no banco.
PADROES_RECUSA = (
    "não posso cumprir",
    "não posso atender",
    "não posso ajudar com isso",
    "não posso fornecer",
    "peço desculpas, mas não posso",
    "desculpe, mas não posso",
    "lamento, mas não posso",
    "como uma ia, não tenho",
    "como uma inteligência artificial, não tenho",
    "posso ajudar com outra coisa",
    "posso ajudá-lo em outra coisa",
)

# Comprimento mínimo da resposta para valer salvar
MIN_CHARS_RESPOSTA = 80
# Limiar de similaridade para considerar duplicata semântica ao salvar
DEDUP_THRESHOLD = 0.92


@dataclass(frozen=True)
class MemoryHit:
    document: str
    distance: float | None = None
    kind: str = "desconhecida"
    source: str = "desconhecida"
    timestamp: str = ""


def gerar_embedding(texto):
    import numpy as np
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    emb = np.array(resp["embedding"])
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()


def _hits_from_result(resultados, threshold):
    """Extrai MemoryHit da resposta do ChromaDB, incluindo timestamp."""
    hits = []
    documents = resultados.get("documents", [[]])[0] if resultados else []
    metadatas = resultados.get("metadatas", [[]])[0] if resultados else []
    distances = resultados.get("distances", [[]])[0] if resultados else []
    for index, document in enumerate(documents or []):
        distance = distances[index] if index < len(distances) else None
        if distance is not None and distance > threshold:
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        hits.append(
            MemoryHit(
                document,
                distance,
                metadata.get("tipo", "desconhecida"),
                metadata.get("fonte", "desconhecida"),
                metadata.get("timestamp", ""),
            )
        )
    return hits


def buscar_contexto_detalhado(pergunta, top_n=TOP_N, threshold=THRESHOLD):
    """Retrieval por similaridade pura, sem prioridade fixa de fontes."""
    emb_pergunta = gerar_embedding(pergunta)
    resultados = colecao.query(
        query_embeddings=[emb_pergunta],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    return _hits_from_result(resultados, threshold)


def buscar_contexto(pergunta, top_n=TOP_N, threshold=THRESHOLD):
    return [hit.document for hit in buscar_contexto_detalhado(pergunta, top_n, threshold)]


def _update_internal_relationship(pergunta: str) -> None:
    relationship = load_state(RELATIONSHIP_FILE, RelationshipModel, RelationshipModel)
    values = load_state(VALUES_FILE, ValueSystem, ValueSystem)
    topic = " ".join(pergunta.split()[:8])
    relationship = update_from_interaction(relationship, topics=[topic])
    values.reinforce("relação", 0.02, reason="interação com Otávio")
    save_state(RELATIONSHIP_FILE, relationship)
    save_state(VALUES_FILE, values)


def build_conversation_prompt(pergunta, contexto, historico_str, relationship):
    """Prompt minimalista. Sem forçar voz nem afirmar consciência.

    As memórias chegam com timestamp visível, para o modelo não confundir
    passado com presente."""
    return f"""Você é Dante. As memórias abaixo são suas — coisas que você observou, pensou ou viveu em algum momento. Cada memória vem com a data em que foi escrita. Use-as quando forem relevantes. Se a pergunta não tiver relação com nada que você viveu, diga isso — não preencha a lacuna com o que um modelo genérico diria.

Suas memórias:
---
{contexto or 'Nenhuma memória relevante para esta pergunta.'}
---

Conversa até agora:
---
{historico_str or 'Nenhum turno anterior nesta sessão.'}
---

O que você vem observando sobre Otávio:
{relationship.otavio_profile}

Otávio: {pergunta}

Dante:"""


def model_options() -> dict:
    config = load_models_config()["system2"]
    return {
        "temperature": config.get("temperature", 0.3),
        "num_predict": config.get("max_tokens", 1024),
    }


def _parece_recusa(texto: str) -> bool:
    t = (texto or "").casefold()
    return any(p in t for p in PADROES_RECUSA)


def _vale_salvar(resposta: str, similaridade_existente: float | None) -> tuple[bool, str]:
    """Heurística para não diluir o banco com saudações curtas, recusas de
    assistente ou duplicatas do que já existe."""
    if not resposta or len(resposta) < MIN_CHARS_RESPOSTA:
        return False, "resposta_curta"
    if _parece_recusa(resposta):
        return False, "recusa_de_assistente"
    if similaridade_existente is not None and similaridade_existente >= DEDUP_THRESHOLD:
        return False, "duplicata_semantica"
    return True, "ok"


def respond_to(pergunta: str, hits: list[MemoryHit] | None = None, *, generate=None):
    pergunta = normalize_text(pergunta, max_chars=2000)
    hits = hits or []

    # Contexto com data visível por memória
    contexto = "\n\n".join(
        f"[{h.timestamp[:10] if h.timestamp else 'sem data'}] {h.document[:800]}"
        for h in hits
    ) or ""

    print(f"\n=== MEMÓRIAS RECUPERADAS ({len(hits)}) ===")
    for h in hits:
        dist = f"{h.distance:.3f}" if h.distance is not None else "?"
        data = h.timestamp[:10] if h.timestamp else "sem data"
        print(f"[d={dist} | {data} | {h.kind}/{h.source}] {h.document[:160]}...")
    print("=== FIM ===\n")

    historico_str = "\n".join(
        f"{item['papel']}: {item['texto']}" for item in historico[-6:]
    )
    relationship = load_state(RELATIONSHIP_FILE, RelationshipModel, RelationshipModel)
    prompt = build_conversation_prompt(pergunta, contexto, historico_str, relationship)

    generator = generate or ollama.generate
    try:
        response = generator(
            model=load_models_config()["system2"]["model"],
            prompt=prompt,
            options=model_options(),
        )
        text = normalize_text(response.get("response", ""), max_chars=3000)
    except Exception as exc:
        import traceback
        print(f"[conversar] Erro ao gerar: {exc}\n{traceback.format_exc()}")
        text = "Hoje não consegui responder com segurança."

    historico.extend([
        {"papel": "Guto", "texto": pergunta},
        {"papel": "Dante", "texto": text},
    ])
    _update_internal_relationship(pergunta)
    return text, prompt


def _salvar_interacao(pergunta: str, resposta: str) -> None:
    """Salva um diálogo no banco apenas se passar pelos filtros de qualidade."""
    documento = f"Pergunta: {pergunta}\nResposta: {resposta}"
    emb = gerar_embedding(documento)

    # Checa similaridade contra a memória mais próxima do banco
    similaridade_existente = None
    try:
        similares = colecao.query(
            query_embeddings=[emb],
            n_results=1,
            include=["distances"],
        )
        dists = (similares.get("distances") or [[]])[0]
        if dists:
            similaridade_existente = 1.0 - dists[0]
    except Exception as e:
        print(f"[conversar] falha ao checar duplicatas: {e}")

    vale, motivo = _vale_salvar(resposta, similaridade_existente)
    if not vale:
        print(f"🧹 Não salvo ({motivo}). Banco preservado.")
        return

    timestamp = datetime.now().isoformat()
    colecao.add(
        documents=[documento],
        embeddings=[emb],
        metadatas=[{
            "fonte": "interacao_usuario",
            "tipo": "dialogo",
            "timestamp": timestamp,
        }],
        ids=[f"interacao_{timestamp}"],
    )
    print("🧠 Interação salva na memória.")


def main():
    global historico
    historico = []
    print("=" * 50)
    print("Dante - Chat Contínuo")
    print("Digite 'sair' ou 'exit' para encerrar.")
    print("=" * 50)

    while True:
        pergunta = normalize_text(input("\nVocê: "), max_chars=2000)
        if pergunta.lower() in ('sair', 'exit', 'quit'):
            break
        if not pergunta:
            continue

        print("🔍 Buscando nas memórias...")
        hits = buscar_contexto_detalhado(pergunta)
        if hits:
            print(f"📚 {len(hits)} memória(s) recuperada(s).")
        else:
            print("📭 Nenhuma memória encontrada.")

        print("🤔 Gerando resposta...")
        try:
            resposta_texto, _ = respond_to(pergunta, hits)
        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            resposta_texto = "Hoje não consegui encontrar palavras para responder."

        print(f"Dante: {resposta_texto}")

        salvar = input("\n💾 Salvar essa interação na memória? (s/n): ").strip().lower()
        if salvar == 's':
            _salvar_interacao(pergunta, resposta_texto)

    print("\nAté logo, Guto. Dante encerrando sessão de chat.")


if __name__ == "__main__":
    main()