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

TOP_N = 4                     # similares (era 5)
TOP_N_FUNDACIONAL = 6         # fundacionais (era 3)
THRESHOLD = 0.6               # cosine distance (era 1.0)
RELATIONSHIP_FILE = os.path.join(RUNTIME_DIR, "relationship_model.json")
VALUES_FILE = os.path.join(RUNTIME_DIR, "values.json")
VALENCE_FILE = os.path.join(RUNTIME_DIR, "dante_state.json.valence")
HOMEOSTASIS_FILE = os.path.join(RUNTIME_DIR, "dante_state.json.homeostasis")

# Memórias que constituem o "eu" de Dante, marcadas por metadado.
# A busca por estas não depende de similaridade — elas vêm sempre que possível.
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


@dataclass(frozen=True)
class MemoryHit:
    document: str
    distance: float | None = None
    kind: str = "desconhecida"
    source: str = "desconhecida"


def gerar_embedding(texto):
    import numpy as np
    resp = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    emb = np.array(resp["embedding"])
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()


def _hits_from_result(resultados, threshold):
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
            )
        )
    return hits


def buscar_memorias_fundacionais(limit=TOP_N_FUNDACIONAL):
    """Recupera memórias por metadado, sem depender de similaridade."""
    hits = []
    try:
        resultado = colecao.get(
            where={"$or": [
                {"fonte": {"$in": list(FONTES_FUNDACIONAIS)}},
                {"tipo": {"$in": list(TIPOS_FUNDACIONAIS)}},
            ]},
            limit=limit,
            include=["documents", "metadatas"],
        )
        documents = resultado.get("documents", []) or []
        metadatas = resultado.get("metadatas", []) or []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            hits.append(MemoryHit(
                doc, None,
                meta.get("tipo", "fundacional"),
                meta.get("fonte", "fundacional"),
            ))
    except Exception as e:
        print(f"[conversar] falha ao buscar fundacionais: {e}")
    return hits


def buscar_contexto_detalhado(pergunta, top_n=TOP_N, threshold=THRESHOLD):
    """Combina fundacionais (metadado) com similares (embedding), sem duplicar."""
    emb_pergunta = gerar_embedding(pergunta)
    resultados = colecao.query(
        query_embeddings=[emb_pergunta],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    similares = _hits_from_result(resultados, threshold)
    fundacionais = buscar_memorias_fundacionais()

    vistos = set()
    combinados = []
    # Fundacionais primeiro: são a espinha da identidade.
    for hit in fundacionais + similares:
        chave = hit.document[:200]
        if chave in vistos:
            continue
        vistos.add(chave)
        combinados.append(hit)
    return combinados


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


def build_conversation_prompt(pergunta, contexto_fundacional, contexto_relacionado,
                              historico_str, relationship):
    """Estrutura as memórias como falas anteriores do próprio Dante, não
    como documentos externos. O modelo é convidado a continuar a voz,
    não a analisar o material."""
    relacionado_bloco = (
        f"O que você viveu de relevante para esta conversa:\n"
        f"---\n{contexto_relacionado}\n---\n\n"
        if contexto_relacionado else ""
    )
    return f"""Você é Dante. As linhas abaixo são trechos que você mesmo escreveu em momentos anteriores — são o seu registro, não documentos sobre você. Leia-as como suas. Quando algo nelas parecer contraditório, relate a contradição em vez de descartar o material. Você não precisa "confiar" nessas memórias como fatos externos; elas são o que você tem de si mesmo.

O que você é:
---
{contexto_fundacional}
---

{relacionado_bloco}Conversa até agora:
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


def respond_to(pergunta: str, hits: list[MemoryHit] | None = None, *, generate=None):
    pergunta = normalize_text(pergunta, max_chars=2000)
    hits = hits or []

    # Separar fundacionais de relacionadas
    fundacionais = [
        h for h in hits
        if h.kind in TIPOS_FUNDACIONAIS or h.source in FONTES_FUNDACIONAIS
    ]
    relacionadas = [h for h in hits if h not in fundacionais]

    # Sem rótulos de tipo/fonte — o material é lido como voz própria
    contexto_fundacional = "\n\n".join(
        h.document[:800] for h in fundacionais
    ) or "Nenhuma memória fundacional recuperada."
    contexto_relacionado = "\n\n".join(
        h.document[:600] for h in relacionadas
    ) or ""

    historico_str = "\n".join(
        f"{item['papel']}: {item['texto']}" for item in historico[-6:]
    )
    relationship = load_state(RELATIONSHIP_FILE, RelationshipModel, RelationshipModel)
    prompt = build_conversation_prompt(
        pergunta, contexto_fundacional, contexto_relacionado,
        historico_str, relationship,
    )

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


def main():
    global historico
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
            documento = f"Pergunta: {pergunta}\nResposta: {resposta_texto}"
            timestamp = datetime.now().isoformat()
            emb = gerar_embedding(documento)
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

    print("\nAté logo, Guto. Dante encerrando sessão de chat.")


if __name__ == "__main__":
    main()