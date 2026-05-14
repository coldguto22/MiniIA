# memoria.py
import chromadb
from chromadb.config import Settings
from datetime import datetime
import os

# Persistência local: pasta relativa ao projeto
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)
PERSIST_JSON = os.path.join(PERSIST_DIR, "memorias.json")
if not os.path.exists(PERSIST_JSON):
    try:
        with open(PERSIST_JSON, 'w', encoding='utf-8') as f:
            f.write('[]')
    except Exception:
        pass

client = None
settings = None
# Try multiple initialization patterns for chromadb client to support different versions
try:
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
    try:
        client = chromadb.Client(settings)
        print(f"ChromaDB client initialized with positional Settings; persist dir={PERSIST_DIR}")
    except Exception:
        try:
            client = chromadb.Client(settings=settings)
            print(f"ChromaDB client initialized with keyword 'settings'; persist dir={PERSIST_DIR}")
        except Exception:
            client = None
except Exception:
    settings = None

if client is None:
    try:
        client = chromadb.Client(persist_directory=PERSIST_DIR)
        print(f"ChromaDB client initialized with persist_directory kwarg; persist dir={PERSIST_DIR}")
    except Exception:
        client = chromadb.Client()
        print("ChromaDB client initialized with default constructor (in-memory)")

# Cria ou acessa uma coleção persistente chamada 'memoria_da_ia'
colecao = client.get_or_create_collection(name="memoria_da_ia")


def store_chunks(documents, metadatas, ids=None, embeddings=None):
    """Armazena uma lista de documentos (chunks) na coleção com metadados e embeddings opcionais.

    documents: list[str]
    metadatas: list[dict]
    ids: list[str] or None
    embeddings: list[list[float]] or None
    """
    if ids is None:
        # gera ids com timestamp + índice
        base = datetime.now().isoformat()
        ids = [f"{base}_{i}" for i in range(len(documents))]

    add_kwargs = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }
    if embeddings is not None:
        add_kwargs["embeddings"] = embeddings

    colecao.add(**add_kwargs)
    print(f"🧠 {len(documents)} chunk(s) armazenado(s) na coleção 'memoria_da_ia'.")
    # Tentativa de persistir em disco para garantir disponibilidade entre processos
    try:
        if hasattr(client, 'persist'):
            client.persist()
    except Exception as e:
        print(f"Aviso: não foi possível chamar client.persist(): {e}")
    # Também armazena em fallback JSON para garantir persistência entre processos
    try:
        import json
        existing = []
        try:
            with open(PERSIST_JSON, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []
        for i, doc in enumerate(documents):
            record = {
                'id': ids[i] if ids and i < len(ids) else f'{datetime.now().isoformat()}_{i}',
                'document': doc,
                'metadata': metadatas[i] if metadatas and i < len(metadatas) else {},
                'embedding': embeddings[i] if embeddings and i < len(embeddings) else None,
            }
            existing.append(record)
        with open(PERSIST_JSON, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Aviso: não foi possível escrever fallback JSON: {e}")


def semantic_search(query_embedding, top_n=3, where=None):
    """Consulta semântica usando um embedding de consulta.

    Retorna o resultado bruto da coleção.query para facilitar interpretação.
    """
    try:
        resultado = colecao.query(query_embeddings=[query_embedding], n_results=top_n, where=where)
        return resultado
    except Exception as e:
        # fallback: retorna peek quando query falhar
        print(f"Aviso: busca semântica falhou ({e}). Usando peek() ou fallback JSON.")
        try:
            peek = colecao.peek()
            # if peek has documents, return it
            if peek and peek.get('documents'):
                return peek
        except Exception:
            pass
        # fallback to reading the JSON file
        try:
            import json
            with open(PERSIST_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            docs = [r.get('document') for r in data]
            metas = [r.get('metadata') for r in data]
            ids = [r.get('id') for r in data]
            return {'ids': ids, 'documents': docs, 'metadatas': metas}
        except Exception:
            return colecao.peek()


def registrar_lembranca(observacao, pensamento, tipo="observacao_passiva", embedding=None):
    """Armazena uma memória simples (documento único) com metadados.

    Se `embedding` for fornecido, tenta salvar também o embedding para permitir buscas.
    """
    timestamp = datetime.now().isoformat()
    documento = f"Observação: {observacao}\nPensamento: {pensamento}"
    metadatas = [{"timestamp": timestamp, "tipo": tipo}]
    ids = [timestamp]
    embeddings = [embedding] if embedding is not None else None

    store_chunks([documento], metadatas, ids=ids, embeddings=embeddings)


def recordar(momento_id=None):
    """Recupera memórias. Pode ser a última ou todas."""
    if momento_id:
        resultado = colecao.get(ids=[momento_id])
    else:
        try:
            resultado = colecao.peek()  # Pega as últimas 10 adições
            # se não há documentos na coleção (ex: modo em memória), usar fallback JSON
            if resultado and resultado.get('documents'):
                return resultado
        except Exception:
            pass
        # fallback: ler arquivo JSON de persistência
        try:
            import json
            with open(PERSIST_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            docs = [r.get('document') for r in data]
            metas = [r.get('metadata') for r in data]
            ids = [r.get('id') for r in data]
            return {'ids': ids, 'documents': docs, 'metadatas': metas}
        except Exception:
            return {'ids': [], 'documents': [], 'metadatas': []}

    return resultado


if __name__ == "__main__":
    # Teste de registro
    registrar_lembranca("Tela inicial do Windows", "Ambiente de trabalho padrão, nada de extraordinário.")

    # Teste de recuperação
    print("--- ÚLTIMAS LEMBRANÇAS ---")
    lembrancas = recordar()
    for doc in lembrancas.get('documents', []):
        print(doc)
        print("-------------------------------")