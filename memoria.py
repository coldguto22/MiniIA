# memoria.py
import chromadb
from datetime import datetime

# Inicializa o cliente ChromaDB (modo local)
client = chromadb.Client()

# Cria ou acessa uma coleção. É como uma tabela no SQL.
colecao = client.get_or_create_collection(name="memoria_da_ia")

def registrar_lembranca(observacao, pensamento):
    """Armazena uma nova memória na linha do tempo."""
    timestamp = datetime.now().isoformat()
    documento = f"Observação: {observacao}\nPensamento: {pensamento}"
    
    colecao.add(
        documents=[documento],
        metadatas=[{"timestamp": timestamp, "tipo": "observacao_passiva"}],
        ids=[timestamp]  # Usamos o timestamp como ID único
    )
    print(f"🧠 Lembrança registrada: {pensamento[:50]}...")

def recordar(momento_id=None):
    """Recupera memórias. Pode ser a última ou todas."""
    if momento_id:
        resultado = colecao.get(ids=[momento_id])
    else:
        resultado = colecao.peek() # Pega as últimas 10 adições
    return resultado

if __name__ == "__main__":
    # Teste de registro
    registrar_lembranca("Tela inicial do Windows", "Ambiente de trabalho padrão, nada de extraordinário.")
    
    # Teste de recuperação
    print("--- ÚLTIMAS LEMBRANÇAS ---")
    lembrancas = recordar()
    for doc in lembrancas['documents']:
        print(doc)
        print("-------------------------------")