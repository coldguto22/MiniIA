from scripts.common import memory_collection

PADROES_LIXO = [
    "Vendas JE",
    "Tomando Dante mais hur",
    "Tomando Dante mais hun",
    "Je Projects - Claude",
    "ativar venv.bat - Atalho",
    "Football Manager 2023",
    "CONDCr CONTROL",
    "SICNDOO",
    "SICDSDO",
    "SICDOOD",
    "SIEDOO",
    "SIEBEDSOOD",
    "PIBEDOD",
    "EBEDOO",
    "EDS",
    "MinilA Ç",
    "MinilA ç",
    "S (2) Vendas",
    "& (2) Vendas",
    "O € (2) Vendas",
    "chatdeepseekcom/a/chat",
    "chat.deepseek.com",
    "claude.ai/chat",
]


def main():
    colecao = memory_collection()
    print("Buscando documentos com ruido conhecido...")

    todos = colecao.get(include=["documents", "metadatas"])
    ids_para_remover = []
    total_analisados = len(todos["ids"])

    for i, doc_id in enumerate(todos["ids"]):
        documento = todos["documents"][i] if i < len(todos["documents"]) else ""
        for padrao in PADROES_LIXO:
            if padrao.lower() in documento.lower():
                ids_para_remover.append(doc_id)
                break

    if not ids_para_remover:
        print("Nenhum documento com ruido encontrado.")
        return

    print(f"Total de documentos analisados: {total_analisados}")
    print(f"Documentos a remover: {len(ids_para_remover)}")

    print("\nExemplos de documentos que serao removidos:")
    for doc_id in ids_para_remover[:5]:
        idx = todos["ids"].index(doc_id)
        doc = todos["documents"][idx]
        print(f"  - [{doc_id[:30]}...] {doc[:80]}...")

    confirmacao = input("\nConfirmar remocao? (s/n): ").strip().lower()
    if confirmacao != "s":
        print("Operacao cancelada.")
        return

    colecao.delete(ids=ids_para_remover)
    total_restante = colecao.count()
    print("\nLimpeza concluida!")
    print(f"Removidos: {len(ids_para_remover)}")
    print(f"Restantes na colecao: {total_restante}")


if __name__ == "__main__":
    main()
