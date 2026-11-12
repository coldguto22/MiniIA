# cerebro.py
from dante.cognition.thought import generate_thought
from dante.core.self_model import SelfModel

def pensar(texto_observado):
    """Encaminha o pensamento para o adapter modular configurado em System 1."""
    try:
        return generate_thought(texto_observado, SelfModel())
    except Exception as e:
        return f"Erro ao gerar pensamento: {e}"

if __name__ == "__main__":
    teste_observacao = "O usuário está lendo um artigo sobre 'Como plantar tomates' e tem uma planilha de custos do jardim aberta."
    print("🤔 Enviando para a LLM (como Dante)...")
    pensamento = pensar(teste_observacao)
    print("--- PENSAMENTO DE DANTE ---")
    print(pensamento)
    print("------------------------")
