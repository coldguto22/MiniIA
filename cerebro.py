# cerebro.py
import subprocess
import json

def pensar(observacao):
    """Envia uma observação para a LLM e retorna a sua reflexão."""
    prompt = f"Você é uma IA observadora. Analise o seguinte texto extraído da tela do usuário e dê um resumo conciso do que pode estar acontecendo. Seu foco é entender a atividade do usuário em poucas palavras.\n\nTexto: {observacao}\n\nResumo:"
    
    comando = ["ollama", "run", "qwen2.5:3b", prompt]
    
    try:
        # Adicionamos encoding='utf-8' e ignoramos erros de caracteres inválidos
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            encoding='utf-8',           # ESSENCIAL para evitar UnicodeDecodeError
            errors='replace'             # Substitui caracteres corrompidos
        )
        if resultado.returncode == 0:
            return resultado.stdout.strip()
        else:
            return f"Erro ao acessar a LLM (código {resultado.returncode}): {resultado.stderr}"
    except FileNotFoundError:
        return "Erro: Comando 'ollama' não encontrado. O Ollama está instalado e em execução?"
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

if __name__ == "__main__":
    teste_observacao = "O usuário está lendo um artigo sobre 'Como plantar tomates' e tem uma planilha de custos do jardim aberta."
    print("🤔 Enviando para a LLM...")
    pensamento = pensar(teste_observacao)
    print("--- PENSAMENTO DA IA ---")
    print(pensamento)
    print("------------------------")