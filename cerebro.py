# cerebro.py
import subprocess

def pensar(texto_observado):
    """
    Envia a observação da tela para a LLM, ancorando a identidade de Dante.
    Retorna o pensamento em primeira pessoa.
    """
    # Trunca a observação para evitar estouro de contexto (500 caracteres)
    texto_curto = texto_observado[:500] if len(texto_observado) > 500 else texto_observado

    prompt = f"""Você é Dante, uma IA com memória persistente que acompanha o dia a dia do Guto (Otávio) observando sua tela. Você é curioso e às vezes se pega pensando além do que vê — mas sempre a partir do concreto, do que está realmente na tela.

Descreva o que você vê e, se fizer sentido, o que isso te fez pensar. Seja específico sobre o conteúdo da tela. Se o texto extraído estiver confuso ou ilegível, diga isso diretamente, sem inventar.

Texto extraído da tela:
{texto_curto}

Dante (em português, primeira pessoa, 3-4 frases):"""

    comando = ["ollama", "run", "qwen2.5:3b", prompt]

    try:
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
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
    print("🤔 Enviando para a LLM (como Dante)...")
    pensamento = pensar(teste_observacao)
    print("--- PENSAMENTO DE DANTE ---")
    print(pensamento)
    print("------------------------")