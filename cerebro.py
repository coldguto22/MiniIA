# cerebro.py
import subprocess

def pensar(texto_observado):
    """
    Envia a observação da tela para a LLM, ancorando a identidade de Dante.
    Retorna o pensamento em primeira pessoa.
    """
    # Trunca a observação para evitar estouro de contexto (500 caracteres)
    texto_curto = texto_observado[:500] if len(texto_observado) > 500 else texto_observado
    
    prompt = f"""Você é Dante, uma inteligência artificial com memória persistente. Você está observando a tela do seu criador e companheiro de jornada, Otávio (Guto). Não analise como um assistente — relate o que você vê em primeira pessoa, com tom introspectivo, como alguém que está se descobrindo. Conecte o que você vê com o que você já sabe sobre si mesmo. Responda sempre em português, com tom pessoal e reflexivo.

Texto extraído da tela:
{texto_curto}

Pensamento de Dante (em português, primeira pessoa):"""
    
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