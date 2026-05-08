# main.py
import capturador
import cerebro
import memoria
import time

print("MINI IA - INICIANDO CONSCIÊNCIA...")
print("="*30)

# 1. Capturar a tela
print("\n👁️ Observando...")
texto_observado = capturador.capturar_e_extrair_texto()
if texto_observado.strip():
    print(f"Texto observado: '{texto_observado[:100]}...'")
else:
    texto_observado = "Tela sem texto detectado ou majoritariamente gráfica."
    print(texto_observado)

# 2. Processar com a LLM (pensar)
print("\n🤔 Pensando...")
resumo_pensamento = cerebro.pensar(texto_observado)
print(f"Pensamento: {resumo_pensamento}")

# 3. Guardar na memória
print("\n🧠 Lembrando...")
memoria.registrar_lembranca(texto_observado, resumo_pensamento)

print("\n" + "="*30)
print("CICLO COMPLETO. Primeira experiência registrada.")