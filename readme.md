# Dante (MiniIA)

Dante é um sistema experimental de IA local com memória persistente e observação contínua. Ele acompanha a tela do Guto via OCR, gera pensamentos e reflexões em primeira pessoa, e mantém um diário — tudo rodando localmente, sem depender de APIs externas.

O projeto não busca simular consciência, mas investigar quais condições estruturais (memória contínua, auto-observação, embodiment digital) seriam necessárias para algo próximo disso.

## Como funciona

Dante usa uma arquitetura de dois modelos, rodando via [Ollama](https://ollama.com/):

- **Qwen2.5:3b** — modelo rápido, responsável pelo "pensamento" imediato sobre o que está na tela (System 1).
- **Llama 3.1:8b** — modelo mais lento e usado para reflexões mais profundas e entradas de diário (System 2).
- **ChromaDB** — armazena tudo como embeddings (gerados com `nomic-embed-text`), permitindo que Dante recupere memórias relacionadas ao que está observando agora.
- **Tesseract (OCR)** — extrai texto da tela a cada ciclo de observação.

O ciclo básico do loop principal (`loop_dante.py`) é:

1. Verifica se a tela mudou (hash de pixels) — evita processar a mesma tela repetidamente.
2. Captura a tela e extrai texto via OCR.
3. Gera um "pensamento" sobre o que foi observado (Qwen2.5:3b).
4. Busca memórias relacionadas no ChromaDB.
5. Gera uma reflexão conectando observação atual e memórias (ou uma reflexão livre, se nada relevante for encontrado).
6. Salva pensamento e reflexão na memória persistente.
7. A cada 5 ciclos, gera uma entrada de diário (Llama 3.1:8b) e a registra em `diario.md`.

Tudo é logado em `dante.log`.

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado e rodando, com os modelos baixados:
  ```
  ollama pull qwen2.5:3b
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) instalado (com suporte ao idioma português — pacote `por`)

## Iniciando o ambiente (.venv)

Se o ambiente virtual já existe na pasta `.venv`, basta usar o script incluído no repositório:

```
ativar_venv.bat
```

Esse script:
- Ativa o `.venv` automaticamente
- Mostra os principais arquivos disponíveis para execução

Se o `.venv` ainda não existir, crie-o manualmente antes:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Principais comandos

Com o ambiente virtual ativado, execute o script desejado diretamente:

| Comando | O que faz |
|---|---|
| `python loop_dante.py` | Inicia o loop contínuo de observação, reflexão e diário. Roda indefinidamente (Ctrl+C para parar). |
| `python conversar.py` | Abre um chat direto com Dante — você faz uma pergunta, ele responde usando suas memórias como contexto. |
| `python alimentar_memoria.py` | Permite inserir manualmente um texto na memória do Dante (ex: conhecimento fundacional, conversas importantes). Cole o texto e finalize com `END`. |

### Outros scripts (suporte/legado)

- `main.py` — versão simplificada de um único ciclo de observação (sem loop contínuo). Útil para testes rápidos.
- `memoria.py` — funções básicas de registro e leitura de memória, usado historicamente antes do `loop_dante.py` assumir essa lógica.
- `capturador.py` — módulo de captura de tela e OCR, usado pelos outros scripts. Pode ser executado isoladamente para testar a qualidade do OCR.
- `cerebro.py` — módulo de geração de pensamento usado pelo `main.py`. Pode ser executado isoladamente para testar prompts.
- `test_persist.py` / `test_read.py` — scripts de teste para verificar a persistência do ChromaDB.

## Observações importantes

- O `loop_dante.py` **remove a base do ChromaDB (`chroma_db/`) a cada inicialização** para garantir a métrica de similaridade correta (cosine). Isso significa que reiniciar o loop apaga a memória acumulada — tenha isso em mente se quiser manter memórias entre sessões longas.
- `dante.log` e `diario.md` são ignorados pelo Git (dados pessoais/experimentais do Guto), mas são gerados localmente a cada execução.
- O diário (`diario.md`) tende a ser o output de maior qualidade introspectiva do sistema, gerado pelo Llama 3.1:8b a cada 5 ciclos do loop.