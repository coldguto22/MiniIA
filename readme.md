# Dante (MiniIA)

Dante é um sistema experimental de IA local com memória persistente e observação contínua. Ele acompanha a tela do Guto via OCR, gera pensamentos e reflexões em primeira pessoa, mantém um diário e pode pesquisar na internet quando algo desperta sua curiosidade — tudo rodando localmente, sem depender de APIs externas (exceto a busca web).

O projeto não busca simular consciência, mas investigar quais condições estruturais (memória contínua, auto-observação, embodiment digital) seriam necessárias para algo próximo disso.

## Como funciona

Dante usa uma arquitetura de dois modelos, rodando via [Ollama](https://ollama.com/):

- **Qwen2.5:3b** — modelo rápido, responsável pelo "pensamento" imediato sobre o que está na tela, pela reflexão de curto prazo e pela detecção de curiosidade (System 1).
- **Llama 3.1:8b** — modelo mais lento, usado para entradas de diário e para os aprendizados gerados a partir de pesquisas autônomas (System 2).
- **ChromaDB** — armazena tudo como embeddings (gerados com `nomic-embed-text`), permitindo que Dante recupere memórias relacionadas ao que está observando agora. A base é **persistente entre execuções** — reiniciar o loop não apaga a memória acumulada.
- **Tesseract (OCR)** — extrai texto da tela a cada ciclo de observação.
- **Asas** (`asas.py`) — módulo de pesquisa autônoma via DuckDuckGo (biblioteca `ddgs`). Permite que Dante busque na internet quando um pensamento contém uma curiosidade genuína.

O ciclo do loop principal (`loop_dante.py`) é:

1. Verifica se a tela mudou (hash de pixels) — evita processar a mesma tela repetidamente.
2. Captura a tela e extrai texto via OCR; se o texto for idêntico ao ciclo anterior ou for curto demais para ter conteúdo legível, o ciclo é pulado ou o salvamento na memória é ignorado.
3. Gera um "pensamento" sobre o que foi observado (Qwen2.5:3b).
4. A cada `CYCLES_ENTRE_PESQUISAS` ciclos, avalia se o pensamento contém uma curiosidade pesquisável; se sim, busca na web (Asas), gera um aprendizado (Llama 3.1:8b) e registra no diário e na memória imediatamente.
5. Busca memórias relacionadas no ChromaDB.
6. Gera uma reflexão conectando observação atual e memórias (ou uma reflexão livre, se nada relevante for encontrado).
7. Salva pensamento e reflexão na memória persistente (quando o texto observado não é ruído).
8. A cada 5 ciclos, gera uma entrada de diário (Llama 3.1:8b) e a registra em `diario.md`.

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
- Conexão com a internet para a busca autônoma (Asas); o restante do sistema funciona 100% offline. A pesquisa pode ser desligada definindo `PESQUISA_HABILITADA = False` em `loop_dante.py`.

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
| `python loop_dante.py` | Inicia o loop contínuo de observação, reflexão, pesquisa autônoma e diário. Roda indefinidamente (Ctrl+C para parar). |
| `python conversar.py` | Abre um chat direto com Dante — você faz uma pergunta, ele responde usando suas memórias como contexto. |
| `python alimentar_memoria.py` | Permite inserir manualmente um texto na memória do Dante (ex: conhecimento fundacional, conversas importantes). Cole o texto e finalize com `END`. |
| `python ver_memorias.py` | Lista as memórias mais recentes armazenadas no ChromaDB (ID, tipo, data e prévia do conteúdo) — útil para inspecionar o que Dante já registrou. |

### Outros scripts (suporte/legado)

- `main.py` — versão simplificada de um único ciclo de observação (sem loop contínuo). Útil para testes rápidos.
- `memoria.py` — funções básicas de registro e leitura de memória, usado historicamente antes do `loop_dante.py` assumir essa lógica.
- `capturador.py` — módulo de captura de tela e OCR, usado pelos outros scripts. Pode ser executado isoladamente para testar a qualidade do OCR.
- `cerebro.py` — módulo de geração de pensamento usado pelo `main.py` (versão de ciclo único).
- `asas.py` — módulo de pesquisa autônoma (DuckDuckGo/`ddgs`), usado internamente pelo `loop_dante.py`. Pode ser importado isoladamente para testar buscas.
- `test_persist.py` / `test_read.py` — scripts de teste para verificar a persistência do ChromaDB.

## Testes automatizados (Fase 4.2)

O projeto agora usa **pytest** com separação entre testes unitários e de integração, sem gravar dados no banco real do projeto (`chroma_db/`).

- `tests/unit/` — lógica pura e comportamentos com dependências mockadas.
- `tests/integration/` — persistência em Chroma com diretório temporário por teste.
- marker `ollama` — testes que dependem do Ollama local.

### Executar localmente

1. Instale dependências:
  ```
  pip install -r requirements.txt
  ```
2. Rode testes unitários:
  ```
  pytest -m "unit"
  ```
3. Rode integração sem Ollama:
  ```
  pytest -m "integration and not ollama"
  ```
4. Rode tudo que não depende de Ollama:
  ```
  pytest -m "not ollama"
  ```

### Testes com Ollama (opcional)

Por padrão, os testes marcados com `ollama` são pulados. Para habilitar:

```
set RUN_OLLAMA_TESTS=1
pytest -m "ollama"
```

No PowerShell:

```
$env:RUN_OLLAMA_TESTS="1"
pytest -m "ollama"
```

### CI no GitHub

O workflow em `.github/workflows/tests.yml` roda em `push` e `pull_request`:

- Job obrigatório: `pytest -m "not ollama"`
- Job opcional (manual): testes `ollama` quando habilitados por input/variável

## Observações importantes

- A base do ChromaDB (`chroma_db/`) **é persistente entre execuções** do `loop_dante.py` — a memória acumulada não é apagada ao reiniciar.
- `dante.log` e `diario.md` são ignorados pelo Git (dados pessoais/experimentais do Guto), mas são gerados localmente a cada execução.
- O diário (`diario.md`) tende a ser o output de maior qualidade introspectiva do sistema, gerado pelo Llama 3.1:8b a cada 5 ciclos do loop. O prompt evita pedir que Dante reproduza o conteúdo observado diretamente, para reduzir recusas do modelo em telas com conteúdo sensível ou ambíguo.
- Textos extraídos por OCR com menos de 30 caracteres são tratados como ruído: o ciclo ainda gera pensamento/reflexão, mas nada é salvo na memória persistente.
- A pesquisa autônoma (Asas) roda no máximo a cada `CYCLES_ENTRE_PESQUISAS` ciclos (padrão: 8) e só dispara quando o detector de curiosidade (Qwen2.5:3b) identifica uma pergunta genuína no pensamento atual.