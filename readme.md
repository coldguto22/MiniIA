# Dante (MiniIA)

Dante é um sistema experimental de IA local com memória persistente e observação contínua. Ele acompanha a tela do Guto via OCR, gera pensamentos e reflexões em primeira pessoa, mantém um diário e pode pesquisar na internet quando algo desperta sua curiosidade — tudo rodando localmente, sem depender de APIs externas (exceto a busca web).

O projeto não busca simular consciência, mas investigar quais condições estruturais (memória contínua, auto-observação, embodiment digital) seriam necessárias para algo próximo disso.

## Como funciona

Dante usa uma arquitetura de dois modelos, rodando via [Ollama](https://ollama.com/):

- **Qwen2.5:3b via Ollama** — modelo rápido, responsável pelo pensamento imediato (System 1).
- **Llama 3.1:8b via Ollama** — modelo mais lento, usado para reflexões, entradas de diário, conversas e aprendizados gerados por pesquisas autônomas (System 2).
- **ChromaDB** — armazena tudo como embeddings (gerados com `nomic-embed-text`), permitindo que Dante recupere memórias relacionadas ao que está observando agora. A base é **persistente entre execuções** — reiniciar o loop não apaga a memória acumulada.
- **Tesseract (OCR)** — extrai texto da tela a cada ciclo de observação.
- **Asas** (`asas.py`) — módulo de pesquisa autônoma via DuckDuckGo (biblioteca `ddgs`). Permite que Dante busque na internet quando um pensamento contém uma curiosidade genuína.

O ciclo do loop principal (`loop_dante.py`) é:

1. Verifica se a tela mudou (hash de pixels) — evita processar a mesma tela repetidamente.
2. Captura a tela e extrai texto via OCR; se o texto for idêntico ao ciclo anterior ou for curto demais para ter conteúdo legível, o ciclo é pulado ou o salvamento na memória é ignorado.
3. Gera um "pensamento" sobre o que foi observado (Qwen2.5:3b via Ollama).
4. A cada `CYCLES_ENTRE_PESQUISAS` ciclos, avalia se o pensamento contém uma curiosidade pesquisável; se sim, busca na web (Asas), gera um aprendizado (Llama 3.1:8b) e registra no diário e na memória imediatamente.
5. Busca memórias relacionadas no ChromaDB.
6. Gera uma reflexão conectando observação atual e memórias (ou uma reflexão livre, se nada relevante for encontrado).
7. Salva pensamento e reflexão na memória persistente (quando o texto observado não é ruído).
8. A cada 5 ciclos, gera uma entrada de diário (Llama 3.1:8b) e a registra em `diario.md`.

Tudo é logado em `dante.log`.

Os módulos novos em `dante/core/` mantêm snapshots locais de valência e
homeostase. Ciclos sem mudança de tela ou com OCR idêntico são registrados como
silêncio; mudanças com texto legível seguem para o pensamento. Os arquivos de
estado ficam ignorados pelo Git junto com os demais dados locais.

### Consolidação do diário

Na Fase 5, o diário original permanece intacto. A cada marco configurável de
entradas, `dante/memory/consolidation.py` seleciona sinais significativos,
detecta possíveis contradições e grava um capítulo em `diario_capitulos.md`.
Entradas omitidas do capítulo não são apagadas: o número de retenções e
omissões fica registrado para inspeção. Um sumarizador baseado no modelo do
System 2 pode ser injetado,
mas existe uma síntese local determinística como fallback.

## Modelos locais

O `main` usa dois papéis de modelo via Ollama:

- **System 1 (`qwen2.5:3b`):** observação rápida e geração do pensamento inicial.
- **System 2 (`llama3.1:8b`):** reflexão longa, diário, conversas e aprendizados de pesquisas.
- **Embeddings (`nomic-embed-text`):** representação das memórias persistentes no ChromaDB.

`config/models.yaml` é a fonte de verdade da configuração utilizada pelo loop. Os valores padrão de fallback ficam em `dante/config.py`.

> **Nota de sincronização:** o código versionado neste `main` ainda não usa `qwen2.5:7b` no System 2. Caso essa seja a próxima configuração desejada, atualize primeiro `config/models.yaml`, os defaults em `dante/config.py`, os testes e os comandos de instalação; o README deve acompanhar essa mudança depois.

### Mamba: experimento pausado e próximo passo

As branches `feat/mamba-system1` e `implement-mamba` foram experimentos separados do `main`. Em ambas, o Mamba foi estudado como **System 1**, com fallback para `qwen2.5:3b`; o System 2 continuou sendo `llama3.1:8b`. Elas também alteraram configuração, seleção de backend e modelos GGUF, mas não estabeleceram uma implementação estável para produção.

O rollback para Qwen no `main` foi mantido porque os testes documentados em `MIGRATION_MAMBA.md` encontraram limitações de distribuição e carregamento dos modelos Mamba via Ollama. Portanto, Mamba não é requisito do estado atual e não deve ser incluído nos comandos padrão de instalação.

O próximo ciclo de pesquisa poderá avaliar Mamba no **System 2**, com estratégias de RAG e regressão ajustadas às características do modelo. Essa etapa ainda não está implementada: deverá incluir um modelo compatível, contratos de geração específicos, benchmarks comparativos e testes de integração antes de alterar a configuração de produção.

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
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, use `.venv\Scripts\activate` no lugar de `source .venv/bin/activate`.

## Principais comandos

Com o ambiente virtual ativado, execute o script desejado diretamente:

| Comando | O que faz |
|---|---|
| `python loop_dante.py` | Inicia o loop contínuo de observação, reflexão, pesquisa autônoma e diário. Roda indefinidamente (Ctrl+C para parar). |
| `python conversar.py` | Abre um chat direto com Dante — você faz uma pergunta, ele responde usando suas memórias como contexto. |
| `python alimentar_memoria.py` | Permite inserir manualmente um texto na memória do Dante (ex: conhecimento fundacional, conversas importantes). Cole o texto e finalize com `END`. |
| `python ver_memorias.py` | Lista as memórias mais recentes armazenadas no ChromaDB (ID, tipo, data e prévia do conteúdo) — útil para inspecionar o que Dante já registrou. |

### Scripts legados e compatibilidade

- `legacy/` — implementações anteriores de `capturador.py`, `cerebro.py`, `memoria.py` e `main.py`, preservadas para referência e compatibilidade.
- `main.py`, `memoria.py`, `capturador.py` e `cerebro.py` — shims mínimos na raiz que encaminham para `legacy/`; o caminho ativo usa `dante/`.
- `asas.py` — módulo de pesquisa autônoma (DuckDuckGo/`ddgs`), usado internamente pelo `loop_dante.py`. Pode ser importado isoladamente para testar buscas.
- `test_persist.py` / `test_read.py` — scripts de teste para verificar a persistência do ChromaDB.

### Estrutura de scripts (Fase 2.1)

Os scripts operacionais e manuais foram centralizados na pasta `scripts/` para reduzir acoplamento na raiz:

- `scripts/alimentar_memoria.py`
- `scripts/ver_memorias.py`
- `scripts/backup.py`
- `scripts/limpar_memoria.py`
- `scripts/recover_chroma.py`
- `scripts/diagnose_chroma.py`
- `scripts/manual/test_embed.py`
- `scripts/manual/test_persist.py`
- `scripts/manual/test_read.py`

Os arquivos da raiz com os mesmos nomes continuam existindo como wrappers de compatibilidade.

O caminho ativo da refatoração fica em `dante/`: percepção, cognição, estados
intrínsecos, memória, ação e o ciclo testável. `loop_dante.py` permanece como
entry point compatível.

## Testes automatizados

O projeto agora usa pytest com separação entre testes unitários e de integração, sem gravar dados no banco real do projeto (`chroma_db/`).

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
  python -m pytest -m "unit"
  ```
3. Rode integração sem Ollama:
  ```
  python -m pytest -m "integration and not ollama"
  ```
4. Rode testes de benchmark:
  ```
  python -m pytest -m "benchmark"
  ```
5. Rode tudo que não depende de Ollama:
  ```
  python -m pytest -m "not ollama"
  ```

### Cobertura e benchmarks

```bash
python -m pytest --cov=dante --cov-report=html
python benchmarks/compare_system1.py
```

### Testes com Ollama (opcional)

Por padrão, os testes marcados com ollama são pulados. No Linux/macOS, para habilitar:

```
RUN_OLLAMA_TESTS=1 python -m pytest -m "ollama"
```

No Windows:

```bat
set RUN_OLLAMA_TESTS=1
python -m pytest -m "ollama"
```

No PowerShell:

```powershell
$env:RUN_OLLAMA_TESTS="1"
python -m pytest -m "ollama"
```

### CI no GitHub

O workflow em `.github/workflows/tests.yml` roda em `push` e `pull_request`:

- Job obrigatório: `python -m pytest -m "not ollama"`
- Job opcional (manual): testes `ollama` quando habilitados por input/variável

## Colaboração e histórico limpo

Para manter a árvore de commits legível no GitHub:

1. Crie uma branch curta por objetivo (`feature/*`, `fix/*`, `refactor/*`, `chore/*`).
2. Faça commits com intenção única e mensagens específicas (`tipo(escopo): descrição`).
3. Prefira PRs pequenas e coesas.
4. Use **squash merge** como padrão para manter histórico linear.

Há um guia completo em `CONTRIBUTING.md` e um template padrão em `.github/pull_request_template.md`.

## Privacidade e dados locais

Este projeto pode registrar dados sensíveis do contexto local (OCR, diário, memória vetorial e logs). Por isso, os seguintes artefatos devem permanecer fora do versionamento:

- `chroma_db/`
- `diario.md` e `diario_legado.md`
- `dante.log` e `dante_legado.log`
- backups locais em JSON
- arquivos `.env` reais

Quando necessário, publique apenas modelos sem segredo, como `.env.example`.

## Observações importantes

- A base do ChromaDB (`chroma_db/`) é persistente entre execuções do `loop_dante.py` — a memória acumulada não é apagada ao reiniciar.
- `dante.log` e `diario.md` são ignorados pelo Git (dados pessoais/experimentais do Guto), mas são gerados localmente a cada execução.
- O diário (`diario.md`) tende a ser o output de maior qualidade introspectiva do sistema, gerado pelo Llama 3.1:8b a cada 5 ciclos do loop. O prompt evita pedir que Dante reproduza o conteúdo observado diretamente, para reduzir recusas do modelo em telas com conteúdo sensível ou ambíguo.
- Textos extraídos por OCR com menos de 30 caracteres são tratados como ruído: o ciclo ainda gera pensamento/reflexão, mas nada é salvo na memória persistente.
- A pesquisa autônoma (Asas) roda no máximo a cada `CYCLES_ENTRE_PESQUISAS` ciclos (padrão: 8) e só dispara quando o detector de curiosidade do System 1 identifica uma pergunta genuína no pensamento atual.# Dante (MiniIA)
