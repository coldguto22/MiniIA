# Dante (MiniIA)

Dante é um sistema experimental de IA local com memória persistente e observação contínua. Ele acompanha a tela do Guto via OCR, gera pensamentos e reflexões em primeira pessoa, mantém um diário e pode pesquisar na internet quando algo desperta sua curiosidade — tudo rodando localmente, sem depender de APIs externas (exceto a busca web).

O projeto não busca simular consciência, mas investigar quais condições estruturais (memória contínua, auto-observação, embodiment digital) seriam necessárias para algo próximo disso.

## Entrega acadêmica

O histórico de decisão dos modelos, benchmarks e validações está em [BENCHMARKS_LLM.md](BENCHMARKS_LLM.md).

- **Disciplina:** Aprendizagem de Máquina, curso: Desenvolvimento de Sistemas Multiplataforma (DSM), Fatec
- **Professor:** Bruno Zolotareff
- **Repositório:** [github.com/coldguto22/MiniIA](https://github.com/coldguto22/MiniIA)
- **Integrantes:**
  - Jorge Rizzini de Castro Filho — [GitHub](https://github.com/JorgeRizzini)
  - Laísa Eugênio Severiano — [GitHub](https://github.com/LaisaEugenio)
  - Otávio Augusto de Lima Beato — [GitHub](https://github.com/coldguto22)

### Resumo da aplicação

Dante é um agente local experimental que observa a tela por OCR, produz pensamentos e reflexões, recupera memórias semânticas no ChromaDB e registra um diário. A pesquisa Dante–Mamba avalia se a continuidade de estado entre ciclos pode melhorar a continuidade funcional, a coerência temporal e a recuperação após reinício, sem afirmar consciência ou agência plena.

### Modelos utilizados

- **System 1:** configurado atualmente com Llama 3.2:latest. Qwen2.5:3b é usado como comparador nos benchmarks e pode ser configurado explicitamente como fallback.
- **System 2:** Llama 3.1:8b, mantido como modelo estabelecido para reflexão, diário, conversa e aprendizados de pesquisas.
- **Mamba:** protótipo experimental isolado, com contrato de sessão, reset e checkpoint; não é usado como modelo de produção enquanto não houver um backend funcional validado.
- **nomic-embed-text:** geração dos embeddings armazenados no ChromaDB.

### Fonte dos dados

O projeto não usa um dataset público externo como base principal. Os dados são provenientes de observações locais consentidas, capturadas por OCR, e de entradas manuais de memória fornecidas durante os testes. O armazenamento persistente é feito no [ChromaDB](https://www.trychroma.com/). Quando a pesquisa autônoma é acionada, os resultados são obtidos via [DuckDuckGo](https://duckduckgo.com/) usando a biblioteca [DDGS](https://pypi.org/project/ddgs/); esses resultados são tratados como fonte externa de pesquisa, não como dataset versionado.

O código é distribuído sob a licença [MIT](LICENSE).

## Como funciona

Dante mantém dois papéis cognitivos configuráveis via [Ollama](https://ollama.com/):

- **Llama 3.2:latest** — configuração atual do pensamento curto (System 1); o benchmark compara esse modelo com Qwen2.5:3b.
- **Llama 3.1:8b** — modelo estabelecido de System 2 para reflexão, diário, conversa e aprendizados de pesquisa; permanece inalterado.
- **ChromaDB** — armazena tudo como embeddings (gerados com `nomic-embed-text`), permitindo que Dante recupere memórias relacionadas ao que está observando agora. A base é **persistente entre execuções** — reiniciar o loop não apaga a memória acumulada.
- **Tesseract (OCR)** — extrai texto da tela a cada ciclo de observação.
- **Asas** (`asas.py`) — módulo de pesquisa autônoma via DuckDuckGo (biblioteca `ddgs`). Permite que Dante busque na internet quando um pensamento contém uma curiosidade genuína.

O ciclo do loop principal (`loop_dante.py`) é:

1. Verifica se a tela mudou (hash de pixels) — evita processar a mesma tela repetidamente.
2. Captura a tela e extrai texto via OCR; se o texto for idêntico ao ciclo anterior ou for curto demais para ter conteúdo legível, o ciclo é pulado ou o salvamento na memória é ignorado.
3. Gera um "pensamento" sobre o que foi observado usando o adapter modular de cognição (Llama 3.2:latest por padrão).
4. A cada `CYCLES_ENTRE_PESQUISAS` ciclos, avalia se o pensamento contém uma curiosidade pesquisável; se sim, busca na web (Asas), gera um aprendizado com o System 2 (Llama 3.1:8b) e registra no diário e na memória imediatamente.
5. Busca memórias relacionadas no ChromaDB.
6. Gera uma reflexão conectando observação atual e memórias (ou uma reflexão livre, se nada relevante for encontrado).
7. Salva pensamento e reflexão na memória persistente (quando o texto observado não é ruído).
8. A cada 5 ciclos, gera uma entrada de diário (Llama 3.1:8b) e a registra em `diario.md`.

Tudo é logado em `dante.log`.

## Arquitetura Mamba (protótipo experimental)

A arquitetura mantém os papéis System 1 e System 2 configuráveis e oferece um fluxo stateful isolado para experimentação:

- **System 1 (padrão):** Llama 3.2:latest, sem estado persistente experimental ativado.
- **Protótipo Mamba:** `MambaModel` + `ModelSession`, com reset, checkpoint/restauração e isolamento por sessão.
- **System 2 (Llama 3.1:8b):** responsável pelo diário, reflexão mais longa e geração de aprendizados; permanece como modelo estabelecido.
- **Fallback:** opcional por configuração; permanece desativado por padrão para usar Llama em todos os papéis. A geração stateful do Mamba falha explicitamente quando o backend não retorna estado serializável.
- **Limite importante:** `ollama-context` é continuação serializável de prompt, não acesso ao estado recorrente interno do Mamba.
- **Configuração centralizada:** `config/models.yaml` define os modelos e parâmetros de geração.

A configuração padrão está em `config/models.yaml`; para experimentar Mamba, use uma configuração separada e chame o adapter fora do loop principal.

### Contexto da entrega acadêmica (Fatec)

O Dante-Mamba é uma pesquisa experimental sobre continuidade de estado em agentes locais. O protótipo avalia, de forma reproduzível, se checkpoints e estado persistente melhoram a continuidade funcional em comparação com reconstrução de contexto e recuperação vetorial. As métricas planejadas incluem retenção temporal, coerência, consistência de prioridades, latência, RAM/VRAM, recuperação após reinício e vazamento entre sessões.

O trabalho separa quatro condições: baseline Transformer + RAG, Mamba com reset, Mamba persistente e Mamba persistente com burn-in. Esta versão entrega o contrato stateful, reset, checkpoint/restauração e testes com Ollama real; não constitui evidência de consciência ou agência plena.

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado e rodando, com os modelos de produção baixados:
  ```
  ollama pull llama3.2:latest
  ollama pull llama3.1:8b
  ollama pull qwen2.5:3b  # opcional; necessário apenas para comparar com Qwen
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

Para comparar Qwen2.5:3b e Llama 3.2:latest em latência, tokens medidos pelo Ollama e qualidade avaliada por um juiz Llama, execute `python benchmarks/compare_models.py`. O comando salva um JSON com as respostas completas e um resumo Markdown em `benchmarks/results/`. A avaliação de qualidade é auxiliar e deve ser revisada junto das respostas brutas. A primeira comparação está em [JSON](benchmarks/results/qwen_llama_20261112_152411.json) e [resumo](benchmarks/results/qwen_llama_20261112_152411.md).

Para comparar respostas ancoradas em OCR de uma captura real, execute `python benchmarks/compare_models_ocr.py --image C:\caminho\captura.png --lang por+eng --psm 3 --crop 250 100 1420 850`. O recorte é opcional; omita-o para usar a imagem inteira. O JSON registra o texto reconhecido, respostas completas, tokens, latência e notas de qualidade. O teste baseado na captura fornecida está em [JSON](benchmarks/results/qwen_llama_ocr_20261112_172625.json) e [resumo](benchmarks/results/qwen_llama_ocr_20261112_172625.md).

### Outros scripts (suporte/legado)

- `main.py` — versão simplificada de um único ciclo de observação (sem loop contínuo). Útil para testes rápidos.
- `memoria.py` — funções básicas de registro e leitura de memória, usado historicamente antes do `loop_dante.py` assumir essa lógica.
- `capturador.py` — módulo de captura de tela e OCR, usado pelos outros scripts. Pode ser executado isoladamente para testar a qualidade do OCR.
- `cerebro.py` — módulo de geração de pensamento usado pelo `main.py` (versão de ciclo único).
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
  pytest -m "unit"
  ```
3. Rode integração sem Ollama:
  ```
  pytest -m "integration and not ollama"
  ```
4. Rode testes de benchmark:
  ```
  pytest -m "benchmark"
  ```
5. Rode tudo que não depende de Ollama:
  ```
  pytest -m "not ollama"
  ```

### Cobertura e benchmarks

```bash
pytest --cov=dante --cov-report=html
python benchmarks/compare_system1.py
```

### Testes com Ollama (opcional)

Por padrão, os testes marcados com ollama são pulados. Para habilitar:

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
