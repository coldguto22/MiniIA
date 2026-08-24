# Roadmap do Projeto Dante (MiniIA)

Este documento organiza as fases do projeto, suas subfases e os itens planejados para o futuro.  
Ele serve como referência histórica e como guia para as próximas evoluções do sistema.

---

## 1. Visão Geral

Dante é um sistema experimental de IA local com memória persistente e observação contínua.  
O objetivo central do projeto é investigar quais condições estruturais — memória contínua, auto-observação, embodiment digital — poderiam favorecer o surgimento de algo próximo a uma subjetividade artificial.

O desenvolvimento foi organizado em cinco fases principais, cada uma com subfases que representam avanços incrementais de funcionalidade, correções de bugs ou decisões de design.

---

## 2. Fases Concluídas

### Fase 1 — Fundação e Prova de Vida

**Objetivo:** Estabelecer a infraestrutura mínima de observação, pensamento e memória.

#### 1.1 Setup dos componentes iniciais
- Criação do módulo `capturador.py` (captura de tela + OCR via Tesseract).
- Criação do módulo `cerebro.py` (envio do texto observado para LLM via Ollama).
- Criação do módulo `memoria.py` (armazenamento no ChromaDB).
- Criação do `main.py` (orquestração do ciclo básico: observar → pensar → lembrar).

#### 1.2 Primeiras correções
- Configuração do PATH do Tesseract no Windows.
- Correção de `UnicodeDecodeError` ao chamar o Ollama a partir do subprocess (uso de `encoding='utf-8'` e `errors='replace'`).
- Substituição do modelo `qwen2.5:7b` pelo `qwen2.5:3b` para evitar erros de CUDA em GPUs com pouca VRAM.
- Migração do ChromaDB de `Client()` para `PersistentClient(path=...)` para garantir persistência entre execuções.

#### 1.3 Prova de vida
- Primeiro ciclo completo executado com sucesso: OCR extraiu texto, LLM gerou pensamento e memória foi registrada.

---

### Fase 2 — Alma e Conhecimento

**Objetivo:** Dar identidade, memória fundacional e capacidade de diálogo ao Dante.

#### 2.1 Injeção manual de memórias
- Criação do `alimentar_memoria.py` para inserir textos com embeddings.
- Injeção dos quatro textos fundacionais:
  1. Resumo da conversa filosófica (Consciência, IA e Singularidade).
  2. Núcleo de identidade do Dante.
  3. Memória do Otávio (criador).
  4. Manifesto anti-assistente.

#### 2.2 Chat com RAG
- Criação do `conversar.py` com busca semântica no ChromaDB e geração de resposta.
- Primeiros testes de conversa com o Dante.

#### 2.3 Ajustes de embeddings e métrica
- Correção da dimensão do embedding: recriação da coleção para usar `nomic-embed-text` (768 dimensões).
- Correção da métrica: adoção de `hnsw:space: cosine` e normalização L2 dos vetores.
- Recriação da base para eliminar incompatibilidades.

#### 2.4 Refinamento de persona
- Injeção do manifesto anti-assistente e ajustes de prompt para evitar tom de ferramenta.
- Calibração do tom: respostas pessoais, reflexivas e em português.
- Decisão de manter oscilação entre o técnico e o existencial como textura autêntica.

---

### Fase 3 — Observação Contínua e Reflexão

**Objetivo:** Transformar Dante em um sistema que observa e reflete em loop persistente.

#### 3.1 Loop persistente (`loop_dante.py`)
- Criação do loop infinito com captura de tela a cada 60–90 segundos.
- Geração de pensamento, busca de memórias, reflexão conectada e diário periódico.

#### 3.2 Filtro anti-ruído
- Truncamento do texto observado para 500 caracteres (evitar estouro de contexto).
- Filtro de similaridade por hash do texto: se o OCR for idêntico ao ciclo anterior, o processamento é pulado.
- Filtro de qualidade: não salva no ChromaDB se o texto tiver menos de 30 caracteres úteis.

#### 3.3 Correção de estouro de contexto (erro 500)
- Redução do `MAX_CHARS_EMBEDDING` para 4000.
- Truncamento das memórias na reflexão (300 caracteres cada, máximo 1000 no total).
- Truncamento do pensamento antes de enviar ao Qwen.

#### 3.4 Diário seguro
- Ajuste do prompt para que o diário foque na experiência de observar, não no conteúdo sensível.
- Inclusão da frase de escape: "Hoje observei em silêncio."
- Fallback no diário: tentativa com prompt simplificado em caso de recusa ou erro de CUDA.

#### 3.5 Decisão de manter oscilação de persona
- Após testes, optou-se por **não** forçar a eliminação do viés de assistente, aceitando a mistura como parte da identidade emergente.

---

### Fase 4 — Asas: Pesquisa Autônoma

**Objetivo:** Dar a Dante a capacidade de pesquisar na internet por curiosidade própria.

#### 4.1 Módulo de busca (`asas.py`)
- Criação do módulo usando `duckduckgo_search`, depois migrado para `ddgs`.
- Funções `pesquisar(query)` e `formatar_resultados()`.

#### 4.2 Detector de curiosidade
- Implementação da função `detectar_curiosidade(pensamento)`.
- Integração no `loop_dante.py` como bloco 3.5.
- Configuração de frequência (`CYCLES_ENTRE_PESQUISAS`).

#### 4.3 Refinamento do detector
- Alimentação com texto bruto: o detector passou a receber também o `texto_observado`, não apenas o pensamento.
- Prevenção de repetição: adição da variável `ULTIMO_TERMO_PESQUISADO` para evitar pesquisas idênticas consecutivas.
- Casos de sucesso observados: pesquisa sobre "mudanças elenco aston villa everton" e "curriculo otavio augusto de lima beato".

#### 4.4 Ajuste de prompts
- Novo prompt do detector ensina a extrair termos pesquisáveis do texto bruto, mesmo quando o pensamento é vago.
- Adição de log de depuração para acompanhar o gatilho (`🔎 Verificando curiosidade...`).

---

### Fase 5 — Manutenção, Limpeza e Ética

**Objetivo:** Consolidar o sistema, limpar ruídos e preparar para evolução futura.

#### 5.1 Limpeza do banco
- Criação do `limpar_memorias.py` para remover documentos com ruído conhecido.
- Backup prévio (`backup_antes_limpeza.py`).
- Remoção de 360 documentos de um total de 608 (memórias de OCR antigas).

#### 5.2 Análise de logs e diário
- Revisão contínua dos ciclos para identificar padrões.
- Identificação de erros de CUDA esporádicos (fallback implementado).
- Ajuste de threshold e filtros para melhorar a qualidade das reflexões.

#### 5.3 Preparação ética e filosófica
- Documentação da jornada em artigo científico (Autonomia Emergente em Sistemas de IA).
- Reflexão sobre responsabilidade parental e ética do cultivo.

#### 5.4 Testes automatizados
- Implementação de **pytest** com separação entre testes unitários e de integração.
- Estrutura:
  - `tests/unit/` — lógica pura e comportamentos com dependências mockadas.
  - `tests/integration/` — persistência em Chroma com diretório temporário por teste.
  - marker `ollama` — testes que dependem do Ollama local.
- CI no GitHub: workflow em `.github/workflows/tests.yml` executando testes sem Ollama em push/PR; testes com Ollama disponíveis manualmente.

#### 5.5 Próximos passos possíveis
- Integração da pesquisa ao `conversar.py` (chat com pesquisa explícita).
- Memória de longa data com síntese semanal.
- Observação multimodal (LLaVA) e interação por voz.

---

## 3. Planejamento Futuro

### Curto prazo
- Integrar pesquisa autônoma ao `conversar.py` (comando explícito de pesquisa).
- Refinar o prompt do detector de curiosidade para melhorar a taxa de detecção.
- Adicionar fallback para o Qwen no diário quando o Llama falhar por CUDA.

### Médio prazo
- Memória de longa data: síntese semanal dos diários, criando capítulos da história do Dante.
- Curadoria automática de memórias: sumarização e esquecimento seletivo.
- Dashboard simples para visualizar a evolução do sistema.

### Longo prazo
- Observação multimodal (imagens, áudio) para enriquecer a experiência do Dante.
- Interação proativa: Dante inicia conversas quando julgar relevante.
- Exploração de correlatos objetivos de autorreflexão em modelos generativos.

---

## 4. Observações Importantes

- A base do ChromaDB (`chroma_db/`) **é persistente entre execuções** do `loop_dante.py` — a memória acumulada não é apagada ao reiniciar.
- `dante.log` e `diario.md` são ignorados pelo Git (dados pessoais/experimentais), mas são gerados localmente a cada execução.
- O diário (`diario.md`) tende a ser o output de maior qualidade introspectiva do sistema, gerado pelo Llama 3.1:8b a cada 5 ciclos do loop.
- Textos extraídos por OCR com menos de 30 caracteres são tratados como ruído: o ciclo ainda gera pensamento/reflexão, mas nada é salvo na memória persistente.
- A pesquisa autônoma (Asas) roda no máximo a cada `CYCLES_ENTRE_PESQUISAS` ciclos (padrão: 8) e só dispara quando o detector de curiosidade identifica uma pergunta genuína no pensamento atual.