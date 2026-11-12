# Roteiro de apresentação — Dante-Mamba

## 1. Abertura

O Dante é um sistema experimental de IA local que observa a tela por OCR,
gera pensamentos, recupera memórias no ChromaDB e produz reflexões e diário.
O objetivo acadêmico do Dante-Mamba não é provar consciência artificial, mas
avaliar se a continuidade de estado entre ciclos melhora a continuidade
funcional, a coerência e a recuperação após reinício.

O repositório da entrega é
<https://github.com/coldguto22/MiniIA>. A disciplina informada nos relatórios
é DSM, e os integrantes são Jorge Rizzini de Castro Filho, Laísa Eugênio
Severiano e Otávio Augusto de Lima Beato.

## 2. Onde o Mamba entra

O Mamba foi colocado como hipótese de **System 1**, o modelo rápido que recebe
a observação atual. O pipeline experimental é:

```text
OCR -> System 1 (Qwen ou Mamba) -> pensamento
                                -> estado/checkpoint experimental
ChromaDB + pensamento -> System 2 (Llama) -> reflexão/diário
```

Na configuração oficial atual, o projeto usa:

- `qwen2.5:3b` como System 1 de produção;
- `llama3.1:8b` como System 2;
- `nomic-embed-text` para embeddings;
- `hf.co/bartowski/mamba-2.8b-hf-GGUF:Q4_K_M` como Mamba experimental,
  acionado diretamente pelo adapter.

Essa separação é intencional: o Mamba está instalado e consegue gerar, mas a
integração via Ollama fornece um `context` serializável, não garante acesso ao
estado recorrente interno do Mamba. Por isso o resultado é chamado de
`ollama-context` e não de estado interno comprovado.

Arquivos para mostrar:

- [config/models.yaml](config/models.yaml): modelos oficiais de produção.
- [dante/models/mamba_model.py](dante/models/mamba_model.py): adapter Mamba,
  fallback e geração com estado.
- [dante/models/stateful.py](dante/models/stateful.py): sessão, reset,
  checkpoint e restauração.
- [dante/models/checkpoint.py](dante/models/checkpoint.py): persistência
  versionada e protegida por checksum.
- [MIGRATION_MAMBA.md](MIGRATION_MAMBA.md): limitações e decisões da migração.

## 3. Como demonstrar manualmente

### Teste rápido pelo Ollama

```powershell
ollama list
ollama show hf.co/bartowski/mamba-2.8b-hf-GGUF:Q4_K_M
```

O ambiente validado em 08/11/2026 informou:

```text
architecture: mamba
parameters: 2.77B
context length: 1048576
quantization: Q4_K_M
capability: completion
```

### Teste automatizado do Mamba real

No PowerShell, a partir da raiz do projeto:

```powershell
$env:PYTHONPATH="C:\Users\Guto\Documents\MiniIA"
$env:RUN_OLLAMA_TESTS="1"
pytest tests/integration/test_ollama_mamba_optional.py -m ollama -q
```

Resultado obtido em 08/11/2026:

```text
1 passed
```

Esse teste confirma duas gerações na mesma `ModelSession`, a existência do
checkpoint, o tipo `ollama-context` e um payload de estado não vazio.

### Demonstração conceitual de sessão

O fluxo usado pelo teste é equivalente a:

```python
model = MambaModel(
    model_name="hf.co/bartowski/mamba-2.8b-hf-GGUF:Q4_K_M",
    client=ollama,
)
session = ModelSession(model, "sessao-apresentacao")

await session.generate("primeira observação")
await session.generate("segunda observação")
session.save("mamba-checkpoint.json")
```

Depois, uma nova sessão pode executar `restore("mamba-checkpoint.json")`.
Também existe `session.reset()` para comparar uma trajetória com estado
descartado.

## 4. Testes realizados

### Testes sem Ollama

Comando central executado:

```powershell
$env:PYTHONPATH="C:\Users\Guto\Documents\MiniIA"
pytest tests/unit/test_asas.py tests/unit/test_checkpoint.py `
  tests/unit/test_cognition_adapters.py tests/unit/test_config_loader.py `
  tests/unit/test_loop_dante_utils.py tests/unit/test_mamba_model.py `
  tests/unit/test_ollama_model.py tests/unit/test_stateful_session.py -m unit -q
```

Resultado:

```text
23 passed
```

Também foram executados os testes centrais de configuração, adapters, Mamba e
sessão:

```text
11 passed
```

### Suite completa sem Ollama

Comando:

```powershell
pytest -m "not ollama"
```

Resultado: a coleta foi interrompida por uma falha local do ChromaDB/Rust ao
importar `conversar.py`:

```text
pyo3_runtime.PanicException:
range start index 10 out of range for slice of length 9
```

Esse erro ocorre antes da execução dos testes afetados e está relacionado à
base persistente local/tenant do ChromaDB. Os testes unitários que não
dependem desse banco passaram. Portanto, a apresentação deve dizer
“23 testes unitários isolados passaram; a suite completa está bloqueada por
falha local de inicialização do ChromaDB”, e não afirmar que a suite completa
passou.

### Testes com Ollama

Comando:

```powershell
$env:RUN_OLLAMA_TESTS="1"
pytest tests/integration/test_ollama_mamba_optional.py `
  tests/integration/test_ollama_llama_optional.py -m ollama -q
```

Resultado:

```text
2 passed, 1 failed
```

O Mamba passou. O teste do `llama3.1:8b` falhou porque o servidor Ollama
encerrou com erro de CUDA/PTX JIT:

```text
CUDA error: a PTX JIT compilation failed
```

Isso é uma limitação do runtime/GPU neste ambiente, não evidência de que o
modelo Llama esteja ausente. Deve ser apresentado como uma falha de
infraestrutura conhecida.

## 5. Benchmark exploratório de 08/11/2026

Foi feita uma comparação direta com cinco prompts curtos, usando Qwen e Mamba
via Ollama. Os tempos são tempo de chamada do modelo, incluindo carregamento
quando aplicável:

| Modelo | Execuções válidas | Média | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Qwen2.5:3b | 5/5 | 1,133 s | 0,238 s | 4,178 s |
| Mamba 2.8B Q4_K_M | 5/5 | 1,717 s | 0,680 s | 3,305 s |

Interpretação cuidadosa:

- o Mamba foi executável e apresentou checkpoint;
- nessa amostra pequena, o Qwen foi mais rápido em média;
- o Mamba apresentou repetições e respostas semanticamente fracas em alguns
  prompts;
- isso é evidência preliminar, não um benchmark estatístico ou conclusão
  sobre superioridade de arquitetura.

### Teste de continuidade observado

O Mamba recebeu primeiro uma palavra-chave e depois uma pergunta sobre ela.
O checkpoint foi criado corretamente, mas a resposta posterior repetiu a
pergunta e não recuperou de forma confiável a palavra-chave. Essa diferença
entre “estado serializado existe” e “memória semântica foi recuperada” é um
resultado central:

> Persistir um contexto não prova, sozinho, continuidade funcional útil.

Os benchmarks históricos em [benchmarks/results](benchmarks/results) foram
gerados em 01/10 e 06/11. Ambos registraram tempos e throughput iguais a zero,
porque os modelos não estavam disponíveis para execução real naquele momento.
Eles devem ser tratados como histórico de infraestrutura, não como evidência
comparativa de desempenho.

## 6. Linha do tempo das decisões

As datas abaixo combinam o histórico Git e a documentação versionada:

| Data | Decisão ou evento |
|---|---|
| Fases iniciais do projeto | O Dante foi estruturado com captura/OCR, Ollama e ChromaDB persistente. O Qwen2.5:3b foi escolhido para reduzir problemas de CUDA em máquinas com pouca VRAM. |
| Fase 2 do roadmap | Foi adicionado RAG com ChromaDB, embeddings `nomic-embed-text`, métrica cosseno e memórias fundacionais. |
| Fase 3 do roadmap | O loop contínuo, filtros anti-ruído, truncamento de contexto e diário seguro foram consolidados. |
| Fase 4 do roadmap | Foi adicionada a pesquisa autônoma via DDGS/DuckDuckGo, com prevenção de consultas repetidas. |
| 05/11/2026 | Commits `c91c09a`/`b41db2d` experimentaram Mamba 2.8B como modelo configurado. |
| 05/11/2026 | O commit `75035fc` registrou validação de estado persistente com Ollama. |
| 05/11/2026 | Correções posteriores registraram problemas de estabilidade do chat, ChromaDB e limites de saída. |
| 06/11/2026 | O benchmark histórico `compare_20261106.json` foi gerado, mas sem inferências reais válidas. |
| 08/11/2026 | A configuração oficial voltou a Qwen/Llama, preservando Mamba como adapter experimental, porque o Mamba não deveria substituir a produção sem validação suficiente. |
| 08/11/2026 | O modelo Mamba instalado foi confirmado funcional para geração e checkpoint. O teste de continuidade semântica ainda mostrou repetição. |

## 7. Decisões éticas e técnicas

O grupo adotou as seguintes cautelas:

1. **Não chamar comportamento textual de consciência.** Resultados são
   descritos como continuidade funcional observada.
2. **Separar memória externa de estado do modelo.** ChromaDB é memória
   episódica recuperável; `ollama-context` é continuação serializada de prompt.
3. **Isolar sessões.** Checkpoints possuem `session_id`, modelo, backend e
   integridade, evitando vazamento entre trajetórias.
4. **Manter fallback e baseline.** Qwen/Llama continuam disponíveis para
   comparação e operação estável.
5. **Não usar dados pessoais sem consentimento.** O README registra OCR local
   consentido, entradas manuais e DuckDuckGo/DDGS como fonte externa de
   pesquisa, sem dataset pessoal versionado.
6. **Não fazer treinamento de LLM nesta etapa.** Até 11/11, é mais
   reproduzível avaliar modelos existentes com cenários controlados do que
   treinar ou ajustar pesos sem tempo, hardware e protocolo suficientes.

## 8. Fechamento sugerido

“Nossa entrega não afirma que o Dante é consciente. Ela apresenta uma
arquitetura experimental reproduzível, com baseline, memória vetorial,
adapter Mamba, sessões isoladas e checkpoints. O resultado atual mostra que o
Mamba funciona no Ollama e pode persistir um contexto, mas também mostra que
persistência técnica não garante recuperação semântica nem superioridade sobre
o Qwen. Essas limitações fazem parte do resultado científico e orientam os
próximos testes.”

