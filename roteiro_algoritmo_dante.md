# Roteiro: Algoritmo do Projeto Dante

## 1. Objetivo da apresentação

Explicar como o Dante funciona do ponto de vista algorítmico, deixando os modelos de linguagem em segundo plano.

O foco está no fluxo de processamento:

```text
percepção
→ detecção de mudança
→ filtragem de ruído
→ recuperação de memória
→ pensamento e reflexão
→ pesquisa eventual
→ persistência
```

O Dante não implementa um algoritmo próprio de treinamento de inteligência artificial. Ele coordena vários algoritmos menores para formar um ciclo contínuo de observação e memória.

---

## 2. Ideia central

O projeto pode ser descrito como:

> Um agente reativo com memória vetorial persistente e reflexão contextual.

Ele observa o ambiente, transforma a observação em texto, procura experiências semelhantes em sua memória e registra novas experiências para uso futuro.

O fluxo geral é:

```text
capturar tela
→ extrair texto com OCR
→ verificar se houve mudança
→ eliminar texto ilegível
→ gerar uma representação vetorial
→ buscar memórias semelhantes
→ produzir pensamento e reflexão
→ salvar a nova experiência
```

---

## 3. Etapa 1 — Percepção da tela

O módulo [`capturador.py`](capturador.py) captura a tela usando `mss`.

Antes do OCR, a imagem passa por um pré-processamento:

1. A tela é capturada.
2. A imagem é redimensionada.
3. A nitidez é aumentada.
4. A imagem é convertida para escala de cinza.
5. O Tesseract extrai o texto em português.

Esse processo transforma uma imagem em uma sequência textual que pode ser processada pelas etapas seguintes.

### Ideia algorítmica

```text
imagem da tela
→ imagem ampliada
→ imagem com maior contraste visual
→ escala de cinza
→ texto extraído
```

O OCR é importante porque o restante do sistema não trabalha diretamente com os pixels. Ele trabalha com o significado aproximado do texto encontrado na tela.

---

## 4. Etapa 2 — Detecção de mudanças

O loop não processa a tela de maneira indiscriminada. Primeiro, ele verifica se houve mudança visual.

Em [`tela_mudou`](loop_dante.py), o programa:

1. Captura a tela atual.
2. Converte seus pixels em bytes.
3. Calcula um hash MD5.
4. Compara o hash atual com o hash anterior.

Se os hashes forem iguais, a tela é considerada estática e o ciclo é ignorado.

Depois do OCR, o sistema também calcula um hash do texto extraído. Isso evita processar novamente uma tela que mudou visualmente, mas continua produzindo exatamente o mesmo texto.

### Duplo filtro de repetição

```text
mudança visual?
  não → ignora o ciclo
  sim → executa OCR

texto OCR mudou?
  não → ignora o ciclo
  sim → continua o processamento
```

Essa estratégia economiza processamento e evita que a memória seja preenchida com cópias da mesma observação.

---

## 5. Etapa 3 — Filtragem de ruído

O OCR pode produzir texto fragmentado ou ilegível. Para reduzir esse problema, o projeto usa a função [`texto_e_ruido`](loop_dante.py).

O texto é considerado ruído quando:

- possui poucos caracteres;
- tem baixa proporção de letras;
- contém poucas palavras reconhecíveis;
- é composto principalmente por símbolos ou fragmentos.

Essa etapa é uma heurística. Ela não compreende profundamente o texto; apenas verifica sinais estatísticos simples de qualidade.

### Exemplo conceitual

```text
"A tela mostra um editor de código Python"
→ provável texto válido

"gr4s | -- W ar o e"
→ provável ruído de OCR
```

Quando o texto é considerado ruído, o sistema ainda pode gerar um pensamento temporário, mas evita salvá-lo na memória persistente.

---

## 6. Etapa 4 — Representação vetorial da memória

Para comparar uma nova observação com experiências anteriores, o sistema transforma o texto em um embedding.

Um embedding é uma representação numérica do conteúdo textual. Textos semanticamente parecidos tendem a ficar próximos no espaço vetorial.

O projeto utiliza o ChromaDB para armazenar:

- o texto original;
- o embedding;
- metadados, como tipo e data;
- um identificador persistente.

A memória não é apenas uma lista cronológica. Ela também pode ser consultada por proximidade semântica.

---

## 7. Etapa 5 — Recuperação por similaridade

Em [`buscar_memorias_relacionadas`](loop_dante.py), o embedding da observação atual é usado para consultar o ChromaDB.

O algoritmo:

1. Gera o embedding do texto atual.
2. Solicita as memórias mais próximas.
3. Recupera até cinco resultados.
4. Examina a distância entre cada resultado e a consulta.
5. Mantém apenas os resultados dentro do limiar de similaridade.

O limite principal do loop é:

```text
TOP_N_MEMORIAS = 5
THRESHOLD_SIMILARIDADE = 0.7
```

No chat, a mesma ideia aparece em [`buscar_contexto_detalhado`](conversar.py), com organização adicional dos metadados e das datas.

### Fluxo da recuperação

```text
observação atual
→ embedding da observação
→ consulta vetorial
→ memórias mais próximas
→ filtragem por distância
→ contexto para a reflexão
```

Esse é o componente de memória semântica do projeto e também caracteriza um RAG simples: recuperar informações antes de gerar uma nova resposta.

---

## 8. Etapa 6 — Pensamento e reflexão

Depois da recuperação, o sistema gera duas camadas de processamento:

### 8.1 Pensamento

O pensamento descreve o que foi observado na tela. Ele funciona como uma interpretação inicial da percepção.

```text
texto OCR
→ descrição curta da situação
```

### 8.2 Reflexão

A reflexão combina o pensamento atual com as memórias recuperadas.

```text
pensamento atual
+ memórias relacionadas
→ reflexão contextual
```

Quando nenhuma memória passa pelo limiar de similaridade, o sistema gera uma reflexão sem contexto anterior.

A diferença algorítmica é importante:

- o pensamento interpreta o presente;
- a reflexão compara o presente com o passado.

---

## 9. Etapa 7 — Curiosidade e pesquisa autônoma

Depois da reflexão, o projeto pode verificar se existe uma dúvida pesquisável.

Essa verificação não acontece em todos os ciclos. O sistema respeita um intervalo mínimo entre pesquisas:

```text
CYCLES_ENTRE_PESQUISAS = 8
```

O detector procura:

- perguntas explícitas;
- dúvidas;
- incertezas;
- desejo de aprender;
- lacunas de conhecimento.

Quando uma curiosidade é encontrada:

1. O sistema extrai uma consulta objetiva.
2. O módulo [`asas.py`](asas.py) pesquisa no DuckDuckGo.
3. Os snippets são formatados.
4. Um aprendizado é produzido a partir dos resultados.
5. O aprendizado é salvo na memória.
6. O aprendizado também é registrado no diário.

### Fluxo da pesquisa

```text
pensamento + reflexão
→ detector de curiosidade
→ consulta de busca
→ resultados da web
→ síntese do aprendizado
→ memória e diário
```

Essa etapa é opcional e depende de conexão com a internet. O restante do ciclo pode funcionar localmente.

---

## 10. Etapa 8 — Persistência e consolidação

Quando a observação não é ruído, o sistema salva:

- a observação e o pensamento;
- a reflexão;
- pesquisas realizadas;
- entradas do diário.

Cada registro é transformado em embedding para que possa ser recuperado futuramente.

O diário é produzido periodicamente, a cada cinco ciclos efetivamente processados:

```text
CYCLES_PARA_DIARIO = 5
```

O contador considera apenas ciclos que passaram pelos filtros de tela e de texto. Assim, ciclos ignorados não aceleram artificialmente a criação do diário.

---

## 11. Algoritmo do chat

O modo de conversa usa uma versão mais direta do mesmo princípio:

```text
pergunta do usuário
→ normalização do texto
→ embedding da pergunta
→ busca de memórias semelhantes
→ montagem do contexto
→ geração da resposta
→ atualização do histórico
→ salvamento opcional
```

Antes de salvar uma interação, o sistema aplica filtros:

- respostas muito curtas não são salvas;
- recusas genéricas não são salvas;
- duplicatas semânticas não são salvas;
- respostas válidas são armazenadas com embedding.

Isso evita diluir a memória com saudações, respostas vazias ou repetições.

---

## 12. Camada de valência e homeostase

O diretório [`dante/core`](dante/core) contém uma camada algorítmica mais recente.

Ela introduz estados internos calculáveis:

- novidade;
- coerência;
- agência;
- prazer/desconforto;
- energia;
- tédio;
- estagnação;
- impulso de conexão.

Em [`valence.py`](dante/core/valence.py), a novidade é calculada a partir da similaridade com memórias existentes:

```text
novidade = 1 - similaridade máxima
```

A coerência é estimada pela similaridade média entre o pensamento e as memórias recuperadas.

Em [`homeostasis.py`](dante/core/homeostasis.py):

- a energia diminui com o passar dos ciclos;
- o tédio aumenta quando não há novidade;
- descobertas e interações podem recuperar energia;
- o sistema pode indicar que deveria descansar ou mudar de atenção.

Essa camada representa uma evolução do algoritmo, porque tenta transformar a memória e a repetição em sinais que influenciam o comportamento.

---

## 13. O que está efetivamente integrado

É importante separar implementação atual de arquitetura planejada.

### Já implementado no fluxo principal

- captura de tela;
- OCR;
- detecção de mudança visual;
- detecção de repetição textual;
- filtragem heurística de ruído;
- embeddings;
- recuperação vetorial no ChromaDB;
- pensamento;
- reflexão contextual;
- pesquisa periódica;
- diário;
- memória persistente.

### Implementado, mas parcialmente desacoplado

- valência;
- homeostase;
- auto-modelo;
- valores emergentes;
- impulso de relação;
- lógica de silêncio.

Esses componentes possuem funções e estruturas próprias, mas ainda não controlam completamente todos os ciclos de [`loop_dante.py`](loop_dante.py).

Por isso, a descrição mais precisa do projeto atual é:

> O Dante é principalmente um pipeline reativo de observação e memória vetorial, com uma camada experimental de estados internos em processo de integração.

---

## 14. Pontos fortes do algoritmo

1. **Evita processamento desnecessário** com hashes de imagem e texto.
2. **Reduz ruído de OCR** antes de persistir dados.
3. **Recupera contexto por significado**, não apenas por palavras exatas.
4. **Mantém memória entre execuções** usando armazenamento persistente.
5. **Separa interpretação imediata de reflexão contextual**.
6. **Controla a frequência de pesquisas e do diário**.
7. **Evita duplicatas e respostas de baixa qualidade no chat**.
8. **Permite evolução para comportamentos baseados em estado interno**.

---

## 15. Limitações algorítmicas atuais

1. Os limiares são definidos manualmente e não aprendidos com dados.
2. A qualidade da memória depende bastante do OCR.
3. A busca vetorial considera principalmente proximidade, não causalidade ou ordem temporal.
4. O sistema pode recuperar memórias semanticamente parecidas, mas irrelevantes para a situação.
5. O detector de curiosidade ainda depende de uma chamada de geração e de parsing textual.
6. A pesquisa autônoma ainda é limitada por regras fixas de intervalo.
7. A camada de valência e homeostase ainda não governa integralmente o loop principal.
8. O comportamento é reativo: ele responde a mudanças externas, em vez de escolher autonomamente objetivos complexos.

---

## 16. Conclusão para apresentação

O algoritmo do Dante não está em um único arquivo ou em uma única fórmula. Ele é um ciclo composto por vários mecanismos:

```text
percepção visual
→ OCR
→ filtragem
→ embedding
→ busca vetorial
→ reflexão contextual
→ pesquisa opcional
→ persistência
```

O elemento central é a combinação entre **memória persistente** e **recuperação por similaridade**. A cada nova observação, o sistema tenta relacionar o presente com experiências passadas e registra o resultado para consultas futuras.

A camada de valência, homeostase e auto-modelo amplia esse mecanismo, mas ainda está em uma fase experimental de integração. Portanto, o projeto atual deve ser entendido como uma base reativa de memória e reflexão, caminhando para um sistema orientado por estados internos.
