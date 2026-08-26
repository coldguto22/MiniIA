# Contribuindo com o MiniIA

Este projeto segue um fluxo simples para manter qualidade tecnica, privacidade local e historico limpo no GitHub.

## Fluxo de trabalho

1. Crie uma branch por objetivo:
- feature/nome-curto
- fix/nome-curto
- refactor/nome-curto
- chore/nome-curto

2. Mantenha PRs pequenas:
- alvo sugerido: ate 300 linhas liquidas alteradas
- prefira separar grandes mudancas em PRs empilhadas

3. Use commits por intencao unica:
- um commit para refactor estrutural
- um commit para testes
- um commit para mudanca funcional
- docs/chore em commits separados

4. Merge padrao:
- use squash merge para manter historico linear e legivel

## Convencao de commit

Use formato tipo(escopo): descricao objetiva.

Exemplos:
- feat(loop): adiciona cooldown para pesquisa autonoma
- fix(conversar): corrige threshold de contexto
- refactor(memoria): extrai cliente chroma compartilhado
- test(integration): isola persistencia em diretorio temporario
- docs(readme): atualiza instrucoes de testes
- chore(repo): limpa regras de ignore

Regras:
- titulo com ate 72 caracteres
- evitar mensagens genericas como update, refine, enhance
- se houver mudanca de comportamento, inclua teste no mesmo PR

## Dados locais e privacidade

Nao versionar artefatos locais com dados sensiveis:
- chroma_db/
- diario.md e diario_legado.md
- dante.log e dante_legado.log
- backups de memoria em JSON
- arquivos .env reais

Se precisar de variaveis de ambiente, versionar apenas modelo:
- .env.example

## Checklist minimo de PR

1. Problema que esta sendo resolvido
2. O que mudou tecnicamente
3. Como validar localmente
4. Riscos conhecidos
5. Plano de rollback

## Validacao local

Com ambiente ativo, execute:

- pytest -m "not ollama"

Testes com Ollama sao opcionais e exigem configuracao explicita:

- RUN_OLLAMA_TESTS=1
- pytest -m "ollama"
