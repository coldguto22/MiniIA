# Benchmarks e validação dos modelos

Registro dos experimentos usados para escolher o modelo do System 1 e validar a integração no MiniIA/Dante. Os arquivos JSON vinculados guardam respostas completas e métricas por chamada; os resumos Markdown facilitam a leitura.

## Configuração de produção

- System 1: `llama3.2:latest`.
- System 2: `llama3.1:8b` (mantido como estabelecido).
- Qwen2.5:3b: comparador nos benchmarks e fallback somente quando configurado explicitamente; não é o modelo ativo do System 1.

## Comparação geral

- Execução única de 20 tarefas; métricas de latência e contagem de tokens retornadas pelo Ollama.
- O juiz de qualidade foi o próprio Llama 3.2, com respostas anonimizadas e ordem alternada. A nota é subjetiva e pode favorecer o próprio modelo; deve ser lida junto às respostas brutas.
- Qwen teve nota média ligeiramente maior (4,56 contra 4,46) e p95 menor (5.803 ms contra 8.033 ms). Llama teve menor latência média (3.693 ms contra 4.744 ms). Os resultados não indicam vencedor absoluto para todos os critérios.
- Artefatos: [dados JSON](benchmarks/results/qwen_llama_20261112_152411.json) · [resumo](benchmarks/results/qwen_llama_20261112_152411.md).

## Benchmark com OCR real

- Base: captura de tela fornecida pelo usuário; Tesseract 5.5.0, idiomas `por+eng`, PSM 3, recorte `(250,100,1420,850)`.
- Foram avaliadas 9 tarefas geradas a partir do texto reconhecido, com uma chamada por modelo/tarefa.
- Llama obteve nota média 4,58 contra 3,81 e menor latência média (3.045 ms contra 4.117 ms). Qwen teve p95 ligeiramente menor (5.108 ms contra 5.533 ms) e gerou menos tokens de saída. A pontuação foi atribuída pelo próprio Llama 3.2, então não deve ser tratada isoladamente como prova de superioridade.
- Artefatos: [dados JSON](benchmarks/results/qwen_llama_ocr_20261112_172625.json) · [resumo](benchmarks/results/qwen_llama_ocr_20261112_172625.md).

## Testes registrados

Executados após a revisão dos adapters legados, sem rodar o loop contínuo do Dante:

| Comando / modo | Resultado |
|---|---:|
| `RUN_OLLAMA_TESTS=0 python -m pytest -q` (ambiente Windows: `$env:RUN_OLLAMA_TESTS='0'`) | 69 aprovados, 5 ignorados, 1,41 s |
| `RUN_OLLAMA_TESTS=1 python -m pytest -q` (Ollama local habilitado) | 74 aprovados, 27,87 s |

Os 5 testes ignorados no modo offline são integrações opcionais que exigem Ollama. A suíte habilitada foi aprovada com os modelos locais disponíveis. O teste de loop local do Dante fica para execução manual do usuário.

## Reprodução

```powershell
python benchmarks/compare_models.py
python benchmarks/compare_models_ocr.py --image C:\caminho\captura.png --lang por+eng --psm 3 --crop 250 100 1420 850
```

O argumento `--crop` é opcional; sem ele, usa-se a imagem inteira. Ambos os scripts salvam JSON e resumo Markdown em `benchmarks/results/`.
