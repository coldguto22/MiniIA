# Migração para Mamba como System 1

## Motivação

O Dante usa dois modelos em pipeline: um modelo rápido para observação imediata e um modelo mais lento para reflexão e diário. A arquitetura atual usa `qwen2.5:3b` como System 1 e `llama3.1:8b` como System 2. A integração de um modelo Mamba expande a arquitetura para um estado recorrente entre tokens, reduzindo algumas limitações de modelos puramente transformadores em cenários de individualidade persistente.

Este documento registra a adoção da Camada 1 da estratégia proposta: substituir o System 1 por um Mamba via Ollama, mantendo o comportamento do restante do sistema e preservando fallback para Qwen quando o Mamba não estiver disponível.

## Passos de migração

1. Configurar a camada de modelos em `config/models.yaml`.
2. Estabelecer a camada abstrata em `dante/models/`.
3. Validar a integração em testes unitários e de integração.
4. Usar `OllamaModel` e `MambaModel` sem alterar o restante do loop.
5. Manter o formato do projeto estável e registrar limitações conhecidas.

## Tabela comparativa

| Modelo | Papel | Latência | Throughput | Observações |
| --- | --- | --- | --- | --- |
| Qwen2.5:3b | System 1 de fallback | Média | Boa | Estável, bem conhecido e amplamente disponível |
| Mamba | System 1 principal | Variável | Boa | Requer suporte via Ollama e pode ter comportamento diferente em prompts longos |
| Llama 3.1:8b | System 2 | Mais alta | Menor | Usado para reflexão profunda e diário |

## Limitações conhecidas

- A integração via Ollama não expõe diretamente o estado interno do Mamba. Isso fica para a Camada 2, que poderá manipular o estado recorrente explicitamente.
- Modelos baseados em GGUF podem não estar disponíveis em todos os ambientes, especialmente em ambientes sem rede ou sem espaço suficiente para download.
- O fallback para Qwen é obrigatório e garante que o pipeline continue operando, mesmo quando o Mamba falha.
- A instalação direta do modelo `hf.co/mradermacher/mamba-2.8b-slimpj-hf-GGUF` via `ollama pull` falhou neste ambiente por um redirecionamento bloqueado para um host diferente do Ollama.
- O modelo local `mamba-dante:latest` foi validado e não carrega corretamente: o servidor do Ollama encerra com erro de `check_tensor_dims` / shape mismatch, então ele não pode ser usado como System 1 válido neste ambiente.
- Foi validado também que o catálogo do Ollama não disponibiliza um pacote Mamba funcional por `ollama pull` nesta instalação; `ollama pull mamba` e `ollama pull jamba` retornaram `file does not exist`, enquanto `qwen2.5:3b` funciona normalmente.
- Em consequência, a base de trabalho foi revertida para Qwen como System 1 em produção, mantendo a arquitetura Mamba pronta para uso quando um pacote compatível estiver disponível.

## Estado atual da migração

- O suporte arquitetural para Mamba foi preservado via `dante/models/mamba_model.py` e `dante/config.py`.
- A implementação permanece compatível e pronta para uma troca futura de modelo sem mexer no loop principal.
- A configuração não deve apontar para o Mamba quebrado em ambiente de produção; o uso real do modelo precisa de um pacote válido do Ollama ou de um arquivo GGUF compatível com a versão do runtime do Ollama.

## Próximos passos

- Camada 2: controlar e persistir o estado recorrente do Mamba.
- Camada 3: combinar Mamba com atenção e um pipeline de passagem de estado do System 1 para o System 2.
- Explorar benchmarks comparativos em `benchmarks/compare_system1.py` para quantificar ganhos reais de latência e qualidade.

## Future Work

- Investigar como o Mamba se comporta em prompts extensos e em raciocínio multi-etapa.
- Medir o efeito da ausência de estado interno em cenários de memória persistente e observação contínua.
- Avaliar hibridização com mecanismos de atenção para melhorar coerência em longas sessões.
