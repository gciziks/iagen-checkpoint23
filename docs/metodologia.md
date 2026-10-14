# Metodologia experimental

## Pergunta e tarefa

É possível adaptar um modelo instrucional pequeno para direcionar solicitações de suporte em português a uma de cinco filas? Entrada: descrição do chamado. Saída: um dos rótulos fechados `acesso`, `pagamento`, `entrega`, `cancelamento`, `outros`.

## Modelo e ajuste

Base: `Qwen/Qwen2.5-0.5B-Instruct`, modelo aberto de aproximadamente meio bilhão de parâmetros, carregado localmente. O treino supervisionado ensina a gerar o rótulo a seguir ao prompt. LoRA ajusta matrizes de projeção de atenção `q_proj`/`v_proj` (rank 8, alpha 16, dropout 0,05); pesos da base ficam congelados. Precisão FP16 reduz memória. A implementação mede avaliação e inferência em GPU CUDA e interrompe se CUDA não estiver disponível.

## Dados e divisão

O repositório inclui 50 exemplos sintéticos de treino (10 por classe), 10 de validação (2 por classe) e 10 de teste (2 por classe). Foram escritos para serem didáticos, com linguagem e exemplos deliberadamente pequenos. Os arquivos são partições fixas, sem sobreposição de frases. A amostra é pequena e artificial: métricas servem apenas para validar o experimento de aula; não estimam qualidade em chamados reais. Não inclua informação pessoal.

## Protocolo

Semente aleatória 23. Configurações: learning rate `1e-4` ou `2e-4`; épocas `2` ou `3`; rank LoRA fixo em 8. Tamanho por dispositivo 2, acumulação de gradiente 2, sequência máxima 192, avaliação ao fim de cada época. O melhor checkpoint é escolhido por macro-F1 de validação; empate é resolvido por accuracy e depois loss de validação. Teste é avaliado uma única vez após a seleção. Não use o teste para ajustar hiperparâmetros.

## Métricas

- Accuracy: fração de exemplos classificados corretamente.
- Macro-F1: média do F1 por classe, pesando cada fila igualmente; adequada para examinar desempenho entre classes.
- Loss de validação: loss média de próximo token na resposta rotulada; menor sugere melhor ajuste, mas não substitui métricas de classe.
- Matriz de confusão e precision/recall/F1 por classe, exportados em `results/test_metrics.json`.

Com apenas dois casos por classe no teste, cada erro muda bastante a métrica. Inclua essa limitação ao discutir resultados e use dados adicionais, revisados e com licença apropriada se fizer afirmações mais amplas.

## Reprodutibilidade

Use as versões instaladas no `requirements.txt`, anote a versão efetivamente instalada e o driver, execute `python scripts/check_gpu.py`, `python scripts/train.py` e `python scripts/evaluate.py`; preserve `results/experiments.csv`, `results/test_predictions.csv` e `results/test_metrics.json`. Mudanças em versão de driver/CUDA ou da biblioteca podem alterar o resultado. Base e tokenizer são baixados do Hub Hugging Face na primeira execução; o fine-tuning e a avaliação são locais, sem serviços de inferência remotos.
