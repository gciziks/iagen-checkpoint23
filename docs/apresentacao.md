# Síntese dos resultados experimentais

O trabalho adaptou o modelo Qwen2.5-0.5B-Instruct para classificar chamados de suporte em português em cinco filas: acesso, pagamento, entrega, cancelamento e outros. O ajuste supervisionado foi realizado localmente com LoRA na NVIDIA GeForce RTX 3060 de 12 GB. A evidência do dispositivo e a confirmação de CUDA estão em [`Evidencia_GPU.png`](../Evidencia_GPU.png) e [`results/gpu_info.txt`](../results/gpu_info.txt).

## Comparação na validação

| Taxa de aprendizado | Épocas | Loss de validação | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| `1e-4` | 2 | 3,732 | 0,50 | 0,381 |
| `1e-4` | 3 | 1,777 | 0,50 | 0,381 |
| `2e-4` | 2 | 0,765 | 0,90 | 0,893 |
| `2e-4` | 3 | 0,158 | 0,90 | 0,893 |

O critério primário foi macro-F1 de validação. As duas configurações com taxa `2e-4` empataram em macro-F1 e accuracy; a execução selecionou três épocas, que obteve menor loss de validação.

## Avaliação final

No teste, a configuração selecionada alcançou accuracy **0,90** e macro-F1 **0,893**, com nove acertos em dez exemplos. As classes acesso, pagamento, entrega e cancelamento tiveram dois acertos cada. Em `outros`, um caso foi classificado corretamente e o outro foi previsto como `acesso`. Os dados completos estão em [`test_metrics.json`](../results/test_metrics.json) e [`test_predictions.csv`](../results/test_predictions.csv).

## Interpretação

O resultado demonstra que o pipeline de fine-tuning, avaliação e inferência local foi executado com sucesso na GPU. Contudo, cada classe tem somente dois exemplos no teste e os textos são sintéticos. Por isso, a pontuação não fornece uma estimativa confiável de desempenho em dados reais. Uma avaliação aplicada exigiria um conjunto maior e representativo, critérios de anotação explícitos e análise de erros com especialistas.

## Evidência computacional

![Terminal PowerShell mostrando nvidia-smi e a verificação de CUDA pelo PyTorch](../Evidencia_GPU.png)
