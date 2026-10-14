# Classificação local de chamados em português com ajuste fino de LLM

Este repositório apresenta um experimento de adaptação supervisionada de um modelo de linguagem para classificar mensagens de suporte em cinco categorias: `acesso`, `pagamento`, `entrega`, `cancelamento` e `outros`. A solução usa o **Qwen2.5-0.5B-Instruct** como modelo-base e ajuste fino eficiente **LoRA**, com treinamento e inferência local em GPU NVIDIA.

## Resumo dos resultados

Foram comparadas quatro configurações, variando taxa de aprendizado e número de épocas. A configuração selecionada pela validação foi taxa `2e-4` e três épocas. No conjunto de teste, composto por dez exemplos sintéticos, obteve accuracy de **0,90** e macro-F1 de **0,893** (nove classificações corretas). Um chamado da classe `outros` foi previsto como `acesso`.

Os resultados comprovam a execução do pipeline, mas não estimam desempenho em atendimento real: o conjunto é pequeno e inteiramente sintético. A interpretação completa, a matriz de confusão e as previsões estão em [`docs/relatorio.md`](docs/relatorio.md) e [`results/`](results/).

## Recursos e metodologia

- Modelo-base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), com aproximadamente 0,5 bilhão de parâmetros.
- Ajuste supervisionado com LoRA aplicado às projeções `q_proj` e `v_proj`.
- Dados balanceados e sintéticos: 50 exemplos de treino, 10 de validação e 10 de teste.
- Avaliação com accuracy, macro-F1, loss de validação e matriz de confusão.
- Interface local em Streamlit para receber um chamado e exibir sua categoria prevista.

## Ambiente experimental

O experimento foi executado em Windows, Python 3.12.6, PyTorch 2.14.1+cu130 e CUDA 13.0, usando uma NVIDIA GeForce RTX 3060 com 12 GB de VRAM. O registro técnico está em [`results/gpu_info.txt`](results/gpu_info.txt), e a captura do terminal PowerShell está em [`Evidencia_GPU.png`](Evidencia_GPU.png).

## Conteúdo do repositório

- [`docs/relatorio.md`](docs/relatorio.md): relatório do problema, método, resultados e limitações.
- [`docs/metodologia.md`](docs/metodologia.md): protocolo experimental e detalhes para reprodução.
- [`docs/apresentacao.md`](docs/apresentacao.md): síntese dos achados e evidência visual.
- `data/`: partições JSONL do conjunto sintético.
- `scripts/`: verificação da GPU, fine-tuning e avaliação.
- `app.py`: frontend local de inferência.
- `results/`: resultados da execução experimental registrada.

## Reprodução do experimento

Com Python 3.12, uma GPU NVIDIA compatível e os drivers instalados, o ambiente é preparado com PyTorch CUDA e os pacotes listados em `requirements.txt`. A execução é feita nesta ordem:

```powershell
python scripts/check_gpu.py
python scripts/train.py
python scripts/evaluate.py
streamlit run app.py
```

O primeiro acesso baixa o modelo-base e o tokenizer do Hugging Face Hub. Treinamento, avaliação e inferência ocorrem localmente; o código interrompe treino e avaliação se CUDA não estiver disponível.

Os adaptadores LoRA produzidos pelo treinamento são salvos localmente em `artifacts/` e não fazem parte deste repositório. Os scripts e os resultados tabulares estão versionados para que o procedimento e os números registrados possam ser examinados e reproduzidos.

## Equipe

- Gabriel Ciziks — RM98215
- Victor Nuzzi — RM98209
- Lucca Sabatini — RM98169
- Cassio Valezzi — RM551059

## Licenças

O código deste projeto está sob MIT (arquivo [`LICENSE`](LICENSE)). O modelo Qwen2.5-0.5B-Instruct é disponibilizado sob Apache-2.0, conforme seu [cartão oficial](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Os exemplos do conjunto são sintéticos e foram criados para este experimento.
