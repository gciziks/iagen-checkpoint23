# Classificador de chamados em português com LLM local

Projeto acadêmico reproduzível para ajustar e avaliar localmente o **Qwen2.5-0.5B-Instruct** com LoRA. O modelo recebe uma mensagem de suporte em português e devolve uma categoria: `acesso`, `pagamento`, `entrega`, `cancelamento` ou `outros`.

> O conjunto incluído é sintético, criado para demonstrar o pipeline; não representa clientes reais nem permite afirmar desempenho em produção. Execute o experimento antes de apresentar métricas. Os resultados são gravados em `results/` e não estão pré-preenchidos.

## Ambiente e GPU

Desenvolvido para Windows, Python 3.11 ou 3.12, CUDA e NVIDIA. O script encerra com erro se CUDA não estiver disponível: tanto treino como avaliação rodam na GPU.

1. Instale o [PyTorch com CUDA](https://pytorch.org/get-started/locally/) compatível com seu driver. Confirme no terminal integrado do VS Code: `nvidia-smi` e `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
2. Crie/ative um ambiente virtual e instale: `pip install -r requirements.txt`.
3. `python scripts/check_gpu.py` para registrar a GPU e versões em `results/gpu_info.txt`.
4. `python scripts/train.py` para executar os experimentos; modelos e métricas ficam em `artifacts/` e `results/`.
5. `streamlit run app.py` para abrir o frontend local. Selecione um adaptador treinado em `artifacts/`.

Na demonstração, capture no VS Code o terminal integrado exibindo `nvidia-smi` e `python scripts/check_gpu.py`. O relatório gerado por `check_gpu.py` também registra os dados detectados, mas o print deve ser capturado na máquina que executa o projeto.

## Experimentos

O treino compara learning rates `1e-4` e `2e-4`, épocas `2` e `3`, com LoRA rank 8. Validação compara accuracy, macro-F1 e loss; teste final usa o melhor macro-F1 de validação. Semente fixa, divisão estratificada e dados determinísticos. Consulte `docs/metodologia.md` para protocolo, limitações e interpretação. Não escolha configuração pelo conjunto de teste.

## Estrutura

- `data/`: exemplos sintéticos em JSONL, partições explícitas.
- `scripts/train.py`: fine-tuning e busca de hiperparâmetros, com GPU obrigatória.
- `scripts/evaluate.py`: avaliação independente em teste.
- `scripts/check_gpu.py`: diagnóstico exportável para evidência.
- `app.py`: frontend Streamlit para inserir prompts e inspecionar resposta/categoria.
- `docs/`: metodologia, execução, modelo e roteiro para apresentação.
- `results/`: métricas e previsões reais da execução; veja também a captura da GPU em `Evidencia_GPU.png`.

## GitHub

Versione o código, documentação, dados sintéticos e tabelas CSV em `results/`. Não envie `artifacts/` (adaptadores) nem logs grandes; publique-os como release se solicitado pela disciplina. Antes da entrega, execute o pipeline na sua GPU, revise os CSVs, capture o terminal do VS Code e confirme que a URL remota GitHub está acessível para o docente.

## Licenças e atribuições

Qwen2.5 usa a licença Apache-2.0. Confira o cartão e os termos atuais do modelo antes de redistribuir adaptadores. O código deste repositório está sob MIT (ver `LICENSE`).
