# Classificação de chamados de suporte em português com ajuste fino de uma LLM local

## Resumo

Este trabalho investiga a adaptação de um modelo de linguagem local para encaminhar chamados de suporte em português a cinco categorias: acesso, pagamento, entrega, cancelamento e outros. Foi utilizado o Qwen2.5-0.5B-Instruct com ajuste fino supervisionado por LoRA, executado em uma NVIDIA GeForce RTX 3060. Quatro combinações de taxa de aprendizado e número de épocas foram comparadas em validação. A configuração selecionada obteve accuracy de 0,90 e macro-F1 de 0,893 no conjunto de teste sintético. O resultado demonstra a viabilidade técnica do fluxo na GPU disponível, mas não deve ser interpretado como evidência de desempenho em uso real devido ao tamanho reduzido e à natureza artificial dos dados.

**Palavras-chave:** modelo de linguagem; fine-tuning; LoRA; classificação de texto; GPU; português.

## 1. Introdução

Mensagens de suporte frequentemente precisam ser encaminhadas a equipes diferentes. A classificação automática pode apoiar essa triagem ao atribuir uma fila temática a cada mensagem. Modelos de linguagem instrucionais podem ser adaptados para tarefas específicas por meio de exemplos rotulados. Para que o experimento caiba em hardware pessoal, este projeto utiliza LoRA, que ajusta um conjunto reduzido de parâmetros adicionais sem atualizar todos os pesos do modelo-base.

O objetivo foi construir e avaliar um protótipo reproduzível de classificação de chamados em português, verificar que o treinamento e a inferência utilizam a GPU local e oferecer uma interface simples para testar entradas. O trabalho adapta um modelo pré-treinado; não treina uma LLM do zero.

## 2. Tarefa e categorias

Cada entrada é uma descrição curta de um chamado. A saída é um único rótulo dentre:

| Categoria | Tipo de solicitação |
|---|---|
| `acesso` | Login, senha, autenticação ou bloqueio de conta |
| `pagamento` | Cobrança, boleto, cartão, Pix ou reembolso |
| `entrega` | Envio, rastreamento, atraso ou recebimento de pedido |
| `cancelamento` | Encerramento ou desistência de pedido ou assinatura |
| `outros` | Assuntos que não se enquadram nas quatro filas anteriores |

## 3. Modelo e implementação

O modelo-base foi `Qwen/Qwen2.5-0.5B-Instruct`, com aproximadamente 0,5 bilhão de parâmetros. O modelo gera o nome da categoria em resposta a um prompt que restringe as saídas às cinco classes. O ajuste supervisionado emprega Low-Rank Adaptation (LoRA) nas projeções de atenção `q_proj` e `v_proj`, com rank 8, alpha 16 e dropout 0,05. Os parâmetros originais permanecem congelados. O carregamento em FP16 reduz a memória necessária.

O sistema inclui scripts para diagnóstico da GPU, treino e avaliação, além de um frontend Streamlit. A aplicação recebe uma mensagem, executa geração determinística local na GPU e mostra a categoria prevista. A inferência não utiliza uma API de modelo remota; os pesos do modelo-base são baixados do Hugging Face Hub na primeira execução.

## 4. Dados

O conjunto foi elaborado para fins didáticos e contém apenas exemplos sintéticos. Há 50 exemplos de treino, 10 de validação e 10 de teste, com distribuição balanceada: 10 exemplos de treino e dois exemplos em cada uma das outras partições por categoria. Os arquivos estão em `data/` no formato JSONL.

O tamanho foi escolhido para demonstrar o funcionamento ponta a ponta, não para representar a variedade de solicitações reais. Em particular, duas amostras por categoria no teste tornam as métricas muito sensíveis a uma única previsão. Os resultados não devem ser generalizados para clientes, organizações ou domínios não representados pelos exemplos.

## 5. Protocolo experimental

Foi utilizada semente aleatória 23. Rank LoRA, alpha, dropout, tamanho de lote, acumulação de gradiente e comprimento máximo permaneceram fixos. A busca comparou duas taxas de aprendizado (`1e-4` e `2e-4`) e duas durações (duas e três épocas). A validação foi realizada ao final de cada época.

Macro-F1 de validação foi o critério primário de seleção; empates foram resolvidos por accuracy e, em seguida, menor loss de validação. Depois de selecionar a configuração, o conjunto de teste foi usado uma única vez para produzir a avaliação final. Esse protocolo separa a escolha de hiperparâmetros da estimativa final no conjunto disponível.

As métricas reportadas são accuracy, macro-F1, loss de validação e métricas por classe. Macro-F1 dá o mesmo peso a cada categoria, enquanto a matriz de confusão mostra a distribuição das classes reais e previstas.

## 6. Ambiente computacional

| Componente | Configuração observada |
|---|---|
| Sistema | Windows |
| Python | 3.12.6 |
| PyTorch | 2.14.1+cu130 |
| Runtime CUDA do PyTorch | 13.0 |
| GPU | NVIDIA GeForce RTX 3060 |
| Memória de GPU | 12,00 GiB |

O diagnóstico confirma que CUDA estava disponível no ambiente. A captura do terminal com `nvidia-smi` e o relatório do PyTorch encontram-se em [`Evidencia_GPU.png`](../Evidencia_GPU.png) e [`results/gpu_info.txt`](../results/gpu_info.txt).

## 7. Resultados

### 7.1 Comparação de hiperparâmetros

| Taxa de aprendizado | Épocas | Loss de validação | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| `1e-4` | 2 | 3,732 | 0,50 | 0,381 |
| `1e-4` | 3 | 1,777 | 0,50 | 0,381 |
| `2e-4` | 2 | 0,765 | 0,90 | 0,893 |
| `2e-4` | 3 | 0,158 | 0,90 | 0,893 |

As duas configurações com taxa `2e-4` obtiveram os maiores valores de accuracy e macro-F1 na validação. O desempate favoreceu três épocas devido à menor loss. Assim, a configuração selecionada foi `2e-4`, três épocas, rank LoRA 8.

### 7.2 Teste final

No teste de dez exemplos, a configuração selecionada classificou corretamente nove, resultando em accuracy 0,90 e macro-F1 0,893. A matriz de confusão abaixo usa a ordem `acesso`, `pagamento`, `entrega`, `cancelamento`, `outros`, tanto nas linhas (classe real) como nas colunas (classe prevista):

| Real \\ Prevista | acesso | pagamento | entrega | cancelamento | outros |
|---|---:|---:|---:|---:|---:|
| acesso | 2 | 0 | 0 | 0 | 0 |
| pagamento | 0 | 2 | 0 | 0 | 0 |
| entrega | 0 | 0 | 2 | 0 | 0 |
| cancelamento | 0 | 0 | 0 | 2 | 0 |
| outros | 1 | 0 | 0 | 0 | 1 |

O único erro ocorreu em uma mensagem da classe `outros`, prevista como `acesso`. As previsões individuais e as métricas completas estão em `results/test_predictions.csv` e `results/test_metrics.json`.

## 8. Discussão e limitações

O experimento confirmou o funcionamento integrado de fine-tuning LoRA, avaliação e inferência local em GPU. A diferença entre as duas taxas de aprendizado nas métricas de validação indica que a escolha de hiperparâmetros influencia o comportamento mesmo nesse conjunto pequeno. O frontend permite inspecionar respostas para entradas digitadas, mas exemplos de demonstração não substituem avaliação controlada.

Há limitações importantes: conjunto sintético e pequeno; somente duas observações de teste por categoria; ausência de avaliação por múltiplos anotadores; ausência de comparação quantitativa com o modelo-base sem ajuste; e ausência de testes em distribuição real. A pontuação de 90% no teste corresponde a nove exemplos específicos e pode variar muito com novas amostras. Portanto, não é evidência de que o sistema esteja pronto para automatizar triagem de chamados reais.

Trabalhos futuros incluem ampliar o conjunto com dados representativos e tratados de acordo com privacidade e consentimento, estabelecer diretrizes de anotação, usar partições mais robustas, medir variabilidade em diferentes sementes, comparar com baselines, analisar erros com avaliadores humanos e medir latência e uso de memória.

## 9. Conclusão

Foi implementado um protótipo local que adapta o Qwen2.5-0.5B-Instruct por LoRA para classificar chamados sintéticos em cinco categorias. Quatro configurações foram comparadas, e a configuração `2e-4` por três épocas foi selecionada na validação. No teste sintético, alcançou accuracy de 0,90 e macro-F1 de 0,893. O resultado atende ao objetivo de demonstrar o pipeline em GPU e uma interface funcional, mantendo explícito que dados e métricas são insuficientes para afirmar generalização a um cenário de suporte real.

## Referências e licenças

- Qwen Team. [Qwen2.5-0.5B-Instruct: cartão do modelo](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Licença indicada pelo repositório do modelo: Apache-2.0.
- Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR, 2022.
- O código desenvolvido para este projeto está sob licença MIT. Os dados sintéticos foram criados para este experimento.
