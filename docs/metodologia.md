# Metodologia experimental

## Objetivo e definição da tarefa

O experimento avalia a adaptação de um modelo de linguagem de pequeno porte para encaminhar chamados de suporte escritos em português. Dada uma mensagem, o modelo deve produzir exatamente um dos rótulos `acesso`, `pagamento`, `entrega`, `cancelamento` ou `outros`. A tarefa é formulada como geração supervisionada do nome da categoria após uma instrução de classificação.

## Modelo-base e ajuste fino

Foi utilizado `Qwen/Qwen2.5-0.5B-Instruct`, modelo instrucional com aproximadamente 0,5 bilhão de parâmetros. O fine-tuning usa Low-Rank Adaptation (LoRA): os pesos originais permanecem congelados e matrizes de baixa dimensão são adicionadas às projeções de atenção `q_proj` e `v_proj`. A configuração fixa é rank 8, alpha 16 e dropout 0,05. Os pesos são carregados em FP16. Esse arranjo reduz os parâmetros treináveis e viabiliza o experimento numa GPU de 12 GB.

O prompt instrui o modelo a responder somente com uma categoria. A loss de treino é calculada sobre os tokens do rótulo-resposta; tokens do prompt são mascarados com `-100`. A inferência usa geração determinística, com amostragem desabilitada e limite de oito tokens para a resposta.

## Dados

O conjunto foi escrito para este trabalho e contém apenas mensagens sintéticas, sem dados pessoais:

| Partição | Exemplos | Exemplos por classe |
|---|---:|---:|
| Treino | 50 | 10 |
| Validação | 10 | 2 |
| Teste | 10 | 2 |

As cinco classes têm a mesma quantidade de exemplos em cada partição. As frases foram separadas entre arquivos e não se repetem literalmente entre partições. A dimensão do conjunto é adequada para demonstrar o funcionamento técnico, mas é insuficiente para medir generalização para conversas reais ou variações linguísticas amplas.

## Protocolo de treinamento

A semente aleatória é 23. Foram testadas quatro combinações de taxa de aprendizado e épocas:

| Taxa de aprendizado | Épocas |
|---:|---:|
| `1e-4` | 2 |
| `1e-4` | 3 |
| `2e-4` | 2 |
| `2e-4` | 3 |

Os demais parâmetros permaneceram constantes: rank LoRA 8, batch por dispositivo 2, acumulação de gradiente 2, comprimento máximo de sequência 192 e avaliação de loss ao final de cada época. O conjunto de validação é utilizado para comparar as configurações. O critério de seleção é macro-F1; empates são resolvidos por maior accuracy e, em seguida, menor loss de validação. O conjunto de teste é reservado para uma avaliação final após a seleção.

## Métricas

- **Accuracy:** proporção de previsões corretas no conjunto.
- **Macro-F1:** média aritmética do F1 das cinco classes, dando o mesmo peso a cada classe.
- **Loss de validação:** média da loss de geração dos tokens da resposta rotulada; utilizada como critério secundário de seleção.
- **Matriz de confusão e métricas por classe:** permitem localizar classes com erros e observar precision, recall e F1.

Como há apenas dois exemplos de cada classe na validação e no teste, uma única previsão modifica substancialmente as métricas. As medidas devem ser lidas como resultado deste conjunto sintético específico, sem extrapolação para uso operacional.

## Execução computacional e reprodutibilidade

Execução observada: Windows, Python 3.12.6, PyTorch 2.14.1+cu130, runtime CUDA 13.0, NVIDIA GeForce RTX 3060 com 12,00 GiB de memória. Os scripts de treinamento e avaliação verificam CUDA e encerram a execução se a GPU não estiver disponível. O relatório da GPU e os arquivos CSV/JSON com resultados estão no diretório `results/`.

O modelo-base e o tokenizer são obtidos do Hugging Face Hub na primeira execução. Após o download, o fine-tuning, a avaliação e a inferência rodam localmente. A configuração de software encontra-se em `requirements.txt`; pequenas diferenças entre versões de bibliotecas, drivers e kernels podem alterar valores numéricos.
