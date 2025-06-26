# Projeto de Segmentação Semântica usando UNET
Este repositório contém a implementação de um modelo de segmentação semântica baseado na arquitetura UNET, desenvolvido para segmentar imagens urbanas do dataset Cityscapes.

## Estrutura do Projeto

<pre>
T1/
├── models/
│   └── model_original.py     # Implementação da arquitetura UNET
├── src/
│   └── customDataset.py      # Classe personalizada para carregamento do dataset
├── tools/
│   └── generate_dataset.py   # Script para preparação do dataset
├── train.py                  # Script principal de treinamento do modelo
└── test.py                   # Script para avaliação e visualização dos resultados
</pre>

## Preparação do Dataset

Antes de treinar o modelo, é necessário preparar o dataset Cityscapes. O script generate_dataset.py organiza as imagens e máscaras em diretórios de treino, validação e teste:

<pre>
python tools/generate_dataset.py <caminho_leftImg8bit> <caminho_gtFine> <diretorio_saida> <num_cidades_val> <num_cidades_teste>
</pre>

Onde:

<caminho_leftImg8bit>: Caminho para o diretório com as imagens originais
<caminho_gtFine>: Caminho para o diretório com as máscaras de segmentação
<diretorio_saida>: Diretório onde o dataset organizado será salvo
<num_cidades_val>: Número de cidades para o conjunto de validação
<num_cidades_teste>: Número de cidades para o conjunto de teste

## Treinamento do Modelo
Para treinar o modelo, execute:

<pre>
python train.py <caminho_dataset> <diretorio_saida> <dispositivo_gpu>
</pre>

Onde:

<caminho_dataset>: Caminho para o dataset preparado
<diretorio_saida>: Diretório onde o modelo treinado será salvo
<dispositivo_gpu>: ID da GPU a ser utilizada (ex: 0, 1)
Durante o treinamento, o script salva automaticamente o modelo com menor perda de validação.

## Avaliação do Modelo

Para avaliar o modelo treinado, execute:

<pre>
python test.py <caminho_dataset> <diretorio_modelo> <dispositivo_gpu>
</pre>

Onde:

<caminho_dataset>: Caminho para o dataset preparado
<diretorio_modelo>: Diretório contendo o modelo treinado (model.pth)
<dispositivo_gpu>: ID da GPU a ser utilizada

O script de teste calcula:

Acurácia de pixels (Pixel Accuracy)
IoU médio (Mean Intersection over Union)
IoU para cada classe
Adicionalmente, o script salva imagens de visualização combinando a imagem original, a máscara ground-truth e a predição do modelo.

## Arquitetura do Modelo

O modelo implementado é uma rede UNET com as seguintes características:

Encoder com quatro blocos de downsampling
Gargalo (bottleneck) com convolução dupla
Decoder com quatro blocos de upsampling
Conexões residuais (skip connections) entre encoder e decoder
Saída com 34 canais, um para cada classe possível
Cada bloco de convolução dupla contém:

Convolução 2D
Batch Normalization
ReLU
Outra convolução 2D
Batch Normalization
ReLU
Parâmetros de Configuração
O projeto utiliza os seguintes parâmetros padrão:

Altura da imagem: 1024px
Largura da imagem: 2048px
Número de épocas: 20
Tamanho do batch de treinamento: 1
Número de classes: 34
Otimizador: Adam com taxa de aprendizado de 1e-3
Função de perda: CrossEntropyLoss

## Resultados

Os resultados da avaliação são salvos em um arquivo results.txt no diretório do modelo, contendo:

Acurácia de pixels total
IoU médio
IoU para cada uma das 34 classes
As imagens de visualização são salvas no subdiretório images/ dentro do diretório do modelo.

Os resultados obtidos foram

<pre>
Pixel Accuracy: 0.8424
Mean IoU: 0.3269

Class 0 IoU = 0.0000 / Unlabeled
Class 1 IoU = 0.9029 / Ego Vegicle
Class 2 IoU = 0.9111 / Rectification Border
Class 3 IoU = 0.9995 / Out of ROI
Class 4 IoU = 0.1391 / Static
Class 5 IoU = 0.0286 / Dynamic
Class 6 IoU = 0.0006 / Ground
Class 7 IoU = 0.8529 / Road
Class 8 IoU = 0.5586 / Sidewalk
Class 9 IoU = 0.0013 / Parking
Class 10 IoU = 0.0033 / Rail Track
Class 11 IoU = 0.7670 / Building
Class 12 IoU = 0.0852 / Wall
Class 13 IoU = 0.2752 / Fence
Class 14 IoU = 0.0000 / Guard Rail
Class 15 IoU = 0.1697 / Bridge
Class 16 IoU = 0.0000 / Tunnel
Class 17 IoU = 0.4241 / Pole
Class 18 IoU = 0.0000 / Polegroup
Class 19 IoU = 0.3682 / Traffic Light
Class 20 IoU = 0.5076 / Traffic Sign
Class 21 IoU = 0.8631 / Vegetation
Class 22 IoU = 0.2977 / Terrain
Class 23 IoU = 0.8439 / Sky
Class 24 IoU = 0.3851 / Person
Class 25 IoU = 0.0435 / Rider
Class 26 IoU = 0.6338 / Car
Class 27 IoU = 0.0093 / Truck
Class 28 IoU = 0.1680 / Bus
Class 29 IoU = 0.0000 / Caravan
Class 30 IoU = 0.0000 / Trailer
Class 31 IoU = 0.0088 / Train
Class 32 IoU = 0.0735 / Motorcycle
Class 33 IoU = 0.4676 / Bicycle
</pre>
