# Segmentação por Textura

Este projeto implementa um sistema de segmentação de imagens com base em textura, usando filtros direcionais aplicados em múltiplas escalas. O objetivo é agrupar regiões semelhantes em termos de textura, utilizando métodos clássicos de visão computacional. Para fins de teste, utilizamos um conjunto de imagens dos personagens do desenho Bob Esponja, devido à diversidade de texturas presentes em suas representações.

## Filtros Utilizados

Foram aplicados filtros com as orientações Horizontal, Vertical, Diagonal 45°, Diagonal 135° e Circular.
Os filtros são usados em três escalas: 3x3, 5x5 e 7x7.
A estrutura dos filtros deve ser definida em um arquivo .yaml, como `texture_segmentation/TA1/src/filters.yaml`, com os seguintes campos:

```
- **filtros**
    - horizontal
    - vertical
    - diagonal_45
    - diagonal_135
    - circular
- **horizontal**
    - kernels
    - filtros
- **vertical**
    - kernels
    - filtros
- **diagonal_45**
    - kernels
    - filtros
- **diagonal_135**
    - kernels
    - filtros
- **circular**
    - kernels
    - filtros
```

Cada filtro deve conter os kernels correspondentes.

## Estruturas de dados

O programa é especificado para a estrutura de diretórios criada para o **Dataset Bob Esponja** encontrado em `texture_segmentation/TA1/bob_esponja_dataset/`, e pode ser alterado e ajustado para diferentes conjuntos de dados por meio do programa `texture_segmentation/TA1/src/customDataset.py`. O sistema espera que as imagens estejam organizadas conforme a estrutura abaixo:

```
bob_esponja_dataset/
├── bob_esponja/
├── garry/
├── larry/
├── lula_molusco/
├── patrick/
├── plankton/
├── sandy/
└── siriguejo/
```
    
## Como Executar

### 1. Extração de Texturas

Use o script `extract_textures.py` para aplicar os filtros e extrair as características de textura das imagens.
Este programa possui as seguintes flags:

**Parâmetros:**
- `--dataset`: Caminho para o diretório do dataset.

- `--output`: Diretório de saída dos arquivos gerados. Não é necessário ser um diretório existente.

- `--grayscale`:  Se `true`, converte a imagem para tons de cinza.

- `--filters_config`: Caminho para o arquivo `.yaml` com os filtros. Encontrado em src/

- `--resize_size`: Tamanho para redimensionar as imagens.

- `--window_size`: Tamanho da janela para aplicar os filtros.

- `--split_factor`: Porcentagem de imagens usadas para treinamento. Não aplicado na segmentação por textura (qualquer valor entre 0 e 1 serve).

#### Exemplo de linha de comando:

```
python3 extract_textures.py --dataset ~/Pictures/bob_esponja_dataset/ --output ./output --grayscale false --filters_config src/filters.yaml --resize_size 512 --window_size 32 --split_factor 0.7
```

### 2. Segmentação por Textura

Utilize o script segmentation.py para segmentar as imagens com base nas texturas extraídas.

**Parâmetros:**

- `--features_dir`: Caminho para os arquivos de características gerados anteriormente.

- `--output`: Diretório onde serão salvas as imagens segmentadas.

- `--limiar`: Valor máximo de diferença para considerar duas regiões como pertencentes à mesma classe.

#### Exemplo de linha de comando:

```
python3 segmentation.py --features_dir ./output --output ./output_segmented --limiar 0.2
```

### 3. Classificação por Texutra (Extra)

Utilize o sript classify.py para classificar as imagens com base nas texturas extraídas. split_factor é aplicado aqui. Além disso, este código aceita o dataset tiny-imagenet, que conta com 100 mil imagens para treinamento entre 200 classes e, para cada uma delas, 50 imagens de teste.

**Parâmetros:**

- `--features_dir`: Caminho para os arquivos de características gerados anteriormente.

- `--knn_factor`: Número de vizinhos (k) usado para decidir a classe da imagem 

#### Exemplo de linha de comando:

```
python3 classify.py --features_dir ./output --knn_factor 1
```

## Autores
- Heloisa Benedet Mendes, GRR20221248
- João Pedro Vicente Ramalho, GRR20224169
- Luan Marko Kujavski, GRR20221236
