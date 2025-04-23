# Segmentação por Textura

Neste trabalho criamos um sistema para segmentar imagens por textura.
Para isso, usamos um conjunto de imagens dos personagens do desenho Bob Esponja. A escolha se deu pela grande variedade de texturas presente nos personagens, assim é possível avaliar o programa com maior amplitude de aplicação.
Aqui são aplicados filtros horizontal, vertical, diagonal 45°, diagonal 135° e circular para os tamanhos 3, 5 e 7.

## Estruturas de dados

O programa é especificado para a estrutura de diretórios criada para o **Dataset Bob Esponja**, e pode ser alterado e ajustado para diferentes conjuntos de dados por meio do programa `texture_segmentation/TA1/src/customDataset.py`. O Dataset segue a seguinte estrutura de diretórios:
- ***bob_esponja_dataset***
    - **eugene_h_krabs**
        - imagens
    - **karen_plankton**
        - imagens
    - **patrick_star**
        - imagens
    - **sandy_cheeks**
        - imagens
    - **spongebob_squarepants**
        - imagens
    - **gary_the_snail**
        - imagens
    - **mrs_puff**
        - imagens
    - **pearl_krabs**
        - imagens
    - **sheldon_j_plankton**
        - imagens
    - **squidward_tentacles**
        - imagens

Além disso, depende de um arquivo que descreva os filtros a serem utilizados. Esse arquivo deve seguir o padrão estabelecido no trabalho pelo arquivo `texture_segmentation/TA1/src/filters.yaml`.
- **filtros**
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

## Como rodar o programa

Para rodar o programa corretamente, primeiro é preciso extrair as texturas das imagens, para isso é necessário o programa `extract_textures.py`.
Este programa possui as seguintes flags:

- --dataset
Indica o diretório onde estão as imagens a serem utilizadas pelo programa. Neste caso, usamos o dataset com os personagens de Bob Esponja que pode ser encontrado em `texture_segmentation/TA1/bob_esponja_dataset/`.

- --output
Indica o diretório onde devem ser armazenados os arquivos gerados pelo programa. Não é necessário ser um diretório pré-existente

- --grayscale

- --filters_config
Caminho para o arquivo .yaml com os filtros a serem utilizados. 

- --resize_size

- --window_size
Tamanho em bytes da janela a ser utilizada ao aplicar os filtros.

- --split_factor

### Exemplo de linha de comando:

```
python3 extract_textures.py --dataset ~/Pictures/bob_esponja_dataset/ --output ./output --grayscale false --filters_config src/filters.yaml --resize_size 512 --window_size 32 --split_factor 0.4
```

Então, é necessário segmentar as imagens com base nas texturas extraídas, para isso use o programa `segmentation.py`, que possui as flags:
- --features_dir
Indica o diretório onde estão as características extraídas, ou seja, os arquivos de saída gerados pelo programa anterior.

- --output
Diretório destino das imagens segmentadas.

- --limiar
Indica o valor máximo de diferença para duas janelas pertencerem à mesma classe.

### Exemplor de linha de comando:

```
python3 segmentation.py --features_dir ./output --output ./output_segmented --limiar 0.2
```

## Autores
- Heloisa Benedet Mendes, GRR20221248
- João Pedro Vicente Ramalho, GRR20224169
- Luan Marko Kujavski, GRR20221236
