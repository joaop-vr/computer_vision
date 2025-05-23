
# Projeto de Visão Computacional - TA3

Este repositório contém o código e recursos do Trabalho Prático 3 (TA3) da disciplina de Visão Computacional. O objetivo principal deste trabalho é aplicar transformações de perspectiva em imagens, permitindo ao usuário selecionar quatro pontos de referência para gerar uma "warp" da imagem original, além de combinar pares de imagens lado a lado com espaçamento.

---

## Estrutura de Diretórios

```
TA3/
├── combine.py
├── comparacao_input_output/   # Pasta contendo imagens de comparação entre entradas e saídas
├── input/                      # Pasta com imagens de teste de entrada
├── main.py
├── output/                     # Pasta para salvar imagens de saída geradas
└── Relatorio.pdf               # Relatório em formato PDF descrevendo a metodologia, resultados e discussões
```

- **combine.py**: Script para combinar duas imagens lado a lado, com espaçamento fixo e fundo branco.
- **main.py**: Script principal para leitura de uma imagem, seleção de quatro pontos via clique do mouse, aplicação de transformação de perspectiva (warp) e salvamento da imagem resultante.
- **comparacao_input_output/**: Contém arquivos que apresentam comparações visuais entre as imagens de entrada e suas respectivas saídas processadas.
- **input/**: Diretório onde o usuário deve colocar as imagens que deseja processar com o script `main.py`.
- **output/**: Diretório onde serão salvas as imagens geradas pelo script `main.py` (imagens "warp") ou pelo `combine.py`.
- **Relatorio.pdf**: Documento que descreve detalhadamente a metodologia, configuração dos experimentos, resultados obtidos e discussão.

---

## Dependências

Certifique-se de ter instalado:
- Python 3.6 ou superior
- OpenCV (`cv2`)
- NumPy

Para instalar via pip:
```
pip install opencv-python numpy
```

---

## Uso do `main.py`

Este script permite ao usuário carregar uma única imagem e selecionar quatro pontos de referência clicando sobre ela. A partir desses pontos, é gerada uma transformação de perspectiva (warp) com dimensão fixa de 600×800 pixels.

### Executando

1. Coloque a imagem desejada na pasta `input/` (ou forneça o caminho completo).
2. No terminal, navegue até o diretório `TA3/`.
3. Execute:
   ```
   python3 main.py <caminho_para_imagem> <caminho_para_pasta_output>
   ```
   Exemplo:
   ```
   python3 main.py input/exemplo.jpg output/
   ```

4. Após clicar nos quatro pontos na janela que será aberta, o script salvará a imagem transformada em:
   ```
   output/exemplo_warp.jpg
   ```

---

## Uso do `combine.py`

Este script recebe dois caminhos de imagens e combina-as lado a lado, com espaçamento de 30 pixels de fundo branco, salvando o resultado em um único arquivo.

### Executando

1. Coloque as duas imagens que deseja combinar em qualquer pasta (por exemplo, `input/`).
2. No terminal, navegue até o diretório `TA3/`.
3. Execute:
   ```
   python3 combine.py <caminho_para_imagem1> <caminho_para_imagem2> <caminho_para_pasta_output>
   ```
   Exemplo:
   ```
   python3 combine.py input/img1.jpg input/img2.jpg output/
   ```

4. O arquivo resultante será salvo em:
   ```
   output/img1_img2_combined.jpg
   ```

