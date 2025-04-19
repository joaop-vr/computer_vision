- peguei imagens PNG pq é o mais recomendado para processamento de imagens. Não tem compressão com perdas, então a textura da imagem fica mais fiel. E é suportado por bibliotecas como OpenCV.
- redimensionei tudo para 256x256 pq Facilita a aplicação dos filtros:
  - Se as imagens têm tamanhos diferentes, o resultado da filtragem (e a segmentação) vai ser inconsistente ou incompatível.

  - Uniformiza as janelas de análise: Quando você dividir a imagem em regiões (janelas) para extrair os vetores de textura, essas janelas precisam ser proporcionais entre as imagens.

  - Agrupamento faz mais sentido: Se você for comparar vetores entre imagens diferentes, eles precisam representar a mesma proporção do conteúdo da imagem. Tamanho igual = comparação justa.

