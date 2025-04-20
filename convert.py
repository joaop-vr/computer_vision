from PIL import Image
import os
import sys

'''
    Este script converte imagens para o formato PNG
'''

# Caminho da pasta com as imagens
entrada = sys.argv[1]
saida = sys.argv[2]

# Cria a pasta de saída se não existir
os.makedirs(saida, exist_ok=True)

# Extensões que vamos converter
extensoes = ('.jpg', '.jpeg', '.webp')

# Percorre os arquivos da pasta de entrada
for nome_arquivo in os.listdir(entrada):
    if nome_arquivo.lower().endswith(extensoes):
        caminho_entrada = os.path.join(entrada, nome_arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]
        caminho_saida = os.path.join(saida, nome_base + ".png")

        # Abre e converte a imagem
        with Image.open(caminho_entrada) as img:
            img = img.convert("RGBA")  # Garante canal alpha, se necessário
            img.save(caminho_saida, "PNG")

        print(f"Convertido: {nome_arquivo} -> {nome_base}.png")
