import cv2
import numpy as np
import sys
import os

def parse_arguments():
    """
    Verifica se o usuário passou os argumentos corretos:
    3 parâmetros obrigatórios: <caminho_imagem1> <caminho_imagem2> <caminho_pasta_saida>
    """
    if len(sys.argv) != 4:
        print("Uso: python3 combine_two.py <caminho_imagem1> <caminho_imagem2> <caminho_pasta_saida>")
        print("Exemplo: python3 combine_two.py img1.jpg img2.png resultados/")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    output_path = sys.argv[3]

    # Verifica se os arquivos de imagem existem
    if not os.path.isfile(img1_path):
        print(f"Erro: não foi possível encontrar a imagem '{img1_path}'.")
        sys.exit(1)
    if not os.path.isfile(img2_path):
        print(f"Erro: não foi possível encontrar a imagem '{img2_path}'.")
        sys.exit(1)

    # Garante que o output_path termine com separador de pasta
    if not output_path.endswith(os.sep):
        output_path += os.sep

    # Cria a pasta de saída, caso não exista
    os.makedirs(output_path, exist_ok=True)

    return img1_path, img2_path, output_path

def resize_to_height(img, target_height):
    """
    Redimensiona img para que sua altura seja target_height,
    preservando proporção de aspecto. Retorna a imagem redimensionada.
    """
    orig_h, orig_w = img.shape[:2]
    scale = target_height / float(orig_h)
    new_w = int(orig_w * scale)
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized

def combine_images(img1, img2):
    """
    Combina duas imagens com a mesma altura lado a lado, deixando um espaçamento fixo entre elas.
    - Se as alturas forem diferentes, redimensiona img2 para a altura de img1.
    - Espaçamento e cor de fundo são definidos internamente.
    Retorna a imagem composta.
    """
    # Define espaçamento fixo (em pixels) e cor de fundo (BGR)
    spacing = 30
    bg_color = (255, 255, 255)  # branco; altere se quiser outra cor

    # Se as alturas forem diferentes, redimensiona img2 para a altura de img1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        img2 = resize_to_height(img2, h1)
        h2, w2 = img2.shape[:2]

    # Agora as duas imagens têm a mesma altura (h1 == h2)
    total_w = w1 + spacing + w2
    total_h = h1

    # Cria canvas preenchido com a cor de fundo
    combined = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    # Copia img1 (na posição x=0 até x=w1)
    combined[:, 0:w1] = img1

    # Copia img2 (na posição x=w1+spacing até x=w1+spacing+w2)
    combined[:, w1 + spacing : w1 + spacing + w2] = img2

    return combined

def main():
    img1_path, img2_path, output_path = parse_arguments()

    # Carrega as duas imagens
    img1 = cv2.imread(img1_path)
    if img1 is None:
        print(f"Erro ao carregar a imagem '{img1_path}'.")
        sys.exit(1)
    img2 = cv2.imread(img2_path)
    if img2 is None:
        print(f"Erro ao carregar a imagem '{img2_path}'.")
        sys.exit(1)

    # Combina as duas imagens lado a lado (com espaçamento)
    try:
        combined = combine_images(img1, img2)
    except Exception as e:
        print(f"Erro ao combinar imagens: {e}")
        sys.exit(1)

    # Define nome de arquivo de saída baseado nos nomes das imagens de entrada
    base1 = os.path.splitext(os.path.basename(img1_path))[0]
    base2 = os.path.splitext(os.path.basename(img2_path))[0]
    combined_filename = f"{output_path}{base1}_{base2}_combined.jpg"

    # Salva a imagem combinada
    if cv2.imwrite(combined_filename, combined):
        print(f"Imagem composta salva em: {combined_filename}")
    else:
        print(f"Erro ao salvar a imagem composta em: {combined_filename}")

if __name__ == "__main__":
    main()
