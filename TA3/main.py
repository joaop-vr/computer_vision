import cv2
import numpy as np
import sys
import os

def parse_arguments():
    """
    Verifica se o usuário passou os argumentos corretos.
    Retorna (image_path, output_path) se estiver tudo certo, 
    ou imprime uma mensagem de uso e encerra o programa.
    """
    if len(sys.argv) != 3:
        print("Uso: python3 main.py <caminho_imagem> <caminho_pasta_saida>")
        print("Exemplo: python3 main.py imagem.jpg resultados/")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]

    # Verifica se o arquivo de imagem existe
    if not os.path.isfile(image_path):
        print(f"Erro: não foi possível encontrar a imagem '{image_path}'.")
        sys.exit(1)

    # Garante que o output_path termine com separador de pasta
    if not output_path.endswith(os.sep):
        output_path += os.sep

    # Cria a pasta de saída, caso não exista
    os.makedirs(output_path, exist_ok=True)

    return image_path, output_path

def get_four_points(img):
    """
    Exibe a imagem e permite que o usuário selecione 4 pontos clicando com o mouse.
    Retorna lista de 4 tuplas (x, y).
    """
    points = []
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Selecione 4 pontos", img)

    cv2.namedWindow("Selecione 4 pontos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecione 4 pontos", 800, 600)
    cv2.setMouseCallback("Selecione 4 pontos", mouse_handler)
    cv2.imshow("Selecione 4 pontos", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # Se o usuário pressionar ESC, interrompe sem ter 4 pontos
        if key == 27:
            print("Seleção cancelada pelo usuário.")
            cv2.destroyAllWindows()
            sys.exit(1)
        # Tendo 4 pontos, encerra
        if len(points) == 4:
            break

    cv2.destroyAllWindows()
    return points

def warp_image(clone, points, width=600, height=800):
    """
    Aplica transformação de perspectiva nos 4 pontos para gerar imagem warp.
    Retorna a imagem warp (com tamanho width x height).
    """
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')
    src = np.array(points, dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(clone, M, (width, height))
    return warp

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

def main():
    image_path, output_path = parse_arguments()

    # Carrega imagem original
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem '{image_path}'.")
        sys.exit(1)
    clone = img.copy()

    # Seleciona 4 pontos pelo mouse
    points = get_four_points(img)
    if len(points) != 4:
        print("Número de pontos inválido. Encerrando.")
        sys.exit(1)

    # Gera a imagem warp (600x800)
    w, h = 600, 800
    warp = warp_image(clone, points, width=w, height=h)

    # Salva apenas a imagem warp
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    warp_filename = f"{output_path}{base_name}_warp.jpg"

    if cv2.imwrite(warp_filename, warp):
        print(f"Warp salvo em: {warp_filename}")
    else:
        print(f"Erro ao salvar warp em: {warp_filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
